import math
import os
from pathlib import Path
import random
import io

from PIL import Image
from absl import flags
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

flags.DEFINE_integer("seed", 42, "fixed seed to apply to all rng entrypoints")
FLAGS = flags.FLAGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_non_empty_file(path):
    if isinstance(path, str):
        path = Path(path)
    return path.is_file() and path.stat().st_size != 0


def set_seeds(seed=42, fully_deterministic=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(model_path, map_location=device):
    model_path = os.path.realpath(model_path)  # Resolve any symlinks
    return torch.load(model_path, map_location=map_location)


class FixedRandomState:
    def __init__(self, seed=0):
        self.seed = seed

    def __enter__(self):
        # Copy current state
        self.random_state = RandomStateCache()

        # Overwrite seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def __exit__(self, *args):
        self.random_state.restore()


class RandomStateCache:
    def __init__(self):
        self.store()

    def store(self):
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_state = torch.cuda.get_rng_state_all()

    def restore(self):
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)
        torch.random.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_state)


class RAdam(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def reset_step_buffer(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Step buffer must be reset when loading RAdam checkpoint
        # to prevent loss spike
        self.reset_step_buffer()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                beta2_t = beta2 ** state["step"]
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)
                    step_size = (
                        group["lr"]
                        * math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        )
                        / (1 - beta1 ** state["step"])
                    )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)
                    step_size = group["lr"] / (1 - beta1 ** state["step"])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class BatchNorm(torch.nn.Module):
    """
    nn.Module to handle turning batch norm on or off within the model
    """

    def __init__(self, num_features, batch_norm_on):
        super().__init__()

        self.num_features = num_features
        self.batch_norm_on = batch_norm_on

        if batch_norm_on:
            self.bn = torch.nn.BatchNorm1d(num_features)
        else:
            self.bn = torch.nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x


class Permute(nn.Module):
    """
    nn.Module to switch sequence length and feature dimention for switching between
    convolutional and linear layers
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class GlobalNormalization(torch.nn.Module):
    """
    nn.Module to track and normalize input variables, calculates running estimates of data
    statistics during training time.
    Optional scale parameter to fix standard deviation of inputs to 1
    Implementation details:
    "https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
    """

    def __init__(self, feature_dim, scale=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer("running_ave", torch.zeros(1, 1, self.feature_dim))
        self.register_buffer("total_frames_seen", torch.Tensor([0]))
        self.scale = scale
        self.register_buffer("running_sq_diff", torch.zeros(1, 1, self.feature_dim))

    def forward(self, inputs):
        # disabling pylint on a couple of lines as it is bugged at present:
        # TODO: re-enable when pylint is fixed
        # https://github.com/PyCQA/pylint/issues/2315
        # pylint: disable=E0203
        # Check input is of correct shape and matches feature size
        if len(inputs.shape) != 3 or inputs.shape[2] != self.feature_dim:
            raise ValueError(
                f"""Inputs do not match required shape [batch_size, window_size, feature_dim], """
                f"""(expecting feature dim {self.feature_dim}), got {inputs.shape}"""
            )
        if self.training:
            self.update_stats(inputs)

        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = (inputs - self.running_ave) / std
        else:
            inputs = inputs - self.running_ave

        return inputs

    def unnorm(self, inputs):
        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = inputs * std + self.running_ave
        else:
            inputs = inputs + self.running_ave

        return inputs

    def update_stats(self, inputs):
        inputs_for_stats = inputs.detach()
        # Update running estimates of statistics
        frames_in_input = inputs.shape[0] * inputs.shape[1]
        updated_running_ave = (
            self.running_ave * self.total_frames_seen
            + inputs_for_stats.sum(dim=(0, 1), keepdim=True)
        ) / (self.total_frames_seen + frames_in_input)

        if self.scale:
            # Update the sum of the squared differences between inputs and mean
            self.running_sq_diff = self.running_sq_diff + (
                (inputs_for_stats - self.running_ave) * (inputs_for_stats - updated_running_ave)
            ).sum(dim=(0, 1), keepdim=True)

        self.running_ave = updated_running_ave
        self.total_frames_seen = self.total_frames_seen + frames_in_input


def wav_to_float(x):
    """
    Input in range -2**15, 2**15 (or what is determined from dtype)
    Output in range -1, 1
    """
    assert x.dtype == torch.int16, f"got {x.dtype}"
    max_value = torch.iinfo(torch.int16).max
    min_value = torch.iinfo(torch.int16).min
    if not x.is_floating_point():
        x = x.to(torch.float)
    x = x - min_value
    x = x / ((max_value - min_value) / 2.0)
    x = x - 1.0
    return x


def float_to_wav(x):
    """
    Input in range -1, 1
    Output in range -2**15, 2**15 (or what is determined from dtype)
    """
    assert x.dtype == torch.float
    max_value = torch.iinfo(torch.int16).max
    min_value = torch.iinfo(torch.int16).min

    x = x + 1.0
    x = x * (max_value - min_value) / 2.0
    x = x + min_value
    x = x.to(torch.int16)
    return x


def mu_law_encoding(x, mu=255.0):
    """
    Input in range -2**15, 2*15 (or what is determined from dtype)
    Output is in range -1, 1 on mu law scale
    """
    x = wav_to_float(x)
    mu = torch.tensor(mu, dtype=x.dtype, device=x.device)
    x_mu = torch.sign(x) * (torch.log1p(mu * torch.abs(x)) / torch.log1p(mu))
    return x_mu


def mu_law_decoding(x_mu, mu=255.0):
    """
    Input is in range -1, 1 on mu law scale
    Output in range -2**15, 2*15 (or what is determined from dtype)
    """
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype, device=x_mu.device)
    x = torch.sign(x_mu) * (1 / mu) * (((1 + mu) ** torch.abs(x_mu)) - 1)
    x = float_to_wav(x)
    return x


def fig2tensor(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    x = np.array(img)
    x = torch.Tensor(x).permute(2, 0, 1) / 255.0
    return x
