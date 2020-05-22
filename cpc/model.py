import torch
from torch import nn
from dataloader.streaming import RawStream, FbankStream
from util import mu_law_encoding, BatchNorm, device


class FeatureEncoder(nn.Module):
    """
    Fully connected 3 layer Encoder that operates on downsampled features
    """

    def __init__(self, feat_dim=80, hidden_size=512, batch_norm_on=True):

        super().__init__()

        self.blocks = nn.Sequential(
            nn.Linear(feat_dim, hidden_size),
            BatchNorm(hidden_size, batch_norm_on),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            BatchNorm(hidden_size, batch_norm_on),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            BatchNorm(hidden_size, batch_norm_on),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.blocks(x)


class RawEncoder(nn.Module):
    """
    Encoder that downsamples raw audio by a factor of 160
    """

    def __init__(self, hidden_size=512):

        super().__init__()

        self.blocks = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_size, hidden_size, kernel_size=8, stride=4, padding=2, bias=False
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_size, hidden_size, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_size, hidden_size, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_size, hidden_size, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.blocks(x)


class CPCModel(nn.Module):
    """
    CPCModel: base model from the paper: 'Representation Learning with Contrastive
    Predictive Coding'
    """

    def __init__(
        self,
        features_in,
        timestep,
        batch_size,
        window_size,
        hidden_size=512,
        out_size=256,
        no_gru_layers=1,
    ):

        super(CPCModel, self).__init__()
        self.features_in = features_in
        self.batch_size = batch_size
        self.timestep = timestep
        self.hidden_size = hidden_size
        self.out_size = out_size
        if features_in == "raw":
            self.encoder = RawEncoder(hidden_size)
            self.seq_len = window_size // 160
        elif features_in == "fbank":
            self.encoder = FeatureEncoder(feat_dim=80, hidden_size=hidden_size)
            self.seq_len = window_size
        self.gru = nn.GRU(
            hidden_size,
            out_size,
            num_layers=no_gru_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.Wk = nn.ModuleList(
            [nn.Linear(out_size, hidden_size) for i in range(timestep)]
        )
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if "weight" in p:
                    nn.init.kaiming_normal_(
                        self.gru.__getattr__(p), mode="fan_out", nonlinearity="relu"
                    )

        self.apply(_weights_init)

    def forward(self, x, hidden):
        # raw: N*C*L, e.g. 8*1*20480, fbank: N*L*C, e.g. 8*128*80
        z = self.encoder(x)

        # z: N*L*C, e.g. 8*128*512
        # reshape to N*L*C for GRU if raw input
        if self.features_in == "raw":
            z = z.transpose(1, 2)

        # pass z into gru to get c
        if hidden is not None:
            hidden = hidden.detach()
        c, hidden = self.gru(z, hidden)  # e.g. 8*128*256

        return z, c, hidden

    def get_cpc_loss(self, z, c, t):
        # take representation at time t
        c_t = c[:, t, :]  # e.g. 8*256

        # infer z_{t+k} for each step in the future: c_t*Wk, where 1 <= k <= timestep
        pred = torch.stack(
            [self.Wk[k](c_t) for k in range(self.timestep)]
        )  # e.g. 12x8*512

        # pick the target z values timestep number of samples after t
        z_samples = z[:, t + 1 : t + 1 + self.timestep, :].permute(
            1, 0, 2
        )  # e.g. 12*8*512

        nce = 0
        correct = 0
        for k in range(self.timestep):
            # calculate the log density ratio: log(f_k) = z_{t+k}^T * W_k * c_t
            log_density_ratio = torch.mm(
                z_samples[k], pred[k].transpose(0, 1)
            )  # e.g. size 8*8

            # positive samples will be from the same batch
            # therefore, correct if highest probability is in the diagonal
            positive_batch_pred = torch.argmax(self.softmax(log_density_ratio), dim=0)
            positive_batch_actual = torch.arange(0, self.batch_size).to(device)
            correct = (
                correct
                + torch.sum(torch.eq(positive_batch_pred, positive_batch_actual)).item()
            )

            # calculate NCE loss
            nce = nce + torch.sum(torch.diag(self.lsoftmax(log_density_ratio)))

        # average over timestep and batch
        nce = nce / (-1.0 * self.batch_size * self.timestep)
        accuracy = correct / (1.0 * self.batch_size * self.timestep)
        return accuracy, nce

    def predict(self, x, hidden):
        # raw: N*C*L, e.g. 8*1*20480, fbank: N*L*C, e.g. 8*128*80
        z = self.encoder(x)
        # z: N*L*C, e.g. 8*128*512
        # reshape to N*L*C for GRU if raw input
        if self.features_in == "raw":
            z = z.transpose(1, 2)
        output, hidden = self.gru(z, hidden)  # e.g. 8*128*256
        return output, hidden


class TrainedCPC(nn.Module):
    """
    Body wrapper class to wrap a trained cpc body for benchmarking
    """

    def __init__(self, cpc_model):
        self.features_in = cpc_model.features_in
        if self.features_in == "raw":
            data_class = RawStream
        elif self.features_in == "fbank":
            data_class = FbankStream

        super().__init__()
        self.feat_dim = cpc_model.hidden_size
        self.data_class = data_class

        self.cpc_model = cpc_model
        self.hidden_state = None

    def forward(self, x):
        with torch.no_grad():
            if self.features_in == "raw":
                x = mu_law_encoding(x).unsqueeze(1)

            x = self.cpc_model.encoder(x)

            if self.features_in == "raw":
                x = x.transpose(1, 2)

            x, self.hidden_state = self.cpc_model.gru(x, self.hidden_state)
        return x

    def stash_state(self):
        if self.hidden_state is None:
            self.state_stash = None
        else:
            self.state_stash = self.hidden_state.detach().clone()
        self.reset_state()

    def reset_state(self):
        self.hidden_state = None

    def pop_state(self):
        self.hidden_state = self.state_stash
        self.state_stash = None
