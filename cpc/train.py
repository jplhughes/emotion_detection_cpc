import numpy as np

import torch
from absl import flags, logging, app
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from cpc.model import CPCModel, TrainedCPC
from dataloader.audio import AudioDataset, AudioDataLoader
from util import (
    set_seeds,
    FixedRandomState,
    device,
    RAdam,
    mu_law_encoding,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("train_data", None, "path to train dbl")
flags.DEFINE_string("val_data", None, "path to validation dbl")
flags.DEFINE_string("expdir", None, "directory to write all experiment data to")
flags.DEFINE_string("model_out", None, "path to where to save trained model")
flags.DEFINE_string("features_in", "raw", "type of features to run cpc on")
flags.DEFINE_float("lr", 4e-4, "learning rate")
flags.DEFINE_integer("steps", 100000, "number of steps to take over streaming dataset")
flags.DEFINE_integer("val_steps", None, "number of steps to take in validation")
flags.DEFINE_integer("batch_size", 32, "batch size, num parallel streams to train on at once")
flags.DEFINE_integer("window_size", 20480, "num frames to push into model at once")
flags.DEFINE_integer("timestep", 12, "the number of frames ahead to predict")
flags.DEFINE_integer("hidden_size", 512, "the hidden layer size of the encoder")
flags.DEFINE_integer("out_size", 256, "the hidden layer size of the gru")
flags.DEFINE_integer("no_gru_layers", 1, "the number of layers in the gru")

flags.DEFINE_integer("val_every", None, "how often to perform validation")
flags.DEFINE_integer("save_every", None, "save every n steps")
flags.DEFINE_integer("log_every", 10, "append to log file every n steps")
flags.DEFINE_integer("log_tb_every", 50, "save tb scalars every n steps")
flags.DEFINE_integer("num_workers", 8, "number of workers for dataloader")

flags.mark_flag_as_required("train_data")
flags.mark_flag_as_required("val_data")
flags.mark_flag_as_required("expdir")


def validation(model, val_dataloader, val_steps, features_in):
    model.eval()
    losses = []
    accuracies = []
    hidden = None
    with FixedRandomState(42):
        for step, val_batch in enumerate(val_dataloader):
            data = val_batch["data"].to(device)
            if features_in == "raw":
                data = mu_law_encoding(data.unsqueeze(1))
            with torch.no_grad():

                z, c, hidden = model(data, hidden)

                possible_t_range = int(model.seq_len - model.timestep)
                for t in range(0, possible_t_range, model.timestep):
                    acc, nce_loss = model.get_cpc_loss(z, c, t)
                    losses.append(nce_loss.item())
                    accuracies.append(acc)
            if step >= val_steps:
                break

    loss_mean = np.mean(losses)
    acc_mean = np.mean(accuracies)

    model.train()
    return acc_mean, loss_mean


def save_models(model, model_out, ext, data=None):
    torch.save(model, model_out + ext)

    trained_model = TrainedCPC(model).to(device)
    torch.save(trained_model, model_out + ext + ".trained")

    # ensure data goes through without errors
    if data is not None:
        _ = trained_model(data)


def train(model, optimizer, scheduler, train_dataloader, val_dataloader, FLAGS):
    model.train()

    tb_logger = SummaryWriter(FLAGS.expdir, flush_secs=10)
    best_loss = np.inf
    hidden = None
    if FLAGS.features_in == "raw":
        sampling_rate = 16000
    else:
        sampling_rate = 100

    for step, train_batch in enumerate(train_dataloader):
        data = train_batch["data"].to(device)
        if FLAGS.features_in == "raw":
            data = mu_law_encoding(data.unsqueeze(1))

        z, c, hidden = model(data, hidden)

        loss = 0
        accuracy = 0
        possible_t_range = int(model.seq_len - model.timestep)
        for num, t in enumerate(range(0, possible_t_range, model.timestep)):
            acc, nce_loss = model.get_cpc_loss(z, c, t)
            loss = loss + nce_loss
            accuracy = accuracy + acc

        loss = loss / (num + 1)
        accuracy = accuracy / (num + 1)

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        global_step = (step + 1) * (num + 1)

        if step % FLAGS.log_every == 0:
            audio_seen_hr = (
                (step + 1) * FLAGS.batch_size * FLAGS.window_size / (sampling_rate * 3600)
            )
            logging.info(
                (
                    f"step {step}, global_step {global_step}, "
                    f"loss {loss.item():.6f}, acc {accuracy:.4f}, "
                    f"lr {scheduler.get_lr()[0]:.6f}, "
                    f"seen {audio_seen_hr:.3f} hours"
                )
            )

        if step % FLAGS.log_tb_every == 0:
            tb_logger.add_scalar("train/loss", loss.item(), global_step)
            tb_logger.add_scalar("train/acc", accuracy, global_step)
            tb_logger.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
            tb_logger.add_scalar("train/audio_seen", audio_seen_hr, global_step)

        if step % FLAGS.val_every == 0 and step != 0:
            val_acc, val_loss = validation(
                model, val_dataloader, FLAGS.val_steps, FLAGS.features_in
            )
            logging.info(f"val loss {val_loss:.6f}, val acc {val_acc:.4f}")
            tb_logger.add_scalar("valid/loss", val_loss, global_step)
            tb_logger.add_scalar("valid/acc", val_acc, global_step)

            if val_loss < best_loss:
                ext = ".bestloss"
                test_data = train_batch["data"].to(device)
                save_models(model, FLAGS.model_out, ext, data=test_data)
                logging.info("New best loss validation model saved.")
                best_loss = val_loss

        if step % FLAGS.save_every == 0 and step != 0:
            ext = ".step" + str(step)
            torch.save(model, FLAGS.model_out + ext)

        if step >= FLAGS.steps:
            ext = ""
            save_models(model, FLAGS.model_out, ext)
            break


def run_cpc(unused_argv):

    # setup logging
    set_seeds(FLAGS.seed)

    # initialise unset flags
    if not FLAGS.model_out:
        FLAGS.model_out = FLAGS.expdir + "/model.pt"
    if not FLAGS.val_every:
        FLAGS.val_every = max(100, FLAGS.steps // 50)
    if not FLAGS.val_steps:
        FLAGS.val_steps = max(20, FLAGS.steps // FLAGS.batch_size // 50)
    if not FLAGS.save_every:
        FLAGS.save_every = FLAGS.val_every
    logging.info(f"model_out {FLAGS.model_out}")
    logging.info(f"steps {FLAGS.steps}")
    logging.info(f"val_steps {FLAGS.val_steps}")
    logging.info(f"log_every {FLAGS.log_every}")
    logging.info(f"log_tb_every {FLAGS.log_tb_every}")
    logging.info(f"val_every {FLAGS.val_every}")
    logging.info(f"save_every {FLAGS.save_every}")

    # model and optimization
    model = CPCModel(
        FLAGS.features_in,
        FLAGS.timestep,
        FLAGS.batch_size,
        FLAGS.window_size,
        FLAGS.hidden_size,
        FLAGS.out_size,
        FLAGS.no_gru_layers,
    ).to(device)
    logging.info(f"param count {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = RAdam(model.parameters(), lr=4e-4)
    scheduler = CosineAnnealingLR(optimizer, FLAGS.steps, eta_min=1e-6)

    # dataloaders
    train_dataset = AudioDataset(FLAGS.train_data)
    train_dataloader = AudioDataLoader(
        train_dataset,
        window_size=FLAGS.window_size,
        batch_size=FLAGS.batch_size,
        feature_transform=FLAGS.features_in,
        num_workers=FLAGS.num_workers,
    )

    val_dataset = AudioDataset(FLAGS.val_data)
    val_dataloader = AudioDataLoader(
        val_dataset,
        window_size=FLAGS.window_size,
        batch_size=FLAGS.batch_size,
        feature_transform=FLAGS.features_in,
        num_workers=FLAGS.num_workers,
    )

    # start training
    train(model, optimizer, scheduler, train_dataloader, val_dataloader, FLAGS)


if __name__ == "__main__":
    app.run(run_cpc)
