from pathlib import Path

from collections import namedtuple
import torch
import warnings
from absl import flags, app

from dataloader.audio import AudioDataset, AudioDataLoader
from cpc.model import NoCPC
from util import load_model, set_seeds, device

Prediction = namedtuple("Prediction", ["label", "start", "end"])
DEFAULT_LABEL = 0

flags.DEFINE_string("eval_file_path", None, "path to file to run on")
flags.DEFINE_string("output_dir", None, "file path to dir to write output to")
flags.DEFINE_string("cpc_path", None, "path to initial backbone weights to load")
flags.DEFINE_string("model_path", None, "trained model")
flags.DEFINE_integer("window_size", 1024, "num frames to push into model at once")
flags.DEFINE_boolean("pad_input", False, "right pad inputs with zeros up to window size")

flags.mark_flag_as_required("eval_file_path")
flags.mark_flag_as_required("model_path")
flags.mark_flag_as_required("output_dir")
FLAGS = flags.FLAGS


def preds_to_output(pred, num_inputs, input_freq_hz, start_time_s):
    """
    Takes an array of predictions from the decode and maps it to real time predictions
    Args:
        pred (obj): raw output from the model
        num_inputs (int): how many input frames were fed to cpc
        input_freq_hz (int): frames per second provided to cpc
        start_time_s (float): offset to apply to prediction timings
    Returns:
        (list): outputs, each is a Prediction namedtuple
        (float): end time of the batch, could be used as start time of the next one
    """
    num_preds = pred.shape[0]
    total_duration_s = float(num_inputs) / input_freq_hz
    pred_duration_s = total_duration_s / num_preds
    outputs = []
    for idx in range(num_preds):
        outputs.append(
            Prediction(
                pred[idx].item(),
                start_time_s + idx * pred_duration_s,
                start_time_s + (idx + 1) * pred_duration_s,
            )
        )
    return outputs, start_time_s + num_preds * pred_duration_s


def main(unused_argv):
    # create output dirs
    output_dir = Path(FLAGS.output_dir)
    Path.mkdir(output_dir, exist_ok=True)

    if FLAGS.cpc_path is not None:
        cpc = load_model(FLAGS.cpc_path).eval().to(device)
    else:
        cpc = NoCPC().eval().to(device)
    model = load_model(FLAGS.model_path).eval().to(device)

    dataset = AudioDataset(FLAGS.eval_file_path, train=False)
    dataloader = AudioDataLoader(
        dataset,
        window_size=None,
        batch_size=1,
        feature_transform=cpc.data_class,
        num_workers=8,
        shuffle=False,
    )

    set_seeds()
    # Need the enumeration to ensure unique files
    for i, batch in enumerate(dataloader):
        data = batch["data"].to(device)
        cpc.reset_state()

        preds = []
        prev_end_s = 0.0
        windows = torch.split(data, FLAGS.window_size, dim=1)
        for window in windows:
            with torch.no_grad():
                features = cpc(window)
                pred = model(features).argmax(dim=2).squeeze(dim=0)

            outputs, prev_end_s = preds_to_output(
                pred,
                window.shape[1],
                dataloader.sampling_rate,
                prev_end_s,
            )
            preds.extend(outputs)

        filename = Path(batch["files"][0])
        with open(str(output_dir / filename.name) + "_" + str(i), "w") as out_f:
            for pred in preds:
                out_f.write("{:.3f} {:.3f} {}\n".format(pred.start, pred.end, pred.label))

        with open(output_dir / "score.dbl", "a") as dbl_fh:
            dbl_fh.write(str(output_dir / filename.name) + "_" + str(i) + "\n")


if __name__ == "__main__":
    app.run(main)
