from pathlib import Path

from collections import namedtuple
import torch
import warnings
from absl import flags, app

from emotion_id.dataset import parse_emotion_dbl
from dataloader.streaming import DblStream, DblSampler
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


def decode_emotions_from_file(filename, cpc, model, window_size):
    datastream = DblStream(
        DblSampler([filename], loop_data=False),
        single_file_stream_class=cpc.data_class,
        window_size=window_size,
        pad_final=FLAGS.pad_input,
    )
    preds = []
    prev_end_s = 0.0
    for batch in datastream:
        data = torch.tensor(batch["data"]).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                features = cpc(data)
                pred = model(features).argmax(dim=2).squeeze(dim=0)
        except RuntimeError as e:
            # this catches when we don't have enough samples to complete a batch
            # instead we output default label for that time duration
            if "Kernel size can't be greater than actual input size" in e.args[0]:
                pred = torch.ones([1], dtype=torch.int32) * DEFAULT_LABEL
                warnings.warn(e.args[0])
            else:
                raise e

        outputs, prev_end_s = preds_to_output(
            pred,
            batch["data"].shape[0],
            datastream.single_file_stream.SAMPLING_RATE_HZ,
            prev_end_s,
        )
        preds.extend(outputs)
    return preds


def main(unused_argv):
    # create output dirs
    output_dir = Path(FLAGS.output_dir)
    Path.mkdir(output_dir, exist_ok=True)

    decode_dbl = parse_emotion_dbl(FLAGS.eval_file_path)

    if FLAGS.cpc_path is not None:
        cpc = load_model(FLAGS.cpc_path).eval().to(device)
    else:
        cpc = NoCPC().eval().to(device)
    model = load_model(FLAGS.model_path).eval().to(device)

    set_seeds()
    # Need the enumeration to ensure unique files
    for i, dbl_entry in enumerate(decode_dbl):
        filename = Path(dbl_entry.audio_path)
        preds = decode_emotions_from_file(filename.as_posix(), cpc, model, FLAGS.window_size)

        with open(str(output_dir / filename.name) + "_" + str(i), "w") as out_f:
            for pred in preds:
                out_f.write("{:.3f} {:.3f} {}\n".format(pred.start, pred.end, pred.label))

        with open(output_dir / "score.dbl", "a") as dbl_fh:
            dbl_fh.write(str(output_dir / filename.name) + "_" + str(i) + "\n")


if __name__ == "__main__":
    app.run(main)
