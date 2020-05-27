import json
from absl import flags, app

from pyannote.core import Annotation, Segment
from pyannote.metrics.identification import IdentificationErrorRate
from emotion_id.dataset import get_emotion_to_id_mapping

# This scoring assumes only one emotion per file in the reference
flags.DEFINE_string("ref", None, "Path to reference emotions")
flags.DEFINE_string("pred", None, "Path to predicted emotions")
flags.DEFINE_string("emotion_set_path", None, "path to emotion set")
flags.DEFINE_boolean("single", False, "Score only a single file")
FLAGS = flags.FLAGS
flags.mark_flag_as_required("ref")
flags.mark_flag_as_required("pred")
flags.mark_flag_as_required("emotion_set_path")


def emotion_timings_to_annotation(input_emotions):
    """
    Converts list of emotions and timings into a pyannote segmentation
    """
    result = Annotation()
    for event in input_emotions:
        start, end, emotion = event
        result[Segment(start, end)] = emotion
    return result


def single_file_stats(ref_annotation, pred_annotation):
    """
    :param ref_annotation: reference annotation in pyannote format
    :param pred_annotation: prediction annotation in pyannote format
    :return: a dict containing results
    """
    IER = IdentificationErrorRate()
    components = IER.compute_components(
        ref_annotation, pred_annotation, uem=ref_annotation.get_timeline()
    )
    hits = components["correct"]
    misses = components["total"] - hits
    results = {"overall": {"hits": hits, "misses": misses}}
    return results


def overall_stats(ref, pred, emotion_set_path, single=False):
    """
    :param ref: Path to file(s) containing reference emotion timings
    :param pred: Path to file(s) containing predicted emotion timings
    :param emotion_set_path: Path to emotion set
    :param single: If true, paths are to single files (assume they are filelists otherwise)
    :return:
    """
    emotion2id = get_emotion_to_id_mapping(emotion_set_path)

    # prepare files
    if single:
        ref_emotions = [ref]
        pred_files = [pred]
    else:
        with open(ref) as inf:
            ref_emotions = [line.strip().split()[1] for line in inf]
        with open(pred) as inf:
            pred_files = [line.strip() for line in inf]

    # loop over files, gathering results
    results = {}
    for ref_emotion, pred_file in zip(ref_emotions, pred_files):
        emotion_id = emotion2id[ref_emotion]
        pred_emotions = []
        with open(pred_file) as in_f:
            for line in in_f:
                items = line.strip().split()
                pred_emotions.append((float(items[0]), float(items[1]), int(items[2])))
        ref_emotions = [(0.0, pred_emotions[-1][1], emotion_id)]

        ref_annotation = emotion_timings_to_annotation(ref_emotions)
        pred_annotation = emotion_timings_to_annotation(pred_emotions)

        nu_results = single_file_stats(ref_annotation, pred_annotation)
        if len(results) == 0:
            results = nu_results
        else:
            for sublevel in nu_results:
                for item in results[sublevel]:
                    results[sublevel][item] += nu_results[sublevel][item]
    for sublevel in results:
        results[sublevel]["accuracy"] = float(results[sublevel]["hits"]) / (
            results[sublevel]["hits"] + results[sublevel]["misses"]
        )
    return results


def score(unused_argv):
    results = overall_stats(FLAGS.ref, FLAGS.pred, FLAGS.emotion_set_path, FLAGS.single)
    print(json.dumps(results))


if __name__ == "__main__":
    app.run(score)
