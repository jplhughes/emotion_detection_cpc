import json
import os
import numpy as np
from absl import flags, app
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
from emotion_id.dataset import get_emotion_to_id_mapping

# This scoring assumes only one emotion per file in the reference
flags.DEFINE_string("ref", None, "Path to reference emotions")
flags.DEFINE_string("pred", None, "Path to predicted emotions")
flags.DEFINE_string("output", None, "Path to predicted emotions")
flags.DEFINE_string("emotion_set_path", None, "path to emotion set")
flags.DEFINE_boolean("single", False, "Score only a single file")
flags.DEFINE_list("actor_ignore", [], "actors to ignore in scoring")
FLAGS = flags.FLAGS
flags.mark_flag_as_required("ref")
flags.mark_flag_as_required("pred")
flags.mark_flag_as_required("output")
flags.mark_flag_as_required("emotion_set_path")


def get_stats(refs, preds, emotion2id, acc_type):
    """
    :param refs: List of reference emotion classes
    :param preds: List of predicted emotion classes
    :param acc_type: String used in confusion matrix png name
    """
    cm = confusion_matrix(refs, preds)
    f1_scores = f1_score(refs, preds, average=None)
    precisions, recalls, _, _ = precision_recall_fscore_support(
        refs, preds, average=None
    )
    acc = accuracy_score(refs, preds)
    print(f"Frame-wise accuracy: {acc:.4f}")
    results = {
        "accuracy": acc,
    }

    for f1, precision, recall, emotion in zip(
        f1_scores, precisions, recalls, emotion2id.keys()
    ):
        print(f"{emotion}, precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
        results[emotion] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    print(cm)
    np.savetxt(f"{FLAGS.output}/confusion_matrix_{acc_type}.txt", cm)
    np.save(f"{FLAGS.output}/confusion_matrix_{acc_type}.npy", cm)

    ax = plt.subplot()
    sns.heatmap(cm, annot=False, ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.savefig(f"{FLAGS.output}/confusion_matrix_{acc_type}.png")
    plt.clf()

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

    all_preds = []
    all_refs = []
    file_preds = []
    file_refs = []
    for ref_emotion, pred_file in zip(ref_emotions, pred_files):
        actor = pred_file.split("-")[-1].split(".")[0]
        if actor in FLAGS.actor_ignore:
            continue
        emotion_id = emotion2id[ref_emotion]
        frame_preds = []
        with open(pred_file) as in_f:
            for line in in_f:
                frame_pred = int(line.strip().split()[2])
                frame_preds.append(frame_pred)
                all_refs.append(emotion_id)
        all_preds.extend(frame_preds)
        counts = np.bincount(np.array(frame_preds))
        file_preds.append(np.argmax(counts))
        file_refs.append(emotion_id)

    results["frame_wise"] = get_stats(all_refs, all_preds, emotion2id, "frame_wise")
    results["file_wise"] = get_stats(file_refs, file_preds, emotion2id, "file_wise")

    return results


def score(unused_argv):
    os.makedirs(FLAGS.output, exist_ok=True)
    results = overall_stats(FLAGS.ref, FLAGS.pred, FLAGS.emotion_set_path, FLAGS.single)
    print(json.dumps(results, indent=4))
    with open(f"{FLAGS.output}/score_results.json", "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    app.run(score)
