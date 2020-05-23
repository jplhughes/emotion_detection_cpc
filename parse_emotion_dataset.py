"""
This script extracts metadata from the emotion dataset and
creates a train, val and test dbl file with 80:10:10 split
"""
import os
from collections import namedtuple
import json
import argparse
from random import random

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_location", "-d", action="store", default="data/Audio_Speech_Actors_01-24"
)
parser.add_argument("--json", "-j", action="store_true", default=False)
parser.add_argument("--output", "-o", action="store", default="data")
args = parser.parse_args()

# Define how emotion files are parsed
Metadata = namedtuple(
    "Metadata", "modeality vocal emotion intensity statement repetition actor gender"
)
modality = {"01": "full-AV", "02": "video-only", "03": "audio-only"}
vocal_channel = {"01": "speech", "02": "song"}
emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
intensity = {"01": "normal", "02": "strong"}
statement = {"01": "Kids are talking by the door", "02": "Dogs are sitting by the door"}
emotion2class = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7,
}


def parse_filename(filepath):
    """ extract meaning from emotion dataset filenames """
    filename = filepath.split("/")[-1].split(".")[0]  # e.g. 03-01-04-02-02-02-02
    m, v, e, i, s, r, a = filename.split("-")

    m = modality[m]
    v = vocal_channel[v]
    e = emotion[e]
    i = intensity[i]
    s = statement[s]
    r = int(r)
    a = int(a)
    g = "male" if a % 2 == 1 else "female"

    meta_named = Metadata(m, v, e, i, s, r, a, g)
    meta_dict = {
        "modeality": m,
        "vocal_channel": v,
        "emotion": e,
        "intensity": i,
        "statement": s,
        "repetition": r,
        "actor": a,
        "gender": g,
    }

    return meta_dict, meta_named


# parse each file
file_dict = {}
file_meta = {}
for subdir, dirs, files in os.walk(args.data_location):
    for file in files:
        file_path = os.path.join(subdir, file)

        if file_path.endswith(".wav"):
            meta_dict, meta_named = parse_filename(file_path)
            file_dict[file_path] = meta_dict
            file_meta[file_path] = meta_named

# save metadata in json
if args.json:
    with open(f"{args.output}/metadata.json", "w") as fp:
        json.dump(file_dict, fp, indent=4)

# save emotion set .txt
with open(f"{args.output}/emotion_set.txt", "w") as fp:
    for emotion in emotion2class.values():
        fp.write(f"{emotion}\n")

# Save dbl files
# 80:10:10 train val test split, two speakers are kept separate from training
train_dbl = open(f"{args.output}/train.dbl", "w")
test_dbl = open(f"{args.output}/test.dbl", "w")
val_dbl = open(f"{args.output}/val.dbl", "w")

prob_not_train = (24 * 0.2 - 2) / (24 - 2)

train_class_counts = {k: 0 for k in emotion.values()}
test_class_counts = {k: 0 for k in emotion.values()}
val_class_counts = {k: 0 for k in emotion.values()}

for filepath, meta in file_meta.items():
    if meta.actor in [23, 24]:
        if random() < 0.5:
            val_dbl.write(f"{filepath} {emotion2class[meta.emotion]}\n")
            val_class_counts[meta.emotion] += 1
        else:
            test_dbl.write(f"{filepath} {emotion2class[meta.emotion]}\n")
            test_class_counts[meta.emotion] += 1
    else:
        if random() < prob_not_train:
            if random() < 0.5:
                val_dbl.write(f"{filepath} {emotion2class[meta.emotion]}\n")
                val_class_counts[meta.emotion] += 1
            else:
                test_dbl.write(f"{filepath} {emotion2class[meta.emotion]}\n")
                test_class_counts[meta.emotion] += 1
        else:
            train_dbl.write(f"{filepath} {emotion2class[meta.emotion]}\n")
            train_class_counts[meta.emotion] += 1

train_dbl.close()
test_dbl.close()
val_dbl.close()

print(train_class_counts)
print(val_class_counts)
print(test_class_counts)
