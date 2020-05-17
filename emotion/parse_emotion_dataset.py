import os
from collections import namedtuple
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_location",
    "-d",
    action="store",
    default="data/Audio_Speech_Actors_01-24",
    dest="data_location",
)
parser.add_argument(
    "--json_output",
    "-j",
    action="store",
    default="emotion/metadata.json",
    dest="json_output",
)
parser.add_argument(
    "--dbl_output", "-j", action="store", default="emotion/all.dbl", dest="dbl_output"
)
args = parser.parse_args()


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


file_dict = {}
file_meta = {}
for subdir, dirs, files in os.walk(args.data_location):
    for file in files:
        file_path = os.path.join(subdir, file)

        if file_path.endswith(".wav"):
            meta_dict, meta_named = parse_filename(file_path)
            file_dict[file_path] = meta_dict
            file_meta[file_path] = meta_named

with open(args.json_output, "w") as fp:
    json.dump(file_dict, fp, indent=4)

with open(args.dbl_output, "w") as fp:
    for filepath, meta in file_meta.items():
        fp.write(f"{filepath} {emotion2class[meta.emotion]}\n")
