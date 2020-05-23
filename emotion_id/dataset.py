from collections import namedtuple
import numpy as np

from dataloader.streaming import FbankStream, SingleFileStream
from util import is_non_empty_file

# presumption is one emotion for an entire audio file
DblEntry = namedtuple("DblEntry", "audio_path emotion")


def parse_emotion_dbl(dbl_path):
    dbl_entries = []
    with open(dbl_path) as in_f:
        for l in in_f.readlines():
            x, y = l.strip().split()
            if is_non_empty_file(x):
                dbl_entries.append(DblEntry(audio_path=x, emotion_type=y))
    if not dbl_entries:
        raise KeyError("dbl list is empty, check paths to dbl files")
    return dbl_entries


def get_emotion_to_id_mapping(emotion_set_path):
    with open(emotion_set_path) as in_f:
        return {p.strip(): i for i, p in enumerate(in_f.readlines())}


class EmotionIDSingleFileStream(SingleFileStream):
    def __init__(
        self, dbl_entry, window_size, emotion_set_path, audiostream_class=FbankStream
    ):
        self.dbl_entry = dbl_entry
        self.audiostream = audiostream_class(dbl_entry.audio_path, window_size)
        self.emotion2id = get_emotion_to_id_mapping(emotion_set_path)
        self.emotion_id = self.emotion2id[dbl_entry.emotion_type]
        self.window_size = window_size
        self.frame_count = 0
        self.sampling_rate_hz = self.audiostream.sampling_rate_hz

    def get_window(self, window_size):
        window = self.audiostream.get_window(window_size)
        if window is None:
            return
        # if there is insufficient data available, window may actually be smaller so we need
        # to take that into account
        actual_window_size = window["data"].shape[0]
        emotions = (self.emotion_id * np.ones((actual_window_size))).astype(np.int64)
        self.frame_count = self.frame_count + actual_window_size
        return {
            "data": window["data"],
            "labels": emotions,
            "frame_idx": np.arange(
                self.frame_count - actual_window_size, self.frame_count
            ),
        }

    def __iter__(self):
        while True:
            window = self.get_window(self.window_size)
            if window is None:
                return
            yield window
