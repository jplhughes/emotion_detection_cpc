import random
from functools import partial

import torch
import torchaudio

from torch.utils.data import DataLoader, Dataset
from util import is_non_empty_file

RAW_SAMPLING_RATE = 16000
FRAME_LENGTH_MS = 25.0
FRAME_SHIFT_MS = 10.0

transform_to_sampling_rate = {
    "mfcc": int(1000 / FRAME_SHIFT_MS),
    "fbank": int(1000 / FRAME_SHIFT_MS),
    "raw": RAW_SAMPLING_RATE,
}


class AudioDataLoader(DataLoader):
    def __init__(
        self,
        *args,
        window_size=2,
        feature_transform="mfcc",
        **kwargs,
    ):
        if kwargs["num_workers"] != 0:
            if "timeout" not in kwargs:
                kwargs["timeout"] = 300
        super().__init__(*args, **kwargs)
        self.collate_fn = partial(
            self._collate_fn,
            window_size=window_size,
            feature_transform=feature_transform,
            train=self.dataset.train,
        )

        self.sampling_rate = transform_to_sampling_rate[feature_transform]

    @staticmethod
    def _collate_fn(
        batch, window_size=256, feature_transform="mfcc", normalization=True, train=True
    ):
        feats, labels, files, positions = [], [], [], []
        for path, label in batch:
            start_idx = 0
            audio_tensor, sample_rate = torchaudio.load(path, normalization=normalization)
            if sample_rate != RAW_SAMPLING_RATE:
                audio_tensor = torchaudio.transforms.Resample(sample_rate, RAW_SAMPLING_RATE)(
                    audio_tensor
                )
            # if training slice audio up into window size chunks
            if train:
                # find raw audio window size to give correct feature window size
                if feature_transform != "raw":
                    seconds_requested = window_size / 100
                    frame_shift_samples = int((FRAME_SHIFT_MS / 1000.0) * RAW_SAMPLING_RATE)  # 160
                    frame_length_samples = int(
                        (FRAME_LENGTH_MS / 1000.0) * RAW_SAMPLING_RATE
                    )  # 400
                    buffer_size = frame_length_samples - frame_shift_samples  # 240
                    raw_window_size = frame_shift_samples * window_size + buffer_size
                else:
                    seconds_requested = window_size / RAW_SAMPLING_RATE
                    raw_window_size = window_size

                # pad or splice depending on number on window size
                channels, num_samples = audio_tensor.shape
                if num_samples < raw_window_size:
                    difference = raw_window_size - num_samples
                    print(
                        f"requested {seconds_requested}s but only have "
                        f"{num_samples/RAW_SAMPLING_RATE}s, adding {difference} zero frames"
                    )
                    padding = torch.zeros((channels, difference))
                    audio_tensor = torch.cat([audio_tensor, padding], 1)
                else:
                    # TODO this is far from efficient, look into iterable dataset in future
                    start_idx = random.randint(0, num_samples - raw_window_size)
                    audio_tensor = audio_tensor.narrow(1, start_idx, raw_window_size)
            # if testing do not slice up
            else:
                assert len(batch) == 1

            feat = feature_fn(audio_tensor, feature_transform, RAW_SAMPLING_RATE)
            if window_size == None:
                window_size = feat.shape[0]
            assert feat.shape[0] == window_size
            feats.append(feat)
            labels.append(torch.ones(window_size).long() * label)
            files.append(path)
            positions.append(start_idx)

        feats = torch.stack(feats)
        labels = torch.stack(labels)
        return {"data": feats, "labels": labels, "files": files, "positions": positions}


def feature_fn(data_tensor, feature_transform, samplerate):
    if feature_transform == "fbank":
        return torchaudio.compliance.kaldi.fbank(
            data_tensor,
            window_type="hamming",
            dither=1.0,
            num_mel_bins=80,
            htk_compat=True,
            use_energy=False,
            frame_length=FRAME_LENGTH_MS,
            frame_shift=FRAME_SHIFT_MS,
            sample_frequency=samplerate,
        )
    elif feature_transform == "mfcc":
        return torchaudio.compliance.kaldi.mfcc(
            data_tensor,
            num_mel_bins=40,
            num_ceps=40,
            use_energy=False,
            high_freq=-400,
            low_freq=20,
            sample_frequency=samplerate,
            dither=0.0,
            energy_floor=0.0,
        )
    elif feature_transform == "raw":
        return data_tensor
    else:
        raise NotImplementedError(f"feature transform {feature_transform} not implemented.")


class AudioDataset(Dataset):
    def __init__(self, dbl_path, train=True):
        super().__init__()
        self.train = train
        self.data = self.parse_audio_dbl(dbl_path)

    def __getitem__(self, index):
        file_audio = self.data[index]
        return file_audio, -1

    def __len__(self):
        return len(self.data)

    @staticmethod
    def parse_audio_dbl(dbl_path):
        dbl_entries = []
        with open(dbl_path) as in_f:
            for line in in_f.readlines():
                audio_path = line.strip().split()[0]
                if is_non_empty_file(audio_path):
                    dbl_entries.append(audio_path)
        if not dbl_entries:
            raise KeyError("dbl list is empty, check paths to dbl files")
        return dbl_entries


class EmotionDataset(Dataset):
    def __init__(self, dbl_path, emotion_set_path, train=True):
        super().__init__()
        self.train = train
        self.emotion2id = self.get_emotion_to_id_mapping(emotion_set_path)
        self.data = self.parse_emotion_dbl(dbl_path)

    def __getitem__(self, index):
        audio_path, emotion_type = self.data[index]
        return audio_path, self.emotion2id[emotion_type]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def parse_emotion_dbl(dbl_path):
        dbl_entries = []
        with open(dbl_path) as in_f:
            for line in in_f.readlines():
                audio_path, emotion_type = line.strip().split()
                if is_non_empty_file(audio_path):
                    dbl_entries.append((audio_path, emotion_type))
        if not dbl_entries:
            raise KeyError("dbl list is empty, check paths to dbl files")
        return dbl_entries

    @staticmethod
    def get_emotion_to_id_mapping(emotion_set_path):
        with open(emotion_set_path) as in_f:
            return {p.strip(): i for i, p in enumerate(in_f.readlines())}
