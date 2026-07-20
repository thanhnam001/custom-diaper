#!/usr/bin/env python3

# Drop-in replacement for PrecomputedKaldiDiarizationDataset
# (precomputed_diarization_dataset.py) that reads the per-chunk torch .pt
# cache format written by the sibling `Master` project's precompute
# pipeline (src/dataloader/diaper/per_chunk_precompute.py), instead of the
# pickle + meta.pkl format written by precompute_features.py in this repo.
# This lets both repos share one precomputed cache on disk -- no
# reprocessing of audio, no duplicated storage. See CLAUDE.md's
# "Datasets" section for how the two caches relate.
#
# Each .pt file is one training chunk, named
# "{rec}__{start_frame}__{end_frame}.pt" (no separate index file -- unlike
# meta.pkl/{idx:08d}.pkl, the directory listing itself is the index) and
# contains:
#   Y            (torch.Tensor float32)  (n_frames, feature_dim), mel
#                features with input_transform (e.g. logmel_meannorm)
#                already baked in
#   T            (torch.Tensor int32)    (n_frames, n_speakers) diarization
#                labels at full (precompute-time) frame rate
#   rec          (str)  recording id
#   start_frame  (int)  raw frame index
#   end_frame    (int)  raw frame index
#   speaker_ids  (list[int])  global speaker indices seen anywhere in the
#                *source recording* -- NOT chunk-local, same caveat as the
#                sibling project's own loader (see precomputed_chunk_dataset.py
#                there)
#   meta         (dict)  feature_dim/input_transform/sampling_rate/
#                frame_size/frame_shift/n_speakers/num_frames/subsampling
#                baked in at precompute time
#
# splice / subsample / specaugment / top-N speaker selection are applied
# here at load time, same as PrecomputedKaldiDiarizationDataset.

import logging
import os
from typing import Tuple

import numpy as np
import torch
import torchaudio.transforms as T

import diaper.common_utils.features as features


def _apply_specaugment(Y: np.ndarray) -> np.ndarray:
    """Mirrors the specaugment branch at the tail of features.transform()."""
    timemasking = T.TimeMasking(time_mask_param=80)
    Y_t = timemasking(torch.from_numpy(Y).unsqueeze(0).transpose(1, 2))
    freqmasking = T.FrequencyMasking(freq_mask_param=80)
    Y_t = freqmasking(Y_t.transpose(1, 2)[0]).numpy()
    return Y_t


class TorchPrecomputedKaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        precomputed_dir: str,
        context_size: int,
        n_speakers: int,
        subsampling: int,
        specaugment: bool,
        dtype: type = np.float32,
    ):
        self.precomputed_dir = precomputed_dir
        self.context_size = context_size
        self.n_speakers = n_speakers
        self.subsampling = subsampling
        self.specaugment = specaugment
        self.dtype = dtype

        self.files = sorted(
            f for f in os.listdir(precomputed_dir) if f.endswith('.pt')
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {precomputed_dir}")

        sample = torch.load(
            os.path.join(precomputed_dir, self.files[0]), map_location='cpu')
        meta = sample.get('meta', {})
        logging.info(
            f"Loaded precomputed (.pt) dataset from {precomputed_dir}: "
            f"#chunks: {len(self.files)} "
            f"(feature_dim={meta.get('feature_dim')}, "
            f"input_transform={meta.get('input_transform')})")

        self.saved = None  # used in case of empty sequence, like the original

    def __len__(self) -> int:
        return len(self.files)

    def _load_chunk(self, i: int):
        path = os.path.join(self.precomputed_dir, self.files[i])
        d = torch.load(path, map_location='cpu')
        Y = d['Y'].numpy()
        T_ = d['T'].numpy()
        return Y, T_, d['rec'], d['start_frame'], d['end_frame'], d['speaker_ids']

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        Y, T_, rec, st, ed, speaker_ids = self._load_chunk(i)

        if Y.shape[0] == 0:
            print(f"{rec, st, ed} is empty: {Y.shape}, "
                  "replacing with saved sequence")
            Y, T_, rec, st, ed, speaker_ids = self.saved
        else:
            self.saved = (Y, T_, rec, st, ed, speaker_ids)

        if self.specaugment:
            Y = _apply_specaugment(Y)

        Y_spliced = features.splice(Y, self.context_size)
        Y_ss, T_ss = features.subsample(Y_spliced, T_, self.subsampling)

        # If the sample contains more than "self.n_speakers" speakers,
        # extract top-(self.n_speakers) speakers -- same as
        # PrecomputedKaldiDiarizationDataset / the original KaldiDiarizationDataset.
        if self.n_speakers and T_ss.shape[1] > self.n_speakers:
            selected_spkrs = np.argsort(
                T_ss.sum(axis=0))[::-1][:self.n_speakers]
            T_ss = T_ss[:, selected_spkrs]

        return torch.from_numpy(np.copy(Y_ss).astype(self.dtype)), \
            torch.from_numpy(np.copy(T_ss)), rec, st, ed, speaker_ids
