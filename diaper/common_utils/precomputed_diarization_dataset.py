#!/usr/bin/env python3

# Drop-in replacement for KaldiDiarizationDataset that reads mel
# spectrograms precomputed by precompute_features.py, and applies
# splice / subsample / specaugment / top-N speaker selection at load
# time (cheap, so you're still free to change context_size, subsampling
# or specaugment without re-running the precompute step).

import logging
import os
import pickle
from typing import Tuple

import numpy as np
import torch
import torchaudio.transforms as T

import diaper.common_utils.features as features
# Side-effect import: lets pickle.load() here read .pkl chunks written by
# precompute_features.py under numpy>=2.0 even when this process is running
# numpy<2.0. See numpy2_pickle_compat.py for why this is needed; it's a
# no-op if numpy is already >=2.0.
import diaper.common_utils.numpy2_pickle_compat  # noqa: F401


def _apply_specaugment(Y: np.ndarray) -> np.ndarray:
    """Mirrors the specaugment branch at the tail of features.transform()."""
    timemasking = T.TimeMasking(time_mask_param=80)
    Y_t = timemasking(torch.from_numpy(Y).unsqueeze(0).transpose(1, 2))
    freqmasking = T.FrequencyMasking(freq_mask_param=80)
    Y_t = freqmasking(Y_t.transpose(1, 2)[0]).numpy()
    return Y_t


class PrecomputedKaldiDiarizationDataset(torch.utils.data.Dataset):
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

        meta_path = os.path.join(precomputed_dir, 'meta.pkl')
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
        self.chunk_indices = self.meta['chunk_indices']

        logging.info(
            f"Loaded precomputed dataset from {precomputed_dir}: "
            f"#chunks: {len(self.chunk_indices)} "
            f"(feature_dim={self.meta['feature_dim']}, "
            f"input_transform={self.meta['input_transform']})")

        self.saved = None  # used in case of empty sequence, like the original

    def __len__(self) -> int:
        return len(self.chunk_indices)

    def _load_chunk(self, i: int):
        path = os.path.join(self.precomputed_dir, f"{i:08d}.pkl")
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d['Y'], d['T'], d['rec'], d['st'], d['ed'], d['speaker_ids']

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
        # extract top-(self.n_speakers) speakers -- same as the original.
        if self.n_speakers and T_ss.shape[1] > self.n_speakers:
            selected_spkrs = np.argsort(
                T_ss.sum(axis=0))[::-1][:self.n_speakers]
            T_ss = T_ss[:, selected_spkrs]

        return torch.from_numpy(np.copy(Y_ss).astype(self.dtype)), \
            torch.from_numpy(np.copy(T_ss)), rec, st, ed, speaker_ids
