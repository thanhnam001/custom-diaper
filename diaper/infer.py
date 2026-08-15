#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2023 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

import os
import sys
# common_utils.diarization_dataset (and, transitively, common_utils.features)
# use package-qualified `diaper.common_utils.*` imports, which only resolve
# if the repo root is on sys.path. That isn't the case when this script is
# run as `python diaper/infer.py` (Python puts diaper/ itself, not the repo
# root, on sys.path[0]). Add the repo root too so both import styles resolve
# regardless of how this script is invoked.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Side-effect import, must happen before torch is imported below: on
# Python 3.10+, restores the `collections.Container`/`Mapping`/ etc
# aliases that old pinned libraries (torch==1.10.0) still reach for
# directly on `collections` instead of `collections.abc`. See
# common_utils/collections_abc_compat.py for details; it's a no-op on
# Python <3.10.
import common_utils.collections_abc_compat  # noqa: E402,F401

from backend.losses import pad_labels_zeros
from backend.models import (
    average_checkpoints,
    get_model,
)
from common_utils.arg_types import str2bool
from common_utils.diarization_dataset import KaldiDiarizationDataset
from common_utils.metrics import calculate_metrics
from os.path import join
from pathlib import Path
from torch.utils.data import DataLoader
from train import _convert
from types import SimpleNamespace
from typing import Dict, List, TextIO, Tuple
from scipy.optimize import linear_sum_assignment
from scipy.signal import medfilt
from tqdm import tqdm
import copy
import csv
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import threadpoolctl
import torch
import yamlargparse


def get_infer_dataloader(args: SimpleNamespace) -> DataLoader:
    infer_set = KaldiDiarizationDataset(
        args.infer_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=True,
        min_length=0,
        specaugment=args.specaugment,
    )
    infer_loader = DataLoader(
        infer_set,
        batch_size=1,
        collate_fn=_convert,
        num_workers=0,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    Y, _, _, _, _, _ = infer_set.__getitem__(0)
    assert Y.shape[1] == \
        (args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of \
        {args.feature_dim} but {Y.shape[1]} found."
    return infer_loader


def hard_labels_to_rttm(
    labels: np.ndarray,
    id_file: str,
    rttm_file: TextIO,
    frameshift: float = 10
) -> None:
    """
    Transform NfxNs matrix to an rttm file
    Nf is the number of frames
    Ns is the number of speakers
    The frameshift (in ms) determines how to interpret the frames in the array
    """
    if len(labels.shape) > 1:
        # Remove speakers that do not speak
        non_empty_speakers = np.where(labels.sum(axis=0) != 0)[0]
        labels = labels[:, non_empty_speakers]

    # Add 0's before first frame to use diff
    if len(labels.shape) > 1:
        labels = np.vstack([np.zeros((1, labels.shape[1])), labels])
    else:
        labels = np.vstack([np.zeros(1), labels])
    d = np.diff(labels, axis=0)

    spk_list = []
    ini_list = []
    end_list = []
    if len(labels.shape) > 1:
        n_spks = labels.shape[1]
    else:
        n_spks = 1
    for spk in range(n_spks):
        if n_spks > 1:
            ini_indices = np.where(d[:, spk] == 1)[0]
            end_indices = np.where(d[:, spk] == -1)[0]
        else:
            ini_indices = np.where(d[:] == 1)[0]
            end_indices = np.where(d[:] == -1)[0]
        # Add final mark if needed
        if len(ini_indices) == len(end_indices) + 1:
            end_indices = np.hstack([
                end_indices,
                labels.shape[0] - 1])
        assert len(ini_indices) == len(end_indices), \
            "Quantities of start and end of segments mismatch. \
            Are speaker labels correct?"
        n_segments = len(ini_indices)
        for index in range(n_segments):
            spk_list.append(spk)
            ini_list.append(ini_indices[index])
            end_list.append(end_indices[index])
    for ini, end, spk in sorted(zip(ini_list, end_list, spk_list)):
        rttm_file.write(
            f"SPEAKER {id_file} 1 " +
            f"{round(ini * frameshift / 1000, 3)} " +
            f"{round((end - ini) * frameshift / 1000, 3)} " +
            f"<NA> <NA> spk{spk} <NA> <NA>\n")


def rttm_to_hard_labels(
    rttm_path: str,
    precision: float,
    length: float = -1
) -> Tuple[np.ndarray, List[str]]:
    """
        reads the rttm and returns a NfxNs matrix encoding the segments in
        which each speaker is present (labels 1/0) at the given precision.
        Ns is the number of speakers and Nf is the resulting number of frames,
        according to the parameters given.
        Nf might be shorter than the real length of the utterance, as final
        silence parts cannot be recovered from the rttm.
        If length is defined (s), it is to account for that extra silence.
        In case of silence all speakers are labeled with 0.
        In case of overlap all speakers involved are marked with 1.
        The function assumes that the rttm only contains speaker turns (no
        silence segments).
        The overlaps are extracted from the speaker turn collisions.
    """
    # each row is a turn, columns denote beginning (s) and duration (s) of turn
    data = np.loadtxt(rttm_path, usecols=[3, 4])
    # speaker id of each turn
    spks = np.loadtxt(rttm_path, usecols=[7], dtype='str')
    spk_ids = np.unique(spks)
    Ns = len(spk_ids)
    if data.shape[0] == 2 and len(data.shape) < 2:  # if only one segment
        data = np.asarray([data])
        spks = np.asarray([spks])
    # length of the file (s) that can be recovered from the rttm,
    # there might be extra silence at the end
    len_file = data[-1][0]+data[-1][1]
    if length > len_file:
        len_file = length

    # matrix in given precision
    matrix = np.zeros([int(round(len_file*precision)), Ns])
    # ranges to mark each turn
    ranges = np.around((np.array([data[:, 0],
                        data[:, 0]+data[:, 1]]).T*precision)).astype(int)

    for s in range(Ns):  # loop over speakers
        # loop all the turns of the speaker
        for init_end in ranges[spks == spk_ids[s], :]:
            matrix[init_end[0]:init_end[1], s] = 1  # mark the spk
    return matrix, spk_ids


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def estimate_diarization_outputs(
    model,
    inputs: torch.Tensor,
    args: SimpleNamespace
) -> List[torch.Tensor]:
    '''
    - ys_active: `[(T, n_active_attractors)]`
    - existence_probs: `(B, n_attractors)`
    - per_prcvblock_latents: `(B, n_latents, d_latents, n_blocks)` 
    - per_prcvblock_attractors: `(B, n_attractors, d_latents, n_blocks)`
    - y_probs: `(B, T, n_attractors)` sigmoided tensor
    '''
    assert args.estimate_spk_qty_thr != -1 or \
        args.estimate_spk_qty != -1, \
        "Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' \
        arguments have to be defined."
    (
        all_frame_embs,
        per_frameenclayer_ys_logits,
        per_frameenclayer_attractors_logits,
        per_frameenclayer_attractors,
        _spk_counting_logits,  # unused: only compute_loss_and_metrics trains
        per_prcvblock_ys_logits,
        per_prcvblock_attractors_logits,
        per_prcvblock_attractors,
        per_prcvblock_l2a_entropy_term,
        per_prcvblock_latents
    ) = model.forward(inputs, args)

    ys_active = []
    existence_probs = torch.sigmoid(per_frameenclayer_attractors_logits[:, :, -1])
    ys = [torch.sigmoid(y) for y in per_frameenclayer_ys_logits[:, :, :, -1]] # [(T, n_attractors)]
    for p, y in zip(existence_probs, ys):
        if args.estimate_spk_qty != -1:
            _, order = torch.sort(p, descending=True)
            ys_active.append(y[:, order[:args.estimate_spk_qty]])
        elif args.estimate_spk_qty_thr != -1:
            active_speakers = torch.where(p >= args.estimate_spk_qty_thr)[0]
            ys_active.append(y[:, active_speakers])
        else:
            NotImplementedError(
                'estimate_spk_qty or estimate_spk_qty_thr needed.')
    return (
        ys_active, existence_probs, per_prcvblock_latents,
        per_prcvblock_attractors, torch.stack(ys))


def get_hard_decisions(
    probabilities,
    threshold: float,
    median_window_length: int,
    normalize_probs: bool
) -> np.ndarray:
    """Threshold probabilities and apply median filter, at the model's
    native (subsampled) frame rate -- i.e. everything postprocess_output
    does except the final upsampling back to raw frame rate."""
    if normalize_probs:
        probabilities = (probabilities - probabilities.min(axis=0)[0]) / \
                probabilities.max(axis=0)[0]
    thresholded = probabilities.to("cpu") > threshold
    filtered = np.zeros(thresholded.shape)
    # in each speaker, apply median filter
    for spk in range(filtered.shape[1]):
        filtered[:, spk] = medfilt(
            thresholded[:, spk].to(float),
            kernel_size=median_window_length).astype(bool)
    return filtered


def postprocess_output(
    probabilities,
    subsampling: int,
    threshold: float,
    median_window_length: int,
    normalize_probs: bool
) -> np.ndarray:
    """Threshold probabilities and apply median filter."""
    filtered = get_hard_decisions(
        probabilities, threshold, median_window_length, normalize_probs)
    probs_extended = np.repeat(filtered, subsampling, axis=0) # Upsampling
    return probs_extended


def get_active_attractor_mask(
    existence_probs: torch.Tensor,
    args: SimpleNamespace
) -> torch.Tensor:
    """Same active/inactive speaker decision as estimate_diarization_outputs
    (top-`estimate_spk_qty` by existence prob, or thresholded at
    `estimate_spk_qty_thr`), but expressed as a fixed-width `(n_attractors,)`
    0/1 mask instead of dropping inactive columns -- needed so predictions
    and reference labels line up column-for-column for calculate_metrics."""
    active_mask = torch.zeros_like(existence_probs)
    if args.estimate_spk_qty != -1:
        _, order = torch.sort(existence_probs, descending=True)
        active_mask[order[:args.estimate_spk_qty]] = 1.0
    else:
        active_mask[existence_probs >= args.estimate_spk_qty_thr] = 1.0
    return active_mask


def get_exists_mask(
    y_probs: torch.Tensor,
    ref_labels: torch.Tensor,
) -> torch.Tensor:
    """Attractor-slot-aligned 1/0 ground truth for whether each predicted
    attractor slot corresponds to a real reference speaker in this file --
    the same thing pit_loss_multispk's `exists_mask` return value is,
    computed the same way (Hungarian assignment on a BCE-style cost between
    per-frame activation probabilities and reference labels; the "logits"
    version there and this "probabilities" version are the same formula,
    since -logsigmoid(logit) == -log(sigmoid(logit)) == -log(p)), needed to
    compute attractor_accuracy per file the way train.py computes it per
    batch. `y_probs`/`ref_labels` are both `(T, n_attractors)` -- caller
    pads/truncates to that shape first."""
    eps = 1e-6
    p = y_probs.clamp(eps, 1 - eps)
    cost_mx = (
        -torch.log(p).t().matmul(ref_labels)
        - torch.log(1 - p).t().matmul(1 - ref_labels)
    )
    _, ref_alig = linear_sum_assignment(cost_mx.to("cpu"))
    active_cols = torch.where(ref_labels.sum(axis=0) != 0)[0]
    n_ref_spk = int(active_cols.max().item()) + 1 if active_cols.numel() > 0 else 0
    spk_labels = torch.zeros(ref_labels.shape[1])
    spk_labels[:n_ref_spk] = 1.0
    return spk_labels[ref_alig]


def compute_file_metrics(
    y_probs: torch.Tensor,
    existence_probs: torch.Tensor,
    ref_labels: torch.Tensor,
    args: SimpleNamespace
) -> Dict[str, float]:
    """Whole-file diarization metrics (frame-level DER/miss/FA/confusion,
    VAD/OSD error rates, attractor existence accuracy), computed with the
    same decision process used to write the RTTM (existence gating +
    threshold + median filter) so this is directly informative about what
    dscore would score -- optionally with the same forgiveness collar
    dscore uses (via `args.collar`, seconds), computed locally per file
    instead of requiring a separate dscore run. With `args.collar == 0.0`
    (the default) this is a strictly harsher, no-collar frame-level metric
    that reads systematically higher than dscore, especially on short/
    turn-dense files -- set --collar to make it comparable in absolute
    terms (e.g. 0.25 for MSDWild, 0.0/unset for RAMC, matching each
    dataset's dscore convention)."""
    active_mask = get_active_attractor_mask(existence_probs, args)
    pred_hard = get_hard_decisions(
        y_probs * active_mask.unsqueeze(0), args.threshold,
        args.median_window_length, args.normalize_probs)
    pred_hard = torch.from_numpy(pred_hard).float()
    ref_padded = pad_labels_zeros([ref_labels], args.n_attractors)[0]
    # Frame counts can differ by a couple of frames at file boundaries
    # between the dataset's and the model's subsampling arithmetic; align
    # by truncating to the shorter of the two rather than erroring out.
    n_frames = min(pred_hard.shape[0], ref_padded.shape[0], y_probs.shape[0])
    # pred_hard/ref_padded are both at the model's native (subsampled)
    # frame rate -- convert the collar from seconds to frames at that rate.
    frame_period = args.subsampling * args.frame_shift / args.sampling_rate
    collar_frames = round(args.collar / frame_period) if args.collar > 0 else 0
    file_metrics, denominators = calculate_metrics(
        ref_padded[:n_frames].unsqueeze(0),
        pred_hard[:n_frames].unsqueeze(0),
        threshold=0.5, collar_frames=collar_frames,
        return_denominators=True)
    file_metrics = {k: float(v) for k, v in file_metrics.items()}
    # calculate_metrics normalizes VAD_FA/miss and OSD_FA/miss by
    # active_frames_tot/overlap_frames_tot -- a population disjoint from
    # what their FA numerators count, so when a file has zero real speech
    # (silent reference) or zero real overlap (any single-speaker-at-a-time
    # file, common in MSDWiLD) the rate blows up to an arbitrary multiple of
    # any nonzero FA count instead of being merely "undefined but small".
    # Mark those as NaN (undefined for this file) rather than reporting a
    # meaningless huge number that would dominate a per-file average.
    if denominators["active_frames_tot"] == 0:
        file_metrics["VAD_FA"] = float("nan")
        file_metrics["VAD_miss"] = float("nan")
    if denominators["overlap_frames_tot"] == 0:
        file_metrics["OSD_FA"] = float("nan")
        file_metrics["OSD_miss"] = float("nan")
    # Same issue for DER/DER_FA/DER_miss/DER_conf, normalized by
    # speech_frames_tot -- doesn't fire on real MSDWiLD/RAMC files (every
    # recording has some reference speech) but guard it anyway since it's
    # the same root cause (a fully-silent-reference file/chunk).
    if denominators["speech_frames_tot"] == 0:
        file_metrics["DER"] = float("nan")
        file_metrics["DER_FA"] = float("nan")
        file_metrics["DER_miss"] = float("nan")
        file_metrics["DER_conf"] = float("nan")

    exists_mask = get_exists_mask(
        y_probs[:n_frames].to("cpu"), ref_padded[:n_frames].to("cpu"))
    file_metrics["attractor_accuracy"] = (
        active_mask.to("cpu") == exists_mask
    ).float().mean().item() * 100

    return file_metrics


def parse_arguments() -> SimpleNamespace:
    """Parse arguments"""
    parser = yamlargparse.ArgumentParser(description='DiaPer inference')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--attractor-existence-loss-weight', default=1.0, type=float,
                        help='weighting parameter')
    parser.add_argument('--compute-metrics', default=False, type=str2bool,
                        help='compute whole-file frame-level diarization '
                        'metrics (DER/miss/FA/confusion, VAD/OSD error '
                        'rates, attractor existence accuracy) per '
                        'recording, using the same '
                        'existence-gating/threshold/median-filter decision '
                        'as the RTTM output, macro-averaged over the test '
                        'set. Writes metrics_per_file.csv and '
                        'metrics_summary.txt into the run\'s rttms_dir '
                        '(sibling of the rttms/ subdirectory). This is a '
                        'frame-level metric (no forgiveness collar unless '
                        '--collar is set), not a substitute for dscore '
                        'scoring -- it is meant for fast, local per-file '
                        'weakness analysis.')
    parser.add_argument('--collar', default=0.0, type=float,
                        help='forgiveness collar in seconds for '
                        '--compute-metrics, mirroring dscore/md-eval.pl\'s '
                        '--collar: frames within this many seconds of a '
                        'reference speaker-turn boundary are excluded from '
                        'scoring. Default 0.0 (no collar, original '
                        'behavior). Only affects --compute-metrics output, '
                        'not the RTTM written to rttms_dir. Use 0.25 to '
                        'match MSDWild\'s dscore convention, 0.0 (i.e. '
                        'leave unset) for RAMC\'s. Since this doesn\'t '
                        'change the RTTM, changing only --collar against an '
                        '--rttms-dir that already has RTTMs from a previous '
                        'run will skip inference for every file (see the '
                        '"RTTM already exists" warning below) -- point '
                        '--rttms-dir at a fresh directory to get complete '
                        'metrics_per_file.csv/metrics_summary.txt with the '
                        'new collar.')
    parser.add_argument('--attractor-frame-comparison', default='dotprod',
                        type=str, choices=['dotprod', 'xattention'],
                        help='how to compare attractors and frame embeddings')
    parser.add_argument('--att-qty-loss-weight', default=0.0, type=float)
    parser.add_argument('--att-qty-reg-loss-weight', default=0.0, type=float)
    parser.add_argument('--condition-frame-encoder', type=str2bool, default=True)
    parser.add_argument('--conformer-conv-kernel-size', default=3, type=int,
                        help='depthwise-conv kernel size for the conformer '
                        'frame encoder (must be odd)')
    parser.add_argument('--conv-norm-type', default='batchnorm', type=str,
                        choices=['batchnorm', 'layernorm'],
                        help='normalization used inside the conformer '
                        'frame encoder\'s ConvolutionModule; must match '
                        'the value the model was trained with')
    parser.add_argument('--context-activations', type=str2bool, default=False)
    parser.add_argument('--context-size', type=int)
    parser.add_argument('--d-latents', type=int,
                        help='dimension of attractors')
    parser.add_argument('--detach-attractor-loss', default=False, type=str2bool,
                        help='If True, avoid backpropagation on attractor loss')
    parser.add_argument('--dropout_attractors', type=float,
                        help='attention dropout for attractors path')
    parser.add_argument('--dropout_frames', type=float,
                        help='attention dropout for frame embeddings path')
    parser.add_argument('--epochs', type=str,
                        help='epochs to average separated by commas \
                        or - for intervals.')
    parser.add_argument('--estimate-spk-qty', default=-1, type=int)
    parser.add_argument('--estimate-spk-qty-thr', default=-1, type=float)
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-encoder-heads', type=int)
    parser.add_argument('--frame-encoder-layers', type=int)
    parser.add_argument('--frame-encoder-units', type=int)
    parser.add_argument('--frame-encoder-type', default='self_attention',
                        type=str, choices=['self_attention', 'conformer'],
                        help='block type used inside the frame encoder loop')
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--fallback-cpu-oom', default=False, type=str2bool,
                        help='if a GPU forward pass raises a CUDA '
                        'out-of-memory error (typically on unusually long '
                        'recordings), retry that single recording on CPU '
                        'instead of skipping it. Has no effect if --gpu < 1. '
                        'CAUTION: the CPU retry has no memory ceiling of '
                        'its own -- on a long enough recording it can '
                        'allocate tens of GB in one shot and risk an '
                        'OS-level OOM that takes down unrelated processes '
                        '(e.g. sshd) rather than just this one, especially '
                        'on a shared/remote host. Pair with '
                        '--max-input-frames so runaway recordings are '
                        'skipped before either device is asked to hold '
                        'that allocation.')
    parser.add_argument('--max-input-frames', default=-1, type=int,
                        help='skip (with a warning, before attempting any '
                        'forward pass on either device) recordings whose '
                        'subsampled frame count exceeds this. Whole-'
                        'recording self-attention (--use-frame-selfattention '
                        'with --num-frames -1) is O(frames^2) in memory, so '
                        'unusually long recordings can demand tens of GB '
                        'for the attention matrices alone -- this bounds '
                        'that risk (for both the primary attempt and the '
                        '--fallback-cpu-oom retry) instead of discovering '
                        'the limit via a crash. -1 (default) disables the '
                        'check, matching prior behavior.')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--infer-data-dir', help='inference data directory.')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--latents2attractors', type=str, default='linear')
    parser.add_argument('--length-normalize', default=False, type=str2bool)
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--median-window-length', default=11, type=int)
    parser.add_argument('--model-type', default='AttractorsPath',
                        help='Type of model (for now only AttractorsPath)')
    parser.add_argument('--models-path', type=str,
                        help='directory with model(s) to evaluate')
    parser.add_argument('--n-attractors', type=int,
                        help='Number of attractors to use')
    parser.add_argument('--n-blocks-attractors', type=int,
                        help='number of blocks in the transformer encoder')
    parser.add_argument('--n-internal-blocks-attractors', type=int, default=1,
                        help='number of Perceiver internal block, which \
                        repeats self-attention layers for attractors')
    parser.add_argument('--n-latents', type=int,
                        help='number of latents')
    parser.add_argument('--n-selfattends-attractors', type=int,
                        help='number of slef-attention layers per block')
    parser.add_argument('--n-sa-heads-attractors', type=int,
                        help='number of self-attention heads per layer')
    parser.add_argument('--n-xa-heads-attractors', type=int,
                        help='number of cross-attention heads per layer')
    parser.add_argument('--normalize-probs', default=False, type=str2bool)
    parser.add_argument('--num-frames', default=-1, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int)
    parser.add_argument('--num-threads', default=-1, type=int,
                        help='cap CPU threads used for feature extraction '
                        '(librosa/numpy BLAS) and model inference. The '
                        'mel-filterbank matmul in librosa otherwise fans '
                        'out across every core via OpenBLAS/MKL. '
                        '-1 leaves the library default (all cores) in place.')
    parser.add_argument('--plot-output', default=False, type=str2bool)
    parser.add_argument('--posenc-maxlen', type=int, default=36000,
                        help="The maximum length allowed for the positional \
                        encoding. i.e. 36000 with 0.1s frames is 1 hour")
    parser.add_argument('--pre-xa-heads', type=int,
                        help='number of pre-Perceiver cross-attention heads')
    parser.add_argument('--ref-rttms-dir', type=str, default='',
                        help='directory with reference RTTMs, used for plots')
    parser.add_argument('--rttms-dir', type=str,
                        help='output directory for rttm files.')
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--speakerid-loss', type=str, default='',
                        choices=['arcface', 'vanilla'])
    parser.add_argument('--speakerid-num-speakers', type=int, default=-1)
    parser.add_argument('--specaugment', type=str2bool, default=False)
    parser.add_argument('--subsampling', type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--time-shuffle', action='store_true',
                        help='Shuffle time-axis order before input to the network')
    parser.add_argument('--use-frame-selfattention', default=False, type=str2bool)
    parser.add_argument('--use-posenc', default=False, type=str2bool)
    parser.add_argument('--use-pre-crossattention', default=False, type=str2bool)
    parser.add_argument('--use-spk-counting-head', default=False, type=str2bool,
                        help='must match whatever the checkpoint being '
                        'loaded was trained with (this is a model-'
                        'architecture flag, not a loss-weight toggle) -- '
                        'see train.py --use-spk-counting-head.')
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    init_args = parser.parse_args()
    return init_args


if __name__ == '__main__':
    args = parse_arguments()

    if args.num_threads > 0:
        # Caps torch's own intra-op thread pool (used by the model forward
        # pass) and, via threadpoolctl, the OpenBLAS/MKL thread pool that
        # numpy/librosa dispatch into (e.g. the mel-filterbank matmul in
        # common_utils.features.transform) -- that matmul is what pegs every
        # core, since OpenBLAS defaults to using all of them per call.
        torch.set_num_threads(args.num_threads)
        threadpoolctl.threadpool_limits(limits=args.num_threads)

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    logging.info(args)

    infer_loader = get_infer_dataloader(args)

    if args.gpu >= 1:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    assert args.estimate_spk_qty_thr != -1 or \
        args.estimate_spk_qty != -1, \
        ("Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' "
         "arguments have to be defined.")
    if args.estimate_spk_qty != -1:
        out_dir = join(args.rttms_dir, f"spkqty{args.estimate_spk_qty}_\
            thr{args.threshold}_median{args.median_window_length}")
    elif args.estimate_spk_qty_thr != -1:
        out_dir = join(args.rttms_dir, f"spkqtythr{args.estimate_spk_qty_thr}_\
            thr{args.threshold}_median{args.median_window_length}")

    model = get_model(args)

    # allow_partial=True: tolerates checkpoints saved before spk_counting_head
    # existed (e.g. every pretrained checkpoint under models/). That head's
    # output is only ever read by losses.py::get_loss (training), never by
    # anything in this inference path, so leaving it randomly-initialized
    # here has no effect on the produced RTTMs.
    model = average_checkpoints(
        args.device, model, args.models_path, args.epochs, allow_partial=True)
    model = model.to(args.device)
    model.eval()

    # Optional per-sample fallback: some recordings are long enough that a
    # single forward pass exceeds available VRAM even though most fit fine.
    # Rather than aborting the whole run, retry just that one recording on
    # a CPU copy of the model kept around for this purpose.
    cpu_model = None
    if args.fallback_cpu_oom:
        if args.device.type != "cuda":
            logging.warning(
                "--fallback-cpu-oom has no effect when --gpu < 1 "
                "(already running on CPU)")
        else:
            cpu_model = copy.deepcopy(
                model.module if hasattr(model, "module") else model
            ).to("cpu")
            cpu_model.eval()

    out_dir = join(
        args.rttms_dir,
        f"epochs{args.epochs}",
        f"timeshuffle{args.time_shuffle}",
        (f"spk_qty{args.estimate_spk_qty}_"
            f"spk_qty_thr{args.estimate_spk_qty_thr}"),
        f"detection_thr{args.threshold}",
        f"median{args.median_window_length}",
        f"subsampling{args.subsampling}",
        "rttms"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    metric_keys = [
        'DER', 'DER_miss', 'DER_FA', 'DER_conf',
        'VAD_FA', 'VAD_miss', 'OSD_FA', 'OSD_miss',
        'avg_ref_spk_qty', 'avg_pred_spk_qty', 'attractor_accuracy',
    ]
    # VAD_FA/VAD_miss/OSD_FA/OSD_miss can be NaN per file (undefined --
    # e.g. no real overlap in that file, see compute_file_metrics), so each
    # key is averaged over however many files actually defined it, not
    # blindly over n_scored.
    acum_test_metrics = {k: 0.0 for k in metric_keys}
    acum_test_counts = {k: 0 for k in metric_keys}
    per_file_metrics = []

    infer_pbar = tqdm(infer_loader, total=len(infer_loader))
    for batch in infer_pbar:
        name = batch['names'][0]
        if os.path.exists(join(out_dir, f"{name}.rttm")):
            if args.compute_metrics:
                logging.warning(
                    f"{name}: RTTM already exists, skipping inference -- "
                    "this file will be missing from metrics_per_file.csv. "
                    "Rerun into an empty --rttms-dir for complete metrics.")
            continue
        n_frames = batch['xs'][0].shape[0]
        if args.max_input_frames > 0 and n_frames > args.max_input_frames:
            logging.warning(
                f"{name}: {n_frames} subsampled frames exceeds "
                f"--max-input-frames {args.max_input_frames} -- "
                "whole-recording self-attention memory is O(frames^2), so "
                "this file risks exhausting GPU VRAM and (if "
                "--fallback-cpu-oom is set) system RAM badly enough to "
                "trigger an OS-level OOM. Skipping rather than attempting "
                "either device.")
            continue
        try:
            input = torch.stack(batch['xs']).to(args.device)
            with torch.no_grad():
                (
                    y_pred,
                    existence_probs,
                    per_prcvblock_latents,
                    per_prcvblock_attractors,
                    y_probs
                ) = estimate_diarization_outputs(model, input, args)
        except RuntimeError as e:
            is_oom = "out of memory" in str(e).lower()
            error_msg = str(e)
            # The exception's traceback holds references to every tensor
            # from the failed forward pass (a reference cycle that plain
            # refcounting can't break), so torch.cuda.empty_cache() alone
            # won't release that VRAM -- gc.collect() first is required to
            # actually drop those refs. See
            # https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-cant-allocate-memory
            del e
            if is_oom and args.device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
            if cpu_model is None or not is_oom:
                logging.error(f"{name}: inference failed: {error_msg}")
                continue
            logging.warning(
                f"{name}: CUDA out of memory, retrying this recording on "
                "CPU")
            try:
                input = torch.stack(batch['xs']).to("cpu")
                with torch.no_grad():
                    (
                        y_pred,
                        existence_probs,
                        per_prcvblock_latents,
                        per_prcvblock_attractors,
                        y_probs
                    ) = estimate_diarization_outputs(cpu_model, input, args)
            except RuntimeError as cpu_e:
                logging.error(
                    f"{name}: CPU fallback also failed: {cpu_e}")
                del cpu_e
                gc.collect()
                continue

        try:
            # Each one has a single sequence
            y_pred = y_pred[0]
            existence_probs = existence_probs[0]
            y_probs = y_probs[0]
            per_prcvblock_attractors = torch.stack([a[0] for a in per_prcvblock_attractors])
            per_prcvblock_latents = torch.stack([lat[0] for lat in per_prcvblock_latents])
            post_y = postprocess_output(
                y_pred, args.subsampling,
                args.threshold, args.median_window_length,
                args.normalize_probs)
            rttm_filename = join(out_dir, f"{name}.rttm")
            with open(rttm_filename, 'w', encoding='UTF-8') as rttm_file:
                hard_labels_to_rttm(post_y, name, rttm_file)
            torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"{name}: postprocessing failed: {e}")
            continue

        if args.compute_metrics:
            try:
                file_metrics = compute_file_metrics(
                    y_probs, existence_probs, batch['ts'][0], args)
                for k in metric_keys:
                    if not np.isnan(file_metrics[k]):
                        acum_test_metrics[k] += file_metrics[k]
                        acum_test_counts[k] += 1
                per_file_metrics.append({'name': name, **file_metrics})
            except Exception as e:
                logging.error(f"{name}: metrics computation failed: {e}")

        if args.plot_output:
            fig, axs = plt.subplots(y_probs.shape[1]+1)
            fig.set_figwidth(y_probs.shape[0]/100)
            for i in range(y_probs.shape[1]):
                y_probs_extended = y_probs[:, i].repeat_interleave(args.subsampling)
                y_probs_postprocessed = postprocess_output(
                    y_probs[:, i].unsqueeze(1),
                    args.subsampling,
                    args.threshold,
                    args.median_window_length,
                    args.normalize_probs)
                axs[i].set_ylim([-0.1, 1.1])
                axs[i].set_xticks([])
                axs[i].plot(range(
                    y_probs_extended.shape[0]),
                    y_probs_extended, linewidth=0.5)
                axs[i].plot(range(
                    y_probs_postprocessed.shape[0]),
                    y_probs_postprocessed, 'r', linewidth=0.2)
                axs[i].title.set_text(f'{existence_probs[i].item():.20f}')
                axs[i].title.set_size(6)
                for j in range(0, y_probs_extended.shape[0], 100):
                    axs[i].axvline(
                        x=j, ymin=-0.5, ymax=1.5, c='black',
                        lw=0.25, ls=':', clip_on=False)
            if args.ref_rttms_dir:
                ref_frames, ref_spks = rttm_to_hard_labels(join(
                    args.ref_rttms_dir, f"{name}.rttm"), 100)
            for i in range(ref_frames.shape[1]):
                axs[y_probs.shape[1]].set_ylim([-0.1, 1.1])
                axs[y_probs.shape[1]].plot(range(
                    ref_frames.shape[0]),
                    (1-0.1*i)*ref_frames[:, i], linewidth=0.5)
            axs[y_probs.shape[1]].title.set_text('Reference')
            axs[y_probs.shape[1]].title.set_size(6)
            for j in range(0, y_probs_extended.shape[0], 100):
                axs[y_probs.shape[1]].axvline(
                    x=j, ymin=-0.5, ymax=1.5, c='black',
                    lw=0.25, ls=':', clip_on=False)
            plt.subplots_adjust(hspace=1)
            png_filename = join(out_dir, f"{name}.png")
            fig.savefig(png_filename, dpi=300)
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()

    if args.compute_metrics:
        metrics_dir = os.path.dirname(out_dir)  # strip the trailing "rttms"
        csv_filename = join(metrics_dir, "metrics_per_file.csv")
        with open(csv_filename, 'w', newline='', encoding='UTF-8') as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=['name'] + metric_keys)
            writer.writeheader()
            for row in per_file_metrics:
                writer.writerow(row)

        n_scored = len(per_file_metrics)
        summary_filename = join(metrics_dir, "metrics_summary.txt")
        with open(summary_filename, 'w', encoding='UTF-8') as summary_file:
            if n_scored == 0:
                summary_file.write("No files were scored.\n")
                logging.warning("--compute-metrics: no files were scored.")
            else:
                # Each key averaged over the files that actually defined it
                # (see acum_test_counts -- VAD_FA/miss and OSD_FA/miss are
                # undefined, not zero, for files with no real speech/overlap
                # respectively, and are excluded rather than included as 0).
                avg_metrics = {
                    k: (acum_test_metrics[k] / acum_test_counts[k]
                        if acum_test_counts[k] > 0 else float("nan"))
                    for k in metric_keys}
                summary_line = (
                    f"n_files={n_scored} collar={args.collar}s "
                    f"DER={avg_metrics['DER']:.2f}% "
                    f"(miss={avg_metrics['DER_miss']:.2f} "
                    f"fa={avg_metrics['DER_FA']:.2f} "
                    f"conf={avg_metrics['DER_conf']:.2f}) "
                    f"VAD(fa/miss)={avg_metrics['VAD_FA']:.2f}/"
                    f"{avg_metrics['VAD_miss']:.2f} "
                    f"(n={acum_test_counts['VAD_FA']}) "
                    f"OSD(fa/miss)={avg_metrics['OSD_FA']:.2f}/"
                    f"{avg_metrics['OSD_miss']:.2f} "
                    f"(n={acum_test_counts['OSD_FA']}) "
                    f"spk_qty(ref/pred)={avg_metrics['avg_ref_spk_qty']:.2f}/"
                    f"{avg_metrics['avg_pred_spk_qty']:.2f} "
                    f"attractor_acc={avg_metrics['attractor_accuracy']:.2f}%"
                )
                summary_file.write(summary_line + "\n")
                logging.info(f"--compute-metrics summary: {summary_line}")
        logging.info(
            f"Per-file metrics written to {csv_filename}, "
            f"summary written to {summary_filename}")
