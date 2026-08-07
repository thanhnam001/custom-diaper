#!/usr/bin/env python3

# Copyright 2023 Brno University of Technology (author: Federico Landini, Mireia Diez)
# Licensed under the MIT license.

from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, Union
import torch


def calculate_metrics(
    target: torch.Tensor,
    decisions: torch.Tensor,
    threshold: float = 0.5,
    collar_frames: int = 0,
    round_digits: int = 2,
    return_denominators: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]]]:
    """`collar_frames` (default 0, i.e. no change from the original
    no-collar behavior): number of frames on each side of a reference
    speaker-turn boundary to exclude from scoring, mirroring dscore/
    md-eval.pl's forgiveness collar. A "boundary" is any frame whose
    reference speaker-activity row differs from the previous frame's (or
    frame 0, treated as a boundary since it's the start of a scored
    segment); frames within `collar_frames` of any boundary are dropped
    from both the error numerators and the scored-frame denominators for
    that sequence before anything else is computed. Converting a collar in
    seconds to `collar_frames` (round(collar_seconds / frame_period)) is the
    caller's job, since this function doesn't know the frame period."""
    epsilon = 1e-6
    res = {}
    decisions = (decisions > threshold).float()
    res["avg_ref_spk_qty"] = 0
    res["avg_pred_spk_qty"] = 0
    # Kept as tensors (not plain 0) from the start: with a large enough
    # collar_frames every sequence in the batch can end up with zero frames
    # left to score (all `continue`d), in which case these are never
    # reassigned by a tensor `+=` below and torch.round() further down
    # needs a Tensor, not a plain int, to not blow up. Placed on
    # target.device (not the default CPU) since the `+=` below adds
    # GPU-resident per-sequence sums (from target/decisions) onto these
    # during training -- a CPU-initialized tensor would make that `+=`
    # fail with "Expected all tensors to be on the same device".
    res["DER_miss"] = torch.tensor(0.0, device=target.device)
    res["DER_FA"] = torch.tensor(0.0, device=target.device)
    res["DER_conf"] = torch.tensor(0.0, device=target.device)
    res["DER"] = torch.tensor(0.0, device=target.device)
    res["VAD_FA"] = 0
    res["VAD_miss"] = 0
    res["OSD_FA"] = 0
    res["OSD_miss"] = 0
    # Each error is accumulated per sequence as they might need
    # different masking. Each sequence counts for the errors independently
    # and the total speech/overlap counts are acumulated.
    # Final values are estimated for the batch and returned.
    active_frames_tot = 0
    speech_frames_tot = 0
    overlap_frames_tot = 0
    for seq_num in range(target.shape[0]):
        # Remove padding positions
        boundary = min(torch.cat((
            torch.where(target[seq_num, :, 0] == -1)[0].to("cpu"),
            torch.tensor([(target[seq_num].shape[0])]))))
        t_seq = target[seq_num, :boundary, :]
        d_seq = decisions[seq_num, :boundary, :]

        if collar_frames > 0 and t_seq.shape[0] > 0:
            changed = (t_seq[1:] != t_seq[:-1]).any(dim=1)
            is_boundary = torch.zeros(t_seq.shape[0], device=t_seq.device)
            is_boundary[0] = 1.0
            is_boundary[1:] = changed.float()
            # Dilate each boundary frame by +/-collar_frames via a max-pool
            # (implicit -inf padding at the sequence edges keeps the collar
            # from wrapping/extending past frame 0 or the last frame).
            collared = torch.nn.functional.max_pool1d(
                is_boundary.view(1, 1, -1),
                kernel_size=2 * collar_frames + 1, stride=1,
                padding=collar_frames).view(-1)
            keep = collared < 0.5
            t_seq = t_seq[keep]
            d_seq = d_seq[keep]

        if t_seq.shape[0] == 0:
            continue

        cost_mx = -d_seq.unsqueeze(0).permute(0, 2, 1).bmm(
            t_seq.unsqueeze(0)) + d_seq.unsqueeze(0).permute(0, 2, 1).bmm(
            1-t_seq.unsqueeze(0))
        pred_alig, ref_alig = linear_sum_assignment(cost_mx[0].to("cpu"))
        t_seq = t_seq[:, ref_alig]
        diff = d_seq - t_seq
        # negative values will be misses and positives will be false alarms
        diff_sum = diff.sum(axis=1)
        miss_counts = -diff_sum[torch.where(diff_sum < 0)].sum()
        fa_counts = diff_sum[torch.where(diff_sum > 0)].sum()
        conf_counts = ((torch.abs(diff).sum(axis=1) -
                        torch.abs(diff.sum(axis=1)))/2).sum()
        res["DER_miss"] += miss_counts
        res["DER_FA"] += fa_counts
        res["DER_conf"] += conf_counts
        res["DER"] += miss_counts + fa_counts + conf_counts

        ref_spk_qty = t_seq.sum(axis=1)
        pred_spk_qty = d_seq.sum(axis=1)
        res["avg_ref_spk_qty"] += torch.mean(ref_spk_qty.double())
        res["avg_pred_spk_qty"] += torch.mean(pred_spk_qty.double())
        # active_frames has frames where at least one speaker is active
        active_frames_tot += torch.where(ref_spk_qty != 0)[0].shape[0]
        # speech_frames has #frames with speech (if n active speakers, n times)
        speech_frames_tot += t_seq.sum()
        # overlap_frames has frames where at least two speakers are active
        overlap_frames_tot += torch.where(ref_spk_qty > 1)[0].shape[0]

        res["VAD_FA"] += torch.where(ref_spk_qty[torch.where(
            pred_spk_qty > 0)[0]] < 1)[0].shape[0]
        res["VAD_miss"] += torch.where(pred_spk_qty[torch.where(
            ref_spk_qty > 0)[0]] < 1)[0].shape[0]

        res["OSD_FA"] += torch.where(ref_spk_qty[torch.where(
            pred_spk_qty > 1)[0]] < 2)[0].shape[0]
        res["OSD_miss"] += torch.where(pred_spk_qty[torch.where(
            ref_spk_qty > 1)[0]] < 2)[0].shape[0]

    # divide by the numerators estimated in the whole batch
    res["DER_miss"] = torch.round(100 * res["DER_miss"] / (
        epsilon + speech_frames_tot) * 10**round_digits) / (10**round_digits)
    res["DER_FA"] = torch.round(100 * res["DER_FA"] / (
        epsilon + speech_frames_tot) * 10**round_digits) / (10**round_digits)
    res["DER_conf"] = torch.round(100 * res["DER_conf"] / (
        epsilon + speech_frames_tot) * 10**round_digits) / (10**round_digits)
    res["DER"] = torch.round(100 * res["DER"] / (
        epsilon + speech_frames_tot) * 10**round_digits / (10**round_digits))
    res["VAD_FA"] = round(100 * res["VAD_FA"] / (
        epsilon + active_frames_tot), 2)
    res["VAD_miss"] = round(100 * res["VAD_miss"] / (
        epsilon + active_frames_tot), 2)
    res["OSD_FA"] = round(100 * res["OSD_FA"] / (
        epsilon + overlap_frames_tot), 2)
    res["OSD_miss"] = round(100 * res["OSD_miss"] / (
        epsilon + overlap_frames_tot), 2)
    res["avg_ref_spk_qty"] = res["avg_ref_spk_qty"] / target.shape[0]
    res["avg_pred_spk_qty"] = res["avg_pred_spk_qty"] / target.shape[0]

    if return_denominators:
        # VAD_FA/VAD_miss and OSD_FA/OSD_miss are normalized by
        # active_frames_tot/overlap_frames_tot respectively, which is a
        # different population than the one their FA numerator counts
        # come from -- so when that denominator is (near) zero the `epsilon`
        # above doesn't make the rate "undefined-but-harmless", it makes it
        # blow up to an arbitrarily large multiple of any nonzero FA count.
        # Fine when pooled over a whole training batch (rarely exactly
        # zero); not fine per-file (e.g. any single-speaker file has
        # overlap_frames_tot == 0), where callers doing a per-file average
        # need these raw counts to detect and exclude that case rather than
        # silently average in numbers like 1e8%.
        denominators = {
            "speech_frames_tot": float(speech_frames_tot),
            "active_frames_tot": float(active_frames_tot),
            "overlap_frames_tot": float(overlap_frames_tot),
        }
        return res, denominators
    return res


def new_metrics() -> Dict[str, float]:
    metrics = {}
    for k in [
        'loss',
        'activation_loss_BCE',
        'l2a_entropy_term',
        'activation_loss_DER',
        'attractor_existence_loss',
        'attractor_accuracy',
        'att_qty_loss',
        'vad_loss',
        'osd_loss',
        'spkid_loss',
        'attractor_diversity_loss',
        'attractor_diversity_unmasked_loss',
        'avg_ref_spk_qty',
        'avg_pred_spk_qty',
        'DER_FA',
        'DER_miss',
        'DER_conf',
        'DER',
        'VAD_FA',
        'VAD_miss',
        'OSD_FA',
        'OSD_miss'
    ]:
        metrics[k] = 0.0
    return metrics


def reset_metrics(acum_dict: Dict[str, float]) -> Dict[str, float]:
    for k in acum_dict.keys():
        acum_dict[k] = 0.0
    return acum_dict


def update_metrics(
    acum_dict: Dict[str, float],
    new_dict: Dict[str, float]
) -> Dict[str, float]:
    for k in new_dict.keys():
        assert (k in acum_dict), \
            f"The key {k} is not defined in the dictionary \
            where metrics are accumulated."
        acum_dict[k] += new_dict[k]
    return acum_dict
