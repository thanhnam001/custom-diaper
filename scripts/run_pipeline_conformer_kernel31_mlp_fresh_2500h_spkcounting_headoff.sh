#!/bin/bash
set -e

# Runs the SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff
# recipe: same lineage as
# scripts/run_pipeline_conformer_kernel31_mlp_fresh_2500h_spkcounting.sh,
# with the speaker-counting head turned OFF from stage 2 (adapt) onward --
# see this lineage's
# models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff/train_10spks_mlp.yaml
# header for the full rationale: a tensorboard read of
# dev_spk_counting_accuracy/loss found the head broken on MSDWild finetune
# (dev accuracy collapses to 2-4%, never recovers), and the DER is still
# far off the paper's numbers, so this broadens the earlier
# MSDWild-finetune-only ablation to remove the head from the point
# multi-speaker training actually starts, on the theory that it may be
# competing for shared attractor-representation capacity through the whole
# adapt stage, not just failing to calibrate during the short MSDWild
# finetune budget.
#
# UNLIKE the sibling script, this one does NOT run stage 1 (pretrain):
# this lineage reuses that lineage's 2500h/2-speaker/spk-counting pretrain
# checkpoint UNCHANGED (stage 1's data is fixed at 2 speakers, where the
# counting task is close to trivial, so it isn't implicated by the
# MSDWild-finetune-stage evidence) -- see train_10spks_mlp.yaml's
# init_model_path. That checkpoint must already exist before running this
# script; this script starts at stage 2 (adapt).
#   2. adapt to 1-10 speakers (2500h), speaker-counting head OFF from here on
#   3. finetune on MSDWild, then RAMC (both fork from this stage's
#      checkpoint, architecture-consistent, no allow_partial_warmstart
#      needed downstream of stage 2)
#
# Configs: models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff/*.yaml
#
# train_10spks_mlp.yaml's train_precomputed_dir/valid_precomputed_dir (and
# its init_model_path) are not yet available on every machine -- see that
# file's header. Fill those in / sync the pretrain checkpoint before
# running this script.
#
# init_epochs: 90-100 is left as-is (matches the repo's standard convention
# for a stage-1/stage-2 checkpoint).
#
# Safe to re-run: train.py auto-resumes each stage from its own latest
# checkpoint, so a stage that already reached its target epoch count exits
# almost immediately instead of retraining.
#
# Multi-GPU (DDP): set NUM_GPUS to spawn one process per GPU via
# torch.multiprocessing.spawn (no torchrun needed -- see --gpu > 1 in
# diaper/train.py). Each rank maps 1:1 onto CUDA_VISIBLE_DEVICES's ordering,
# so NUM_GPUS must match the number of device IDs you export, e.g. 2 GPUs:
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 ./scripts/run_pipeline_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff.sh
# DIST_BACKEND defaults to nccl (training only ever runs on Linux GPU boxes
# here); override to gloo if you ever need to run this on Windows or
# another box without NCCL installed.
# DIST_PORT overrides the loopback TCP rendezvous port (default 29500 in
# diaper/train.py) -- only needed if you're running more than one DDP job
# on the same machine at once and need them on different ports:
#   CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 DIST_PORT=29501 ./scripts/run_pipeline_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff.sh
# train_batchsize in each yaml is already a per-process (per-GPU) batch
# size -- DistributedSampler shards the dataset across ranks, so going from
# 1 to 2 GPUs doubles the effective global batch size at the same
# train_batchsize, same as this repo's existing DataParallel `--gpu`
# convention.

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff"
# Respects an already-set CUDA_VISIBLE_DEVICES from the calling shell (e.g.
# `CUDA_VISIBLE_DEVICES=1 ./scripts/run_pipeline_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff.sh`),
# falling back to GPU 0 only if the caller didn't set one.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
DIST_PORT="${DIST_PORT:-29500}"

run_stage () {
    local name="$1"
    local config="$2"
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting stage: ${name}"
    echo "  config: ${config}"
    echo "  gpu: ${NUM_GPUS} (dist-backend: ${DIST_BACKEND}, dist-port: ${DIST_PORT})"
    echo "=================================================================="
    python diaper/train.py -c "${config}" \
        --gpu "${NUM_GPUS}" --dist-backend "${DIST_BACKEND}" \
        --dist-port "${DIST_PORT}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished stage: ${name}"
}

run_stage "1/3 adapt (1-10 speakers, mlp, 2500h, speaker-counting head OFF)" \
    "${CONFIG_DIR}/train_10spks_mlp.yaml"

run_stage "2/3 finetune MSDWild" \
    "${CONFIG_DIR}/finetune_msdwild_10spks_mlp.yaml"

run_stage "3/3 finetune RAMC" \
    "${CONFIG_DIR}/finetune_ramc_10spks_mlp.yaml"

echo "Pipeline complete."
