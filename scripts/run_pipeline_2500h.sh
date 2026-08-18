#!/bin/bash
set -e

# Runs the SC_LibriSpeech_2spk_2500h recipe end-to-end: the paper's OWN
# baseline architecture (self_attention frame encoder, latents2attractors:
# weighted_average -- no conformer/mlp/diversity-loss deviations) at the
# paper's actual Table III data scale (2500h for both the 2-speaker
# pretrain and the 1-10 speaker adaptation), followed by real-data
# finetuning. This is the genuine paper-reproduction pipeline; every other
# run_pipeline_*.sh script in this directory is a named single-variable
# ablation on top of it (conformer frame encoder, mlp attractor
# projection, kernel size, diversity loss, speaker-counting head,
# overlap-loss-weight, etc.).
#   1. pretrain on 2-speaker LibriSpeech-simulated data (2500h)
#   2. adapt to 1-10 speakers (2500h)
#   3. finetune on MSDWild
#   4. finetune on RAMC
#
# Configs: models/10attractors/SC_LibriSpeech_2spk_2500h/*.yaml
#
# train.yaml / train_10spks.yaml point at the 2500h SC precompute dirs
# already used by the conformer_kernel31_mlp_fresh_2500h lineage (same
# underlying corpus, only the downstream architecture differs) -- see
# train.yaml's header if those paths ever need to change.
#
# init_epochs: 90-100 is left as-is in every stage-2/3/4 config (matches
# the repo's standard convention -- see CLAUDE.md / MEMORY.md's
# init_epochs 90-100 convention note).
#
# Safe to re-run: train.py auto-resumes each stage from its own latest
# checkpoint, so a stage that already reached its target epoch count exits
# almost immediately instead of retraining.
#
# Multi-GPU (DDP): set NUM_GPUS to spawn one process per GPU via
# torch.multiprocessing.spawn (no torchrun needed -- see --gpu > 1 in
# diaper/train.py). Each rank maps 1:1 onto CUDA_VISIBLE_DEVICES's ordering,
# so NUM_GPUS must match the number of device IDs you export, e.g. 2 GPUs:
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 ./scripts/run_pipeline_2500h.sh
# DIST_BACKEND defaults to nccl (training only ever runs on Linux GPU boxes
# here); override to gloo if you ever need to run this on Windows or
# another box without NCCL installed.
# DIST_PORT overrides the loopback TCP rendezvous port (default 29500 in
# diaper/train.py) -- only needed if you're running more than one DDP job
# on the same machine at once and need them on different ports:
#   CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 DIST_PORT=29501 ./scripts/run_pipeline_2500h.sh
# train_batchsize in each yaml is already a per-process (per-GPU) batch
# size -- DistributedSampler shards the dataset across ranks, so going from
# 1 to 2 GPUs doubles the effective global batch size at the same
# train_batchsize, same as this repo's existing DataParallel `--gpu`
# convention.

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_2500h"
# Respects an already-set CUDA_VISIBLE_DEVICES from the calling shell (e.g.
# `CUDA_VISIBLE_DEVICES=1 ./scripts/run_pipeline_2500h.sh`), falling back
# to GPU 0 only if the caller didn't set one.
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

run_stage "1/4 pretrain (2 speakers, self-attention, 2500h)" \
    "${CONFIG_DIR}/train.yaml"

run_stage "2/4 adapt (1-10 speakers, self-attention, 2500h)" \
    "${CONFIG_DIR}/train_10spks.yaml"

run_stage "3/4 finetune MSDWild" \
    "${CONFIG_DIR}/finetune_msdwild_10spks.yaml"

run_stage "4/4 finetune RAMC" \
    "${CONFIG_DIR}/finetune_ramc_10spks.yaml"

echo "Pipeline complete."
