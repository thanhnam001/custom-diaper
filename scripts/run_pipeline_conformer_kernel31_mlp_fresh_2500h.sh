#!/bin/bash
set -e

# Runs the SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h recipe
# end-to-end: this is row 17 in results.csv ("Conformer base (mlp +
# conv_kernel_31 + diversity)") re-run at the paper's SC data scale (2500h
# for both the 2-speaker pretrain and the 1-10 speaker adaptation, per
# Table III steps G/H) instead of the original 500h/300h.
#   1. pretrain on 2-speaker LibriSpeech-simulated data (2500h),
#      latents2attractors: mlp from the start
#   2. adapt to 1-10 speakers (2500h), no unmasked diversity loss (matches
#      row 17, not row 18's ablation)
#   3. finetune on MSDWild, then RAMC
#
# Configs: models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h/*.yaml
#
# train.yaml / train_10spks_mlp.yaml have train_precomputed_dir /
# valid_precomputed_dir left as <placeholder> -- fill those in once the
# 2500h SC precompute finishes, before running this script.
#
# init_epochs: 90-100 is left as-is in every stage-2/3 config (matches the
# paper's convention and has consistently been the well-trained range in
# this repo's runs so far).
#
# Safe to re-run: train.py auto-resumes each stage from its own latest
# checkpoint, so a stage that already reached its target epoch count exits
# almost immediately instead of retraining.
#
# Multi-GPU (DDP): set NUM_GPUS to spawn one process per GPU via
# torch.multiprocessing.spawn (no torchrun needed -- see --gpu > 1 in
# diaper/train.py). Each rank maps 1:1 onto CUDA_VISIBLE_DEVICES's ordering,
# so NUM_GPUS must match the number of device IDs you export, e.g. 2 GPUs:
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 ./scripts/run_pipeline_conformer_kernel31_mlp_fresh_2500h.sh
# DIST_BACKEND defaults to nccl (training only ever runs on Linux GPU boxes
# here); override to gloo if you ever need to run this on Windows or
# another box without NCCL installed.
# train_batchsize in each yaml is already a per-process (per-GPU) batch
# size -- DistributedSampler shards the dataset across ranks, so going from
# 1 to 2 GPUs doubles the effective global batch size at the same
# train_batchsize, same as this repo's existing DataParallel `--gpu`
# convention. Not applicable to --speakerid-loss runs (known DDP hang, see
# commit b60f9cf) -- none of this lineage's configs set it, so it doesn't
# apply here.

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h"
# Respects an already-set CUDA_VISIBLE_DEVICES from the calling shell (e.g.
# `CUDA_VISIBLE_DEVICES=1 ./scripts/run_pipeline_conformer_kernel31_mlp_fresh_2500h.sh`),
# falling back to GPU 0 only if the caller didn't set one.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"

run_stage () {
    local name="$1"
    local config="$2"
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting stage: ${name}"
    echo "  config: ${config}"
    echo "  gpu: ${NUM_GPUS} (dist-backend: ${DIST_BACKEND})"
    echo "=================================================================="
    python diaper/train.py -c "${config}" \
        --gpu "${NUM_GPUS}" --dist-backend "${DIST_BACKEND}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished stage: ${name}"
}

run_stage "1/4 pretrain (2 speakers, mlp, 2500h)" \
    "${CONFIG_DIR}/train.yaml"

run_stage "2/4 adapt (1-10 speakers, mlp only, 2500h)" \
    "${CONFIG_DIR}/train_10spks_mlp.yaml"

run_stage "3/4 finetune MSDWild" \
    "${CONFIG_DIR}/finetune_msdwild_10spks_mlp.yaml"

run_stage "4/4 finetune RAMC" \
    "${CONFIG_DIR}/finetune_ramc_10spks_mlp.yaml"

echo "Pipeline complete."
