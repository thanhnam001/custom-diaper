#!/bin/bash
set -e

# Runs the SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff_msdwildft
# ablation: same lineage as
# scripts/run_pipeline_conformer_kernel31_mlp_fresh_2500h_spkcounting.sh up
# through stage 2 (adapt), then MSDWild finetune with the speaker-counting
# head turned OFF for this stage only -- see this lineage's
# models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff_msdwildft/finetune_msdwild_10spks_mlp.yaml
# header for the full rationale (tensorboard evidence the head is broken
# specifically during MSDWild finetuning: dev accuracy collapses to 2-4%
# and never recovers, well below a trivial baseline, while train accuracy
# climbs to ~75-78%).
#
# UNLIKE the sibling scripts, this one does NOT run stage 1 (pretrain) or
# stage 2 (adapt): it reuses the plain spkcounting lineage's stage-2
# checkpoint UNCHANGED (that stage doesn't show the same red flag -- dev
# spk-counting accuracy there is merely mediocre, not collapsed) --
# finetune_msdwild_10spks_mlp.yaml's init_model_path points directly at it,
# with allow_partial_warmstart: true to drop the incompatible
# spk_counting_head.* tensors on load. That checkpoint must already exist
# before running this script.
#
# This also does NOT touch RAMC finetune: the same tensorboard read found
# the head performing well there (dev accuracy 90%+), so
# SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting's own
# finetune_ramc_10spks_mlp.yaml/RAMC results are unaffected and reused as
# they are -- nothing to rerun.
#
# Configs: models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff_msdwildft/*.yaml
#
# init_epochs: 90-100 is left as-is (matches the repo's standard convention
# for a stage-2 checkpoint).
#
# Safe to re-run: train.py auto-resumes from its own latest checkpoint, so
# a run that already reached its target epoch count exits almost
# immediately instead of retraining.
#
# Multi-GPU (DDP): set NUM_GPUS to spawn one process per GPU via
# torch.multiprocessing.spawn (no torchrun needed -- see --gpu > 1 in
# diaper/train.py). Each rank maps 1:1 onto CUDA_VISIBLE_DEVICES's ordering,
# so NUM_GPUS must match the number of device IDs you export, e.g. 2 GPUs:
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 ./scripts/run_pipeline_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff_msdwildft.sh
# DIST_BACKEND defaults to nccl (training only ever runs on Linux GPU boxes
# here); override to gloo if you ever need to run this on Windows or
# another box without NCCL installed.
# DIST_PORT overrides the loopback TCP rendezvous port (default 29500 in
# diaper/train.py) -- only needed if you're running more than one DDP job
# on the same machine at once and need them on different ports.
# train_batchsize in the yaml is already a per-process (per-GPU) batch
# size -- DistributedSampler shards the dataset across ranks, so going from
# 1 to 2 GPUs doubles the effective global batch size at the same
# train_batchsize, same as this repo's existing DataParallel `--gpu`
# convention.

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff_msdwildft"
# Respects an already-set CUDA_VISIBLE_DEVICES from the calling shell (e.g.
# `CUDA_VISIBLE_DEVICES=1 ./scripts/run_pipeline_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff_msdwildft.sh`),
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

run_stage "1/1 finetune MSDWild (speaker-counting head off)" \
    "${CONFIG_DIR}/finetune_msdwild_10spks_mlp.yaml"

echo "Pipeline complete."
