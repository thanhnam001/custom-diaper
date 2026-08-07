#!/bin/bash
set -e

# Runs the SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh recipe
# end-to-end on a single GPU: a shared fresh mlp stage-1 pretrain, forked
# into two stage-2/3 pipelines that isolate
# attractor_diversity_unmasked_loss_weight as the only variable:
#   1. pretrain on 2-speaker LibriSpeech-simulated data (500h),
#      latents2attractors: mlp from the start (SHARED by both pipelines
#      below -- run once)
#   2a. pipeline 1: adapt to 1-10 speakers WITH the unmasked diversity loss
#   2b. pipeline 2: adapt to 1-10 speakers WITHOUT it (the ablation)
#   3a. pipeline 1: finetune on MSDWild, then RAMC (branch off 2a)
#   3b. pipeline 2: finetune on MSDWild, then RAMC (branch off 2b)
#
# Configs: models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh/*.yaml
# See that directory's train.yaml for why this is a separate lineage from
# SC_LibriSpeech_2spk_conformer_kernel31/ (that one patches mlp in from
# stage 2 via allow_partial_warmstart on top of a weighted_average stage-1
# pretrain; this one trains mlp from stage 1, and both pipelines below
# fork off the SAME stage-1 checkpoint so the ablation only costs stage
# 2+3, not a second 500h pretrain).
#
# init_epochs: 90-100 is left as-is in every stage-2/3 config (matches the
# paper's convention and has consistently been the well-trained range in
# this repo's runs so far) -- not something this script needs to pick.
#
# Safe to re-run: train.py auto-resumes each stage from its own latest
# checkpoint (see diaper/train.py's checkpoint listing before the training
# loop), so a stage that already reached its target epoch count exits
# almost immediately instead of retraining.

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh"
# Respects an already-set CUDA_VISIBLE_DEVICES from the calling shell (e.g.
# `CUDA_VISIBLE_DEVICES=1 ./scripts/run_pipeline_conformer_kernel31_mlp_fresh.sh`),
# falling back to GPU 0 only if the caller didn't set one.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_stage () {
    local name="$1"
    local config="$2"
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting stage: ${name}"
    echo "  config: ${config}"
    echo "=================================================================="
    python diaper/train.py -c "${config}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished stage: ${name}"
}

run_stage "1/7 pretrain (2 speakers, mlp, SHARED)" \
    "${CONFIG_DIR}/train.yaml"

run_stage "2/7 adapt (1-10 speakers, pipeline 1: mlp + unmaskeddiv)" \
    "${CONFIG_DIR}/train_10spks_mlp_unmaskeddiv.yaml"

run_stage "3/7 adapt (1-10 speakers, pipeline 2: mlp only)" \
    "${CONFIG_DIR}/train_10spks_mlp.yaml"

run_stage "4/7 finetune MSDWild (pipeline 1: mlp + unmaskeddiv)" \
    "${CONFIG_DIR}/finetune_msdwild_10spks_mlp_unmaskeddiv.yaml"

run_stage "5/7 finetune RAMC (pipeline 1: mlp + unmaskeddiv)" \
    "${CONFIG_DIR}/finetune_ramc_10spks_mlp_unmaskeddiv.yaml"

run_stage "6/7 finetune MSDWild (pipeline 2: mlp only)" \
    "${CONFIG_DIR}/finetune_msdwild_10spks_mlp.yaml"

run_stage "7/7 finetune RAMC (pipeline 2: mlp only)" \
    "${CONFIG_DIR}/finetune_ramc_10spks_mlp.yaml"

echo "Pipeline complete."
