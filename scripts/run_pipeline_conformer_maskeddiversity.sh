#!/bin/bash
set -e

# Runs the 2-stage SC_LibriSpeech_2spk_adapted1-10_conformer_maskeddiversity
# recipe end-to-end on a single GPU:
#   1. adapt to 1-10 speakers (300h "maximum 10 speakers" data), starting
#      from the *existing* 2-speaker conformer (conv kernel 3, batchnorm)
#      pretrained checkpoint -- no pretrain stage here, see train_10spks.yaml
#   2. finetune on MSDWild and finetune on RAMC (two independent runs,
#      both branching off stage 1's checkpoint, run one after another
#      since there is only one GPU)
#
# Configs: models/10attractors/SC_LibriSpeech_2spk_adapted1-10_conformer_maskeddiversity/*.yaml
# Loss-composition variant of the existing conv_kernel3 conformer recipe
# (models/10attractors/SC_LibriSpeech_2spk_adapted1-10_conformer/train_conv_kernel3_10spks.yaml
# etc.): l2a_entropy_loss_weight: 0.0 (drops the latents2attractors:
# weighted_average entropy term) and attractor_diversity_loss_weight: 0.1
# (enables masked_attractor_diversity_loss instead) are the only changes --
# architecture (kernel size 3, batchnorm, weighted_average) and the 2-speaker
# pretrained checkpoint it adapts from are identical to that baseline.
#
# Safe to re-run: train.py auto-resumes each stage from its own latest
# checkpoint (see diaper/train.py's checkpoint listing before the training
# loop), so a stage that already reached its target epoch count exits
# almost immediately instead of retraining.

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_adapted1-10_conformer_maskeddiversity"
# Respects an already-set CUDA_VISIBLE_DEVICES from the calling shell (e.g.
# `CUDA_VISIBLE_DEVICES=1 ./scripts/run_pipeline_conformer_maskeddiversity.sh`),
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

run_stage "1/3 adapt (1-10 speakers)" \
    "${CONFIG_DIR}/train_10spks.yaml"

run_stage "2/3 finetune (MSDWild)" \
    "${CONFIG_DIR}/finetune_msdwild_10spks.yaml"

run_stage "3/3 finetune (RAMC)" \
    "${CONFIG_DIR}/finetune_ramc_10spks.yaml"

echo "Pipeline complete."
