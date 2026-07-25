#!/bin/bash
set -e

# Runs the 3-stage SC_LibriSpeech_2spk_linear_diversity recipe end-to-end on
# a single GPU (tuned for 1x V100 32GB; configs already set num_workers: 4):
#   1. pretrain on 2-speaker LibriSpeech-simulated data (500h)
#   2. adapt to 1-10 speakers (300h "maximum 10 speakers" data)
#   3. finetune on MSDWild and finetune on RAMC (two independent runs,
#      both branching off stage 2's checkpoint, run one after another
#      since there is only one GPU)
#
# Configs: models/10attractors/SC_LibriSpeech_2spk_linear_diversity/*.yaml
# Difference from the older SC_LibriSpeech_2spk / SC_LibriSpeech_2spk_adapted1-10
# recipe: latents2attractors: linear (bug-fixed, see git history) instead of
# weighted_average, attractor_diversity_loss_weight enabled, and
# noam_model_size/noam_warmup_steps auto-computed by train.py instead of
# being hand-copied from common_utils/noam_lr_calc.py output.
#
# Safe to re-run: train.py auto-resumes each stage from its own latest
# checkpoint (see the checkpoint_0.tar check in diaper/train.py), so a
# stage that already reached its target epoch count exits almost
# immediately instead of retraining.

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_linear_diversity"
export CUDA_VISIBLE_DEVICES=0

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

run_stage "1/4 pretrain (2 speakers)" \
    "${CONFIG_DIR}/train.yaml"

run_stage "2/4 adapt (1-10 speakers)" \
    "${CONFIG_DIR}/train_10spks.yaml"

run_stage "3/4 finetune (MSDWild)" \
    "${CONFIG_DIR}/finetune_msdwild_10spks.yaml"

run_stage "4/4 finetune (RAMC)" \
    "${CONFIG_DIR}/finetune_ramc_10spks.yaml"

echo "Pipeline complete."
