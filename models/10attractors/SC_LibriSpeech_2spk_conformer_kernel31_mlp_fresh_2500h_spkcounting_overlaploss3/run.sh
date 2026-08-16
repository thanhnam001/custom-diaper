#!/usr/bin/env bash
# Runs this lineage's adapt + finetune stages in order (stage 1, the 2500h
# 2-speaker pretrain, is reused unchanged from the sibling
# mlp_fresh_2500h_spkcounting directory and is not run here -- see
# train_10spks_mlp.yaml's header for why, and for what init_model_path/
# train_precomputed_dir/valid_precomputed_dir need to point at before this
# will run at all).
#
# Run from the repo root:
#   ./models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_overlaploss3/run.sh
#
# Each `python diaper/train.py` call auto-resumes from its own output_path
# if checkpoints already exist there (see train.py's checkpoint-listing
# logic), so re-running this script after an interruption continues rather
# than restarting -- safe to just re-invoke on failure once whatever broke
# it is fixed.
#
# finetune_msdwild_10spks_mlp.yaml and finetune_ramc_10spks_mlp.yaml both
# branch independently off the adapt stage's checkpoint (see their headers)
# -- they're run sequentially here (one GPU, CUDA_VISIBLE_DEVICES=0,
# matching this repo's train.sh convention), not chained to each other. If
# running on a machine with a free GPU per stage, launch them separately in
# parallel instead of via this script.
set -e

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." >/dev/null 2>&1 && pwd )"
cd "$CDIR"

echo "=== stage 2/3: adapt (1-10 speakers, overlap_loss_weight 3.0) ==="
CUDA_VISIBLE_DEVICES=0 python diaper/train.py \
    -c models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_overlaploss3/train_10spks_mlp.yaml

echo "=== stage 3/3: finetune on MSDWild ==="
CUDA_VISIBLE_DEVICES=0 python diaper/train.py \
    -c models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_overlaploss3/finetune_msdwild_10spks_mlp.yaml

echo "=== stage 3/3: finetune on RAMC ==="
CUDA_VISIBLE_DEVICES=0 python diaper/train.py \
    -c models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_overlaploss3/finetune_ramc_10spks_mlp.yaml

echo "=== done. Evaluate with infer_msdwild_mlp.yaml / infer_ramc_mlp.yaml (see their headers for --epochs/dscore usage). ==="
