#!/bin/bash
set -e

# Runs the single finetune stage for
# models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer_subsampling5/
# finetune_ramc_10spks.yaml -- the subsampling-5 (50ms/frame) RAMC finetune
# experiment for E-Branchformer(mlp), starting from the EXISTING
# E-Branchformer paperlr adapt checkpoint (no pretrain/adapt rerun needed;
# init_model_path in the config already points at it). See that config's
# header for the full rationale: unlike the MSDWild subsampling5 sibling,
# this does not introduce a new train/infer mismatch -- RAMC has always
# been inferred at subsampling 5 by convention, so this closes a
# longstanding mismatch rather than creating one.
#
# GPU/DDP semantics, re-run safety, and usage are identical to
# run_finetune_2500h_paperlr_ebranchformer_subsampling5.sh -- see that
# script's header.
#
# Run from the repo root, on the server:
#   ./scripts/run_finetune_2500h_paperlr_ebranchformer_subsampling5_ramc.sh
# Or with more GPUs:
#   GPUS=0,1 ./scripts/run_finetune_2500h_paperlr_ebranchformer_subsampling5_ramc.sh

CONFIG="models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer_subsampling5/finetune_ramc_10spks.yaml"
GPUS="${GPUS:-0}"
LOG_DIR="${LOG_DIR:-logs/paperlr_ebranchformer_subsampling5_ramc}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"

mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS="${#GPU_ARR[@]}"

echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting: finetune E-Branchformer(mlp) RAMC @ subsampling 5"
echo "  config: ${CONFIG}"
echo "  CUDA_VISIBLE_DEVICES=${GPUS}   --gpu ${NUM_GPUS}   log: ${LOG_DIR}/finetune_ramc_subsampling5.log"
echo "=================================================================="

CUDA_VISIBLE_DEVICES="${GPUS}" python diaper/train.py -c "${CONFIG}" \
    --gpu "${NUM_GPUS}" --dist-backend "${DIST_BACKEND}" \
    2>&1 | tee "${LOG_DIR}/finetune_ramc_subsampling5.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished. Next: ./scripts/run_infer_2500h_paperlr_ebranchformer_subsampling5_ramc.sh"
