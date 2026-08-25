#!/bin/bash
set -e
set -o pipefail

# NOTE: pipefail matters here specifically because run_stage() pipes
# through `tee` -- `cmd | tee file` on its own reports tee's exit status,
# not cmd's, so `set -e` alone would NOT stop the pipeline if a
# `python diaper/train.py` stage crashed or OOM'd partway through: the
# script would silently continue to the next stage using whatever
# (possibly incomplete) checkpoint that stage left behind.

# Runs the SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer recipe
# end-to-end, one stage after another (pretrain -> adapt -> finetune
# MSDWild -> finetune RAMC), with every stage spanning the full $GPUS pool
# via DistributedDataParallel. Despite the directory name, this is NOT the
# paperlr Noam schedule -- see the lineage's train.yaml "NOAM SCHEDULE"
# section: this config deliberately uses noam_warmup_fraction: 0.10 with
# batch 80 / 16 / 32 (per rank -- see "EFFECTIVE BATCH SIZE" below), matching
# SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h's own setup, not the
# paperlr-corrected explicit noam_model_size/noam_warmup_steps. The lineage
# also uses latents2attractors: mlp + attractor_diversity_loss_weight (not
# weighted_average) -- see train.yaml's "LOSS COMPOSITION" section. So
# versus the paperlr baseline, this ablation has three deliberate
# differences (frame encoder, loss composition, Noam schedule), not one --
# it is not a controlled single-variable comparison against
# run_pipeline_2500h_paperlr.sh's output.
#
# Configs: models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer/*.yaml
#
# WHY EVERYTHING IS SEQUENTIAL NOW
# ---------------------------------
# An earlier version of this script ran the two finetunes concurrently on
# separate single GPUs (they have no ordering constraint -- both only read
# stage 2's checkpoint and write to disjoint output_paths). That design
# assumed at most 2 GPUs. With every stage now claiming the whole $GPUS
# pool via DDP, there is no GPU left over to run two stages at once, so all
# 4 stages run strictly one after another instead.
#
# GPU ASSIGNMENT AND --gpu SEMANTICS
# -----------------------------------
# GPUS (default "0,1") is a comma-separated list of *physical* device ids.
# Every stage sees CUDA_VISIBLE_DEVICES=$GPUS and is launched with
# --gpu <N> where N = number of ids in GPUS:
#
#   GPUS=0,1,2,3 ./scripts/run_pipeline_2500h_paperlr_ebranchformer.sh
#
# train.py's --gpu is a device COUNT, not a boolean or a device index: it
# only spawns DistributedDataParallel processes when N > 1 (train.py
# ~line 1314: `world_size = args.gpu if args.gpu > 1 else 1`), so GPUS=0
# alone runs an ordinary single-GPU job with no DDP involved, and any N > 1
# spans N GPUs for every stage via DDP.
#
# EFFECTIVE BATCH SIZE WARNING
# ------------------------------
# DDP does not change train_batchsize in the yaml -- each of the N ranks
# still processes exactly train_batchsize examples per step -- but
# gradients are averaged across ranks every step, so the EFFECTIVE batch an
# optimizer step sees is train_batchsize * N. At GPUS=0,1,2,3 (N=4) that is
# an effective 320 for pretrain (train_batchsize: 80) and 64 for adapt
# (train_batchsize: 16), not the 80/16 that train.yaml/train_10spks.yaml
# document and that noam_warmup_fraction: 0.10 was matched against (see
# those files' "NOAM SCHEDULE" section -- this lineage deliberately does
# NOT use the paperlr batch-scaled Noam correction). noam_warmup_fraction
# is recomputed from the (now ~4x shorter) per-rank train_loader at the
# start of each run, so the WARMUP FRACTION still comes out at 10%, but
# noam_peak_lr stays fixed at its default (9.882e-5) regardless of the
# larger effective batch -- no sqrt-scaling is applied automatically the
# way the paperlr lineage's train.yaml derives one by hand. This was a
# known, accepted tradeoff when this script was changed to use DDP, not an
# oversight -- if results look like Table VIII's underfitting symptom
# (excess missed speech), revisit --noam-peak-lr before re-adding a batch
# correction. Finetune stages (Adam, flat lr: 1e-6) are unaffected by any
# of this -- DDP does not change Adam's LR.
#
# Safe to re-run: train.py auto-resumes each stage from its own latest
# checkpoint, so a stage that already hit its target epoch count exits
# almost immediately instead of retraining. Do not change GPUS (or the
# yaml's train_batchsize) mid-run across a resume of the SAME stage, or the
# realized Noam warmup step count will not match earlier checkpoints'
# schedule.
#
# Logs: each stage writes to $LOG_DIR (default ./logs/paperlr_ebranchformer).

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer"
GPUS="${GPUS:-0,1}"
LOG_DIR="${LOG_DIR:-logs/paperlr_ebranchformer}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"

mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS="${#GPU_ARR[@]}"

run_stage () {
    local name="$1" config="$2" logfile="$3"
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting stage: ${name}"
    echo "  config: ${config}"
    echo "  CUDA_VISIBLE_DEVICES=${GPUS}   --gpu ${NUM_GPUS}   log: ${logfile}"
    echo "=================================================================="
    CUDA_VISIBLE_DEVICES="${GPUS}" python diaper/train.py -c "${config}" \
        --gpu "${NUM_GPUS}" --dist-backend "${DIST_BACKEND}" 2>&1 | tee "${logfile}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished stage: ${name}"
}

run_stage "1/4 pretrain (2 speakers, 2500h, batch 80/rank)" \
    "${CONFIG_DIR}/train.yaml" "${LOG_DIR}/1_pretrain.log"

run_stage "2/4 adapt (1-10 speakers, 2500h, batch 16/rank)" \
    "${CONFIG_DIR}/train_10spks.yaml" "${LOG_DIR}/2_adapt.log"

run_stage "3/4 finetune MSDWild (batch 32/rank)" \
    "${CONFIG_DIR}/finetune_msdwild_10spks.yaml" "${LOG_DIR}/3_finetune_msdwild.log"

run_stage "4/4 finetune RAMC (batch 32/rank)" \
    "${CONFIG_DIR}/finetune_ramc_10spks.yaml" "${LOG_DIR}/4_finetune_ramc.log"

echo "Pipeline complete. Next: ./scripts/run_infer_2500h_paperlr_ebranchformer.sh"
