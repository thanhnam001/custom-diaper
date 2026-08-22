#!/bin/bash
set -e

# Runs the SC_LibriSpeech_2spk_2500h_paperlr recipe end-to-end: the paper's
# own baseline architecture (self_attention frame encoder,
# latents2attractors: weighted_average) at the paper's Table III data scale
# (2500h SC for both the 2-speaker pretrain and the 1-10 speaker
# adaptation), with the paper's Noam schedule *shape* restored and batch
# sizes set to 128 / 22 / 32.
#
#   1. pretrain on 2-speaker SC (2500h), batch 128
#   2. adapt to 1-10 speakers (2500h), batch 22
#   3. + 4. finetune on MSDWild and RAMC -- IN PARALLEL, batch 32 each
#
# Configs: models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr/*.yaml
# See that directory's train.yaml header for why this lineage exists and
# how its noam_model_size/noam_warmup_steps were derived. In short: the
# SC_LibriSpeech_2spk_2500h sibling took 3.75x fewer optimizer steps than
# the paper and decayed its LR to 3.4x below where the paper finishes,
# which underfits in exactly the way the paper's Table VIII documents
# ("the model tends to find less speech, increasing the missed speech rate
# considerably").
#
# WHY STAGES 3 AND 4 CAN RUN CONCURRENTLY
# ---------------------------------------
# Both finetunes read the same stage-2 adapt checkpoint (read-only, via
# init_model_path) and write to disjoint output_paths
# (.../models_finetuneMSDWILD vs .../models_finetuneRAMC). They share no
# mutable state, so there is no ordering constraint between them -- they
# were only sequential in run_pipeline_2500h.sh because that script
# predates having two GPUs free at this point in the recipe.
#
# GPU ASSIGNMENT
# --------------
# Stages 1-2 use FT_GPUS's first device. Stages 3-4 then take one device
# each from FT_GPUS. Set FT_GPUS to two comma-separated *physical* device
# ids (default "0,1"):
#
#   FT_GPUS=2,3 ./scripts/run_pipeline_2500h_paperlr.sh
#
# If you only have one GPU, set FT_GPUS to a single id and the two
# finetunes fall back to running sequentially on it (a warning is printed)
# -- running both at once on one device would thrash memory rather than
# save wall-clock time.
#
# Each stage's `gpu: 1` in the yaml only means "use CUDA" (train.py treats
# --gpu as a process count, not a device index), so CUDA_VISIBLE_DEVICES is
# what actually pins each job to a physical device.
#
# Safe to re-run: train.py auto-resumes each stage from its own latest
# checkpoint, so a stage that already hit its target epoch count exits
# almost immediately instead of retraining. That resume-safety is also why
# this lineage pins noam_model_size/noam_warmup_steps explicitly instead of
# using noam_warmup_fraction -- the fraction is recomputed from
# len(train_loader) on every resume and so silently drifts if the batch
# size ever changes mid-run (which is what happened to the sibling
# lineage; see train.yaml's header).
#
# Logs: each stage writes to $LOG_DIR (default ./logs/paperlr). The two
# parallel finetunes MUST be logged to separate files -- their stdout is
# interleaved otherwise and neither is readable.

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr"
FT_GPUS="${FT_GPUS:-0,1}"
LOG_DIR="${LOG_DIR:-logs/paperlr}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"

mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_ARR <<< "$FT_GPUS"
SC_GPU="${GPU_ARR[0]}"

run_stage () {
    local name="$1" config="$2" gpu="$3" logfile="$4"
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting stage: ${name}"
    echo "  config: ${config}"
    echo "  CUDA_VISIBLE_DEVICES=${gpu}   log: ${logfile}"
    echo "=================================================================="
    CUDA_VISIBLE_DEVICES="${gpu}" python diaper/train.py -c "${config}" \
        --gpu 1 --dist-backend "${DIST_BACKEND}" 2>&1 | tee "${logfile}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished stage: ${name}"
}

run_stage "1/4 pretrain (2 speakers, 2500h, batch 128)" \
    "${CONFIG_DIR}/train.yaml" "${SC_GPU}" "${LOG_DIR}/1_pretrain.log"

run_stage "2/4 adapt (1-10 speakers, 2500h, batch 22)" \
    "${CONFIG_DIR}/train_10spks.yaml" "${SC_GPU}" "${LOG_DIR}/2_adapt.log"

if [ "${#GPU_ARR[@]}" -lt 2 ]; then
    echo "WARNING: FT_GPUS=${FT_GPUS} lists only one device -- running the"
    echo "         two finetunes sequentially instead of in parallel."
    run_stage "3/4 finetune MSDWild (batch 32)" \
        "${CONFIG_DIR}/finetune_msdwild_10spks.yaml" "${SC_GPU}" \
        "${LOG_DIR}/3_finetune_msdwild.log"
    run_stage "4/4 finetune RAMC (batch 32)" \
        "${CONFIG_DIR}/finetune_ramc_10spks.yaml" "${SC_GPU}" \
        "${LOG_DIR}/4_finetune_ramc.log"
else
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting stages 3+4 IN PARALLEL"
    echo "  MSDWild -> GPU ${GPU_ARR[0]}   log: ${LOG_DIR}/3_finetune_msdwild.log"
    echo "  RAMC    -> GPU ${GPU_ARR[1]}   log: ${LOG_DIR}/4_finetune_ramc.log"
    echo "=================================================================="

    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python diaper/train.py \
        -c "${CONFIG_DIR}/finetune_msdwild_10spks.yaml" \
        --gpu 1 --dist-backend "${DIST_BACKEND}" \
        > "${LOG_DIR}/3_finetune_msdwild.log" 2>&1 &
    PID_MSDWILD=$!

    CUDA_VISIBLE_DEVICES="${GPU_ARR[1]}" python diaper/train.py \
        -c "${CONFIG_DIR}/finetune_ramc_10spks.yaml" \
        --gpu 1 --dist-backend "${DIST_BACKEND}" \
        > "${LOG_DIR}/4_finetune_ramc.log" 2>&1 &
    PID_RAMC=$!

    # Wait on each PID separately and capture its own exit status: a bare
    # `wait` returns the status of the LAST job only, which would silently
    # swallow a failure in the other one. `set -e` does not apply to
    # background jobs either, so both statuses must be checked by hand.
    FAILED=0
    if ! wait "$PID_MSDWILD"; then
        echo "ERROR: MSDWild finetune failed -- see ${LOG_DIR}/3_finetune_msdwild.log"
        FAILED=1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished stage: 3/4 finetune MSDWild"
    fi
    if ! wait "$PID_RAMC"; then
        echo "ERROR: RAMC finetune failed -- see ${LOG_DIR}/4_finetune_ramc.log"
        FAILED=1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished stage: 4/4 finetune RAMC"
    fi
    [ "$FAILED" -eq 0 ] || exit 1
fi

echo "Pipeline complete. Next: ./scripts/run_infer_2500h_paperlr.sh"
