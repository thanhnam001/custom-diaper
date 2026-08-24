#!/bin/bash
set -e

# Runs the SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer recipe
# end-to-end. Despite the directory name, this is NOT the paperlr Noam
# schedule -- see the lineage's train.yaml "NOAM SCHEDULE" section: this
# config deliberately uses noam_warmup_fraction: 0.10 with batch 80 / 16 /
# 32, matching SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h's own
# setup, not the paperlr-corrected explicit noam_model_size/
# noam_warmup_steps. The lineage also uses latents2attractors: mlp +
# attractor_diversity_loss_weight (not weighted_average) -- see train.yaml's
# "LOSS COMPOSITION" section. So versus the paperlr baseline, this ablation
# has three deliberate differences (frame encoder, loss composition, Noam
# schedule), not one -- it is not a controlled single-variable comparison
# against run_pipeline_2500h_paperlr.sh's output.
#
#   1. pretrain on 2-speaker SC (2500h), batch 80
#   2. adapt to 1-10 speakers (2500h), batch 16
#   3. + 4. finetune on MSDWild and RAMC -- IN PARALLEL, batch 32 each
#
# Configs: models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer/*.yaml
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
#   FT_GPUS=2,3 ./scripts/run_pipeline_2500h_paperlr_ebranchformer.sh
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
# almost immediately instead of retraining. CAVEAT: this lineage uses
# noam_warmup_fraction (not pinned noam_model_size/noam_warmup_steps), which
# train.py recomputes from len(train_loader) on every resume -- so do not
# change train_batchsize mid-run, or the realized warmup fraction will
# silently drift from what train.yaml documents (see that file's "NOAM
# SCHEDULE" section, and the sibling paperlr lineage's train.yaml header for
# the failure mode this caused there).
#
# Logs: each stage writes to $LOG_DIR (default ./logs/paperlr). The two
# parallel finetunes MUST be logged to separate files -- their stdout is
# interleaved otherwise and neither is readable.

CONFIG_DIR="models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer"
FT_GPUS="${FT_GPUS:-0,1}"
LOG_DIR="${LOG_DIR:-logs/paperlr_ebranchformer}"
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

run_stage "1/4 pretrain (2 speakers, 2500h, batch 80)" \
    "${CONFIG_DIR}/train.yaml" "${SC_GPU}" "${LOG_DIR}/1_pretrain.log"

run_stage "2/4 adapt (1-10 speakers, 2500h, batch 16)" \
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

echo "Pipeline complete. Next: ./scripts/run_infer_2500h_paperlr_ebranchformer.sh"
