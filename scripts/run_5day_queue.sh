#!/bin/bash
# Five-day, 2-GPU experiment queue for the conformer / E-Branchformer
# lineages. Start it once and leave it:
#
#   nohup ./scripts/run_5day_queue.sh > logs/5day_queue/driver.log 2>&1 &
#
# Deliberately NO `set -e`. This runs unattended for days; a stage that
# fails must log and let its lane continue to the next item, not take the
# remaining four days down with it. Every stage is retried once, and
# train.py auto-resumes from its own latest checkpoint, so a retry (or a
# re-run of this whole script after a crash) picks up where it stopped
# rather than restarting.
#
#
# WHAT THIS QUEUE IS FOR
# ----------------------
# Every conformer / E-Branchformer run in private/results.csv (rows 21-25)
# was trained with noam_warmup_fraction: 0.10 -- the deficient schedule
# whose correction, on an otherwise-unchanged vanilla model, moved RAMC
# test DER 23.85 -> 20.80 (row 20). The architecture runs were hit HARDER
# than vanilla was: 4-GPU DDP with per-rank batches gave their optimiser
# ~2.5-2.9x fewer steps than a single-GPU run over the same data, while
# noam_warmup_fraction silently re-derived itself from the shortened
# per-rank loader. So "the conformers lose to vanilla" has never actually
# been tested -- it is a statement about two different optimisation
# budgets. Wave 1 below removes that confound; wave 2 attacks what is left.
#
#
# LANE LAYOUT -- TWO SINGLE-GPU LANES, NOT ONE DDP JOB
# -----------------------------------------------------
# Lane A (E-Branchformer) on one GPU, lane B (conformer k31) on the other,
# running concurrently and independently. Three reasons this beats running
# one 2-GPU DDP job at a time:
#   1. The Noam step counts in the configs are exact only when the yaml's
#      train_batchsize IS the effective batch. Under DDP the effective
#      batch multiplies by the rank count and the schedule silently drifts
#      -- which is the original bug. (If you do switch to DDP anyway,
#      halve noam_warmup_steps per doubling of ranks; noam_model_size
#      stays 1534. See the adapt configs' headers.)
#   2. A crash takes out one lane, not both.
#   3. The V100 memory ceiling (batch 96 at 600 frames, 16 at 2400) is a
#      per-device number anyway, so DDP buys throughput, not batch size.
#
#
# QUEUE ORDER (per lane, strictly by value -- a slow box truncates the
# tail rather than leaving two half-finished experiments)
# ------------------------------------------------------------------
#   0. D1  diagnostic: re-score the EXISTING finetuned checkpoint at
#          subsampling 10 / median 11                       ~1.5 h
#   1. W1  adapt, corrected Noam                            ~58 h
#   2. W1  finetune RAMC + infer + dscore                   ~10 h
#   3. W2  adapt, corrected Noam + SpecAugment              ~58 h
#   4. W2  finetune RAMC + infer + dscore                   ~10 h
#   5. W1  finetune MSDWild + infer + dscore                ~14 h
#   6. W2  finetune MSDWild + infer + dscore                ~14 h
#
# BOTH RAMC finetunes come before EITHER MSDWild finetune, on purpose.
# RAMC (43 files) resolves the 2-3 DER effects at stake here; MSDWild
# pooled DER cannot resolve anything below ~1.5 DER, so its whole current
# leaderboard ordering is inside the noise band. When you do score MSDWild,
# read it bucketed by reference speaker count (2 / 3 / 4), never pooled.
# (See MEMORY.md, diaper-der-statistical-power.)
#
#
# CALIBRATION GATE -- CHECK THIS AT HOUR 2, DO NOT SKIP
# ------------------------------------------------------
# After three adapt epochs, read the real minutes-per-epoch off the
# checkpoint mtimes:
#   ls -l --time-style=full-iso <adapt output_path>/models
# then pick a tier:
#   <= 20 min/ep  -> the queue as written, and it will get through the
#                    MSDWild finetunes too.
#   20-35 min/ep  -> the queue as written. The MSDWild finetunes may not
#                    finish. That is fine -- run with RUN_MSDWILD=0 if you
#                    would rather spend the tail on fillers.
#   > 35 min/ep   -> RUN_WAVE2=0 on one lane, or shorten only the WAVE 2
#                    adapt to 60 epochs (recompute: noam_lr_calc.py
#                    --epochs 60 --iters-per-epoch 1243 --warmup-fraction
#                    0.537 --peak-lr 9.882e-5  =>  2557 / 40050).
#                    Never shorten the WAVE 1 adapt -- a truncated
#                    schedule reintroduces the exact confound this queue
#                    exists to remove.
#
#
# ENV KNOBS
# ---------
#   LANE_A_GPU / LANE_B_GPU   physical device ids       (default 0 / 1)
#   RUN_DIAGNOSTICS=0         skip the D1 re-scores     (default 1)
#   RUN_PRETRAIN=1            also run fresh 2-spk pretrains before the
#                             adapt stages -- adds ~68 h PER LANE and is
#                             NOT the default. The default warm-starts the
#                             adapt stage from the existing 2500h pretrain
#                             checkpoints, which is both cheaper and the
#                             cleaner delta against results.csv.
#   RUN_WAVE2=0               stop after wave 1         (default 1)
#   RUN_MSDWILD=0             RAMC only                 (default 1)
#   ONLY_LANE=A|B             run a single lane
#   LOG_DIR                   default logs/5day_queue
#
# Everything is idempotent: re-running the script after an interruption
# resumes each stage and skips inference for stages with no checkpoints.

LANE_A_GPU="${LANE_A_GPU:-0}"
LANE_B_GPU="${LANE_B_GPU:-1}"
LOG_DIR="${LOG_DIR:-logs/5day_queue}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
RUN_PRETRAIN="${RUN_PRETRAIN:-0}"
RUN_WAVE2="${RUN_WAVE2:-1}"
RUN_MSDWILD="${RUN_MSDWILD:-1}"
ONLY_LANE="${ONLY_LANE:-}"

DIAPER_ENV="${DIAPER_ENV:-/data/ocr/namvt17/custom-diaper/.venv}"
DSCORE_SRC="${DSCORE_SRC:-/data/ocr/namvt17/custom-diaper/dscore}"
DSCORE_ENV="${DSCORE_ENV:-/data/ocr/namvt17/custom-diaper/dscore/.dscore}"
MAX_CHECKPOINTS_TO_AVERAGE="${MAX_CHECKPOINTS_TO_AVERAGE:-10}"

EBF_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_ebf
CNF_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_conformer_k31

# Existing (old-schedule) finetuned lineages, used only by diagnostic D1.
EBF_OLD_INFER_RAMC=models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer/infer_ramc.yaml
CNF_OLD_INFER_RAMC=models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff/infer_ramc_mlp.yaml

mkdir -p "$LOG_DIR"

if [ "${USE_CONDA_RUN:-1}" = "1" ]; then
    PY=(conda run -p "$DIAPER_ENV" --no-capture-output python)
    DSCORE_PY=(conda run -p "$DSCORE_ENV" --no-capture-output python -u)
else
    PY=(python)
    DSCORE_PY=(python -u)
fi

log () { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

yaml_get () {  # yaml_get <key> <file>
    grep "^$1:" "$2" | head -1 | sed "s|^$1: *||"
}

# ---------------------------------------------------------------------------
# train_stage <label> <gpu> <config> <logfile> [extra train.py args...]
# ---------------------------------------------------------------------------
train_stage () {
    local label="$1" gpu="$2" cfg="$3" logfile="$4"; shift 4
    local attempt
    if [ ! -f "$cfg" ]; then
        log "SKIP $label -- config not found: $cfg"
        return 1
    fi
    for attempt in 1 2; do
        log "START $label (attempt $attempt/2) gpu=$gpu cfg=$cfg"
        CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/train.py -c "$cfg" \
            --gpu 1 "$@" >> "$logfile" 2>&1
        local rc=$?
        if [ $rc -eq 0 ]; then
            log "DONE  $label"
            return 0
        fi
        log "FAIL  $label (exit $rc) -- see $logfile"
        # A second attempt is nearly free: train.py resumes from the latest
        # checkpoint, so a transient OOM/NCCL/filesystem blip costs minutes,
        # not the stage. A config or data error will fail identically twice
        # and the lane moves on.
    done
    return 1
}

# ---------------------------------------------------------------------------
# infer_stage <label> <gpu> <config> <models_path> <rttms_dir> <collar>
#             <median> <score_suffix> <logfile> [extra infer.py args...]
#
# collar: "" means no --collar flag (0 s, RAMC). "0.25" for MSDWild.
# Mirrors scripts/run_infer_*.sh: auto-detects the checkpoints on disk and
# averages the last $MAX_CHECKPOINTS_TO_AVERAGE, then scores with dscore.
# ---------------------------------------------------------------------------
infer_stage () {
    local label="$1" gpu="$2" cfg="$3" models_path="$4" rttms_dir="$5"
    local collar="$6" median="$7" suffix="$8" logfile="$9"; shift 9

    if [ ! -d "$models_path" ]; then
        log "SKIP infer $label -- no models dir at $models_path"
        return 1
    fi

    local infer_data_dir ref_rttm
    infer_data_dir=$(yaml_get infer_data_dir "$cfg")
    ref_rttm="${infer_data_dir}/rttm"

    mapfile -t ckpt_epochs < <(find "$models_path" -maxdepth 1 -name 'checkpoint_*.tar' \
        -exec basename {} \; | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n)
    if [ "${#ckpt_epochs[@]}" -eq 0 ]; then
        log "SKIP infer $label -- $models_path has no checkpoint_*.tar"
        return 1
    fi

    local last_idx last_epoch start_idx first_epoch epochs_range
    last_idx=$(( ${#ckpt_epochs[@]} - 1 ))
    last_epoch="${ckpt_epochs[$last_idx]}"
    start_idx=$(( ${#ckpt_epochs[@]} > MAX_CHECKPOINTS_TO_AVERAGE \
                  ? ${#ckpt_epochs[@]} - MAX_CHECKPOINTS_TO_AVERAGE : 0 ))
    first_epoch="${ckpt_epochs[$start_idx]}"
    # parse_epochs' "-" syntax averages (start, end] -- subtract 1 so the
    # first epoch in the window is actually included.
    epochs_range="$(( first_epoch - 1 ))-${last_epoch}"

    log "START infer $label gpu=$gpu epochs=$epochs_range median=$median"
    CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/infer.py -c "$cfg" \
        --models-path "$models_path" \
        --rttms-dir "$rttms_dir" \
        --median-window-length "$median" \
        --epochs "$epochs_range" \
        --num-threads 4 "$@" >> "$logfile" 2>&1
    if [ $? -ne 0 ]; then
        log "FAIL  infer $label -- see $logfile"
        return 1
    fi

    # Scoped to this run's own epochs<range>/ dir so a stale window from an
    # earlier run against the same rttms_dir can't be scored by accident.
    mapfile -t sys_rttms < <(find "$rttms_dir/epochs${epochs_range}" \
        -path "*/median${median}/*/rttms/*.rttm" -type f 2>/dev/null)
    if [ "${#sys_rttms[@]}" -eq 0 ]; then
        log "WARN  no RTTMs under $rttms_dir/epochs${epochs_range} (median${median}) -- not scored"
        return 1
    fi

    local collar_args=() collar_label="0"
    if [ -n "$collar" ]; then
        collar_args=(--collar "$collar")
        collar_label="$collar"
    fi

    local score_log="${rttms_dir}/dscore_collar${collar_label}${suffix}.log"
    "${DSCORE_PY[@]}" "$DSCORE_SRC/score.py" "${collar_args[@]}" \
        -r "$ref_rttm" -s "${sys_rttms[@]}" > "$score_log" 2>&1

    log "SCORED $label -> $score_log"
    grep -h "OVERALL" "$score_log" || log "WARN no OVERALL line in $score_log"
    return 0
}

# ---------------------------------------------------------------------------
# lane <letter> <config dir> <gpu> <old-lineage RAMC infer cfg for D1>
# ---------------------------------------------------------------------------
lane () {
    local L="$1" CFG="$2" GPU="$3" D1_CFG="$4"
    local P="${LOG_DIR}/lane${L}"
    mkdir -p "$P"

    local adapt_out ft_ramc_out ft_msd_out
    adapt_out=$(yaml_get output_path "$CFG/train_10spks.yaml")
    ft_ramc_out=$(yaml_get output_path "$CFG/finetune_ramc_10spks.yaml")
    ft_msd_out=$(yaml_get output_path "$CFG/finetune_msdwild_10spks.yaml")

    # Wave 2 = wave 1 + SpecAugment, written to sibling *_specaug dirs via
    # CLI overrides rather than a duplicated set of yaml files.
    local adapt2_out="${adapt_out}_specaug"
    local ft_ramc2_out="${adapt2_out}/models_finetuneRAMC"
    local ft_msd2_out="${adapt2_out}/models_finetuneMSDWILD"

    log "LANE $L on gpu $GPU -- adapt output: $adapt_out"

    # -- 0. D1 diagnostic ---------------------------------------------------
    # Re-score the EXISTING old-schedule checkpoint at MSDWild's resolution
    # settings (subsampling 10 / median 11) instead of RAMC's (5 / 1).
    # Hypothesis: the conv branches carry a fixed-size depthwise kernel, so
    # their receptive field in SECONDS halves at subsampling 5, while plain
    # self-attention (use_posenc: False) has no fixed temporal kernel and
    # does not care. If the conv variants gain >= 2 DER here and paperlr
    # gains < 1, the eval protocol has been taxing them all along -- report
    # it as an EXTRA results.csv column, never a replacement, since it
    # departs from the paper's prescribed collar-0 protocol (Section IV.D).
    if [ "$RUN_DIAGNOSTICS" = "1" ] && [ -f "$D1_CFG" ]; then
        infer_stage "D1-lane${L}" "$GPU" "$D1_CFG" \
            "$(yaml_get models_path "$D1_CFG")" \
            "$(yaml_get rttms_dir "$D1_CFG")" \
            "" 11 "_subsampling10" "$P/0_d1_subsampling10.log" \
            --subsampling 10
    fi

    # -- 1. optional fresh pretrain ----------------------------------------
    local adapt_init_args=()
    if [ "$RUN_PRETRAIN" = "1" ]; then
        train_stage "L${L} W1 pretrain (2 spk, 2500h, batch 96)" "$GPU" \
            "$CFG/train.yaml" "$P/1_pretrain.log"
        adapt_init_args=(--init-model-path "$(yaml_get output_path "$CFG/train.yaml")/models")
    fi

    # -- 2. wave 1: adapt on the corrected Noam schedule --------------------
    train_stage "L${L} W1 adapt (1-10 spk, 2500h, batch 16, noam 1534/66749)" \
        "$GPU" "$CFG/train_10spks.yaml" "$P/2_w1_adapt.log" "${adapt_init_args[@]}"

    # -- 3. wave 1: finetune RAMC + score  << THE DECIDING NUMBER >> --------
    train_stage "L${L} W1 finetune RAMC" "$GPU" \
        "$CFG/finetune_ramc_10spks.yaml" "$P/3_w1_ft_ramc.log"
    infer_stage "L${L} W1 RAMC" "$GPU" "$CFG/infer_ramc.yaml" \
        "${ft_ramc_out}/models" "${ft_ramc_out}/ramc_test_pred" \
        "" 1 "" "$P/3_w1_ft_ramc_infer.log"

    # -- 4/5. wave 2: + SpecAugment ----------------------------------------
    # The conv family fits the simulated-conversation domain far better than
    # self-attention (adapt dev DER 10.0-12.0 vs 14.67) and tests worse --
    # an overfit-to-source-domain signature, with 73% more parameters.
    # specaugment is False in every config in this repo and has never been
    # tried; the precomputed dataset applies it at load time, so it costs
    # nothing and needs no re-precompute.
    if [ "$RUN_WAVE2" = "1" ]; then
        train_stage "L${L} W2 adapt + specaugment" "$GPU" \
            "$CFG/train_10spks.yaml" "$P/4_w2_adapt.log" \
            --specaugment True --output-path "$adapt2_out" "${adapt_init_args[@]}"

        train_stage "L${L} W2 finetune RAMC + specaugment" "$GPU" \
            "$CFG/finetune_ramc_10spks.yaml" "$P/5_w2_ft_ramc.log" \
            --specaugment True \
            --init-model-path "${adapt2_out}/models" \
            --output-path "$ft_ramc2_out"
        infer_stage "L${L} W2 RAMC" "$GPU" "$CFG/infer_ramc.yaml" \
            "${ft_ramc2_out}/models" "${ft_ramc2_out}/ramc_test_pred" \
            "" 1 "" "$P/5_w2_ft_ramc_infer.log"
    fi

    # -- 6/7. MSDWild finetunes, last on purpose ---------------------------
    if [ "$RUN_MSDWILD" = "1" ]; then
        train_stage "L${L} W1 finetune MSDWild" "$GPU" \
            "$CFG/finetune_msdwild_10spks.yaml" "$P/6_w1_ft_msdwild.log"
        infer_stage "L${L} W1 MSDWild" "$GPU" "$CFG/infer_msdwild.yaml" \
            "${ft_msd_out}/models" "${ft_msd_out}/msdwild_test_pred" \
            "0.25" 11 "" "$P/6_w1_ft_msdwild_infer.log"

        if [ "$RUN_WAVE2" = "1" ]; then
            train_stage "L${L} W2 finetune MSDWild + specaugment" "$GPU" \
                "$CFG/finetune_msdwild_10spks.yaml" "$P/7_w2_ft_msdwild.log" \
                --specaugment True \
                --init-model-path "${adapt2_out}/models" \
                --output-path "$ft_msd2_out"
            infer_stage "L${L} W2 MSDWild" "$GPU" "$CFG/infer_msdwild.yaml" \
                "${ft_msd2_out}/models" "${ft_msd2_out}/msdwild_test_pred" \
                "0.25" 11 "" "$P/7_w2_ft_msdwild_infer.log"
        fi
    fi

    log "LANE $L COMPLETE"
}

# ---------------------------------------------------------------------------

log "5-day queue starting"
log "  lane A (E-Branchformer) gpu=$LANE_A_GPU  cfg=$EBF_DIR"
log "  lane B (conformer k31)  gpu=$LANE_B_GPU  cfg=$CNF_DIR"
log "  diagnostics=$RUN_DIAGNOSTICS pretrain=$RUN_PRETRAIN wave2=$RUN_WAVE2 msdwild=$RUN_MSDWILD"
log "  logs: $LOG_DIR"

pids=()
if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "A" ]; then
    lane A "$EBF_DIR" "$LANE_A_GPU" "$EBF_OLD_INFER_RAMC" &
    pids+=($!)
fi
if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "B" ]; then
    lane B "$CNF_DIR" "$LANE_B_GPU" "$CNF_OLD_INFER_RAMC" &
    pids+=($!)
fi

for pid in "${pids[@]}"; do
    wait "$pid"
done

log "5-day queue finished."
log "Compare RAMC test DER against: E-Branchformer 23.57 / conformer 23.71"
log "(their own old-schedule selves) and 20.80 (corrected-schedule vanilla)."
log "Before recording any win, run the paired bootstrap -- RAMC resolves"
log "2-3 DER, not less, and MSDWild pooled resolves nothing under ~1.5."

# ===========================================================================
# FILLERS -- run any of these in a gap, or on a lane whose queue ran dry.
# Each is finetune-or-inference only and costs under ~12 GPU-h. Not in the
# automatic queue because two of them are gated on what the D1/D2/D3
# diagnostics say. Copy-paste, adjusting <LANE_CFG> and <ADAPT_OUT>.
#
# F1  RAMC finetune at the resolution it is EVALUATED at (subsampling 5),
#     closing half of the two known train/inference mismatches. Promote
#     this into a wave-2 slot if D1 showed the conv variants gaining >= 2
#     DER at subsampling 10. (It does NOT touch the other half: RAMC files
#     are ~31 min at inference against 60 s training chunks, a ~37x length
#     extrapolation no finetune setting fixes.)
#
#   CUDA_VISIBLE_DEVICES=0 python diaper/train.py \
#     -c <LANE_CFG>/finetune_ramc_10spks.yaml --gpu 1 \
#     --subsampling 5 --num-frames 1200 --train-batchsize 16 \
#     --output-path <ADAPT_OUT>/models_finetuneRAMC_sub5
#   # then infer with --subsampling 5 (already the RAMC default)
#
# F2  Finetune LR sweep. Every finetune in this repo has used Adam at a
#     flat lr 1e-6, never swept, on models carrying 73% more parameters
#     than the one that value was chosen for. Two points, ~10 h each:
#
#   for LR in 3e-6 1e-5; do
#     CUDA_VISIBLE_DEVICES=0 python diaper/train.py \
#       -c <LANE_CFG>/finetune_ramc_10spks.yaml --gpu 1 --lr $LR \
#       --output-path <ADAPT_OUT>/models_finetuneRAMC_lr$LR
#   done
#
# F3  Attractor diversity weight 0.1 -> 0.5. Targets the confusion-shaped
#     RAMC deficit (E-Branchformer confusion 7.19 vs paperlr's 5.49, with a
#     LOWER miss rate). Only worth a slot if D2's threshold sweep comes
#     back flat and D3 shows its attractors really are less separated.
#
#   CUDA_VISIBLE_DEVICES=0 python diaper/train.py \
#     -c <LANE_CFG>/finetune_ramc_10spks.yaml --gpu 1 \
#     --attractor-diversity-loss-weight 0.5 \
#     --output-path <ADAPT_OUT>/models_finetuneRAMC_div05
#
# F4  Score the best-dev window instead of the last one. With
#     early_stopping_patience: 100 every run continues 100 epochs past its
#     best dev DER, and inference averages the LAST 10 checkpoints -- so
#     the scored window sits ~100 epochs past peak (vanilla RAMC: best
#     15.60 @ep281, scored 16.00 @ep481). Add --keep-last-n-checkpoints 0
#     to a finetune (<= 0 disables pruning), then re-infer with --epochs
#     set to the 10 around best dev. Costs disk, not compute.
