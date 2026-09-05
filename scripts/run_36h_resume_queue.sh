#!/bin/bash
# 36-hour, 2-GPU queue that RESUMES the 9 2500h-family finetune runs that
# early-stopped under ~300 epochs (see memory diaper_finetune_epoch_budget_scan.md,
# produced 2026-09-05). Start it once and leave it:
#
#   mkdir -p logs/36h_resume_queue
#   nohup ./scripts/run_36h_resume_queue.sh > logs/36h_resume_queue/driver.log 2>&1 &
#
# Upload the checkpoints first with scripts/pack_resume_weights_for_36h_queue.sh
# -- every run below reads train.py's auto-resume-from-latest-checkpoint
# behaviour, so if a run's output_path has no checkpoint_*.tar on this box,
# its stage trains FROM SCRATCH instead of resuming (preflight below warns,
# does not abort by default -- see STRICT_PREFLIGHT).
#
#
# WHY THESE 9, AND WHY NOW
# ------------------------
# All 9 stopped because of early_stopping_patience, not because they
# converged with confidence:
#   - The conformer_k31 lane (A1-A5) never set early_stopping_patience in its
#     yaml, so it ran under train.py's bare default of 30 -- a full 3x
#     shorter than the patience=100 already PROVEN insufficient below.
#   - The ebf/paperlr_ebf lane (B1, B2, B4) used patience=100.
#   - B3 (paperlr vanilla MSDWild) also ran under the OLD patience=100; its
#     yaml was later edited in place to patience=350 specifically BECAUSE of
#     what this queue is retesting (see that yaml's own comment).
# The reference case: extending the E-Branchformer(mlp) subsampling-10
# MSDWild finetune from its patience-30-analogue stop (305 ep) to the full
# 750-epoch budget bought -1.46 pooled DER for free, with dev_DER staying
# genuinely flat only from ~epoch 450 onward. So "stopped early on a short
# patience" has direct local precedent for hiding real, free DER.
#
#
# LANE LAYOUT AND ORDER -- BY VALUE, so a slow box truncates the tail
# --------------------------------------------------------------------
# Lane A (gpu 0): conformer_k31 family, cheapest per epoch, 5 runs.
#   1. RAMC (base)         2. RAMC @ subsampling5   3. MSDWild (base)
#   4. RAMC @ lr 1e-5      5. RAMC @ lr 3e-6
# Lane B (gpu 1): ebf / paperlr family, pricier per epoch, 4 runs.
#   1. paperlr_ebf RAMC (base, architecture-matched-to-paper lineage)
#   2. fixednoam_ebf RAMC @ subsampling5
#   3. paperlr vanilla MSDWild (base)
#   4. fixednoam_ebf RAMC @ lr 1e-5
# RAMC before MSDWild throughout: RAMC (43 files) resolves 2-3 DER effects,
# pooled MSDWild resolves nothing below ~1.5 (diaper_der_statistical_power).
#
#
# PER-STAGE TIME BUDGET, NOT JUST RAISED PATIENCE
# -------------------------------------------------
# Unlike the 5-day queue (which had days to spare and let patience alone
# decide when a stage ends), this window is a hard 36 h across 2 GPUs, so
# every stage is wrapped in `timeout`. train.py checkpoints every epoch and
# auto-resumes, so a timeout-kill is exactly as safe as a crash: whatever
# epoch it reached is a real, keepable improvement over the original
# early-stopped checkpoint, even if raised patience never actually fires.
#
# Budgets below assume ~1.6 min/epoch (RAMC) / ~2.8 min/epoch (MSDWild) for
# the conformer_k31 lane -- MEASURED on this exact lineage during the 5-day
# queue (diaper_fixednoam_conformer_queue.md). The ebf/paperlr lane's costs
# are EXTRAPOLATED (not directly measured) with a 1.3x safety multiplier,
# and subsampling5 stages get ~2x for the doubled token count. Lane totals
# are sized to ~32-34h, leaving 2-4h of buffer per lane for setup, the two
# inline MSDWild inference stages, and slop in the estimates above.
#
# CALIBRATE EARLY: after each lane's first stage has run ~2h, check real
# throughput --
#   ls -l --time-style=full-iso <output_path>/models | tail -5
# -- and if it's wildly different from the assumption, edit this file's
# *_TIMEOUT_MIN values (or export the env var overrides below) before the
# mismatch compounds across 4-5 stages. There is no automatic recalibration.
#
# Every *_TIMEOUT_MIN is overridable, e.g.:
#   A1_TIMEOUT_MIN=300 ./scripts/run_36h_resume_queue.sh
#
#
# ENV KNOBS (subset of run_5day_queue.sh's; same meanings)
# ---------------------------------------------------------
#   LANE_A_GPU / LANE_B_GPU   physical device ids     (default 0 / 1)
#   ONLY_LANE=A|B             run a single lane
#   RAMC_INFER_DEVICE=gpu     try RAMC scoring on GPU instead of CPU
#   RAMC_CPU_THREADS          default 8
#   RAMC_LOCK_TIMEOUT         default 21600 (6h)
#   SKIP_PREFLIGHT=1          don't check for resumable checkpoints first
#   STRICT_PREFLIGHT=1        abort a lane if any of its checkpoints are missing
#   LOG_DIR                   default logs/36h_resume_queue
#   A1_TIMEOUT_MIN .. A5_TIMEOUT_MIN, B1_TIMEOUT_MIN .. B4_TIMEOUT_MIN
#                             per-stage wall-clock cap in minutes

LANE_A_GPU="${LANE_A_GPU:-0}"
LANE_B_GPU="${LANE_B_GPU:-1}"
LOG_DIR="${LOG_DIR:-logs/36h_resume_queue}"
ONLY_LANE="${ONLY_LANE:-}"

RAMC_INFER_DEVICE="${RAMC_INFER_DEVICE:-cpu}"
RAMC_CPU_THREADS="${RAMC_CPU_THREADS:-8}"
RAMC_LOCK_TIMEOUT="${RAMC_LOCK_TIMEOUT:-21600}"

DIAPER_ENV="${DIAPER_ENV:-/data/ocr/namvt17/custom-diaper/.venv}"
DSCORE_SRC="${DSCORE_SRC:-/data/ocr/namvt17/custom-diaper/dscore}"
DSCORE_ENV="${DSCORE_ENV:-/data/ocr/namvt17/custom-diaper/dscore/.dscore}"
MAX_CHECKPOINTS_TO_AVERAGE="${MAX_CHECKPOINTS_TO_AVERAGE:-10}"

CNF_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_conformer_k31
EBF_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_ebf
PLR_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr
PLE_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer

# Per-stage wall-clock caps, minutes (see header for the reasoning).
A1_TIMEOUT_MIN="${A1_TIMEOUT_MIN:-480}"   # conformer_k31 MSDWild
A2_TIMEOUT_MIN="${A2_TIMEOUT_MIN:-420}"   # conformer_k31 RAMC
A3_TIMEOUT_MIN="${A3_TIMEOUT_MIN:-270}"   # conformer_k31 RAMC lr1e-5
A4_TIMEOUT_MIN="${A4_TIMEOUT_MIN:-270}"   # conformer_k31 RAMC lr3e-6
A5_TIMEOUT_MIN="${A5_TIMEOUT_MIN:-480}"   # conformer_k31 RAMC sub5
B1_TIMEOUT_MIN="${B1_TIMEOUT_MIN:-300}"   # fixednoam_ebf RAMC lr1e-5
B2_TIMEOUT_MIN="${B2_TIMEOUT_MIN:-600}"   # fixednoam_ebf RAMC sub5
B3_TIMEOUT_MIN="${B3_TIMEOUT_MIN:-540}"   # paperlr vanilla MSDWild
B4_TIMEOUT_MIN="${B4_TIMEOUT_MIN:-600}"   # paperlr_ebf RAMC

mkdir -p "$LOG_DIR"
RAMC_LOCK="${LOG_DIR}/.ramc_infer.lock"

if [ "${USE_CONDA_RUN:-1}" = "1" ]; then
    PY=(conda run -p "$DIAPER_ENV" --no-capture-output python)
    DSCORE_PY=(conda run -p "$DSCORE_ENV" --no-capture-output python -u)
else
    PY=(python)
    DSCORE_PY=(python -u)
fi

log () { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Same signal handling as run_5day_queue.sh -- Ctrl+C alone will NOT stop
# background lane subshells (POSIX ignores SIGINT/SIGQUIT in a background job
# started by a non-interactive shell), so escalate to the process group.
#   Ctrl+C, or kill <driver pid>   -> whole group torn down
#   touch $LOG_DIR/STOP            -> no NEW stage starts; current one finishes
# ---------------------------------------------------------------------------
STOP_FILE="${LOG_DIR}/STOP"
stop_requested () { [ -f "$STOP_FILE" ]; }

on_signal () {
    trap - INT TERM
    log "caught SIG$1 -- stopping the queue"
    : > "$STOP_FILE"
    local mypgid
    mypgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    if [ -n "$mypgid" ] && [ "$mypgid" = "$$" ]; then
        log "terminating process group $mypgid"
        kill -TERM 0 2>/dev/null
        sleep 5
        kill -KILL 0 2>/dev/null
    else
        log "not a process-group leader (pgid=${mypgid:-unknown}); killing tracked children"
        local p
        for p in ${pids[@]+"${pids[@]}"}; do kill -TERM "$p" 2>/dev/null; done
        pkill -TERM -P $$ 2>/dev/null
    fi
    exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

yaml_get () { grep "^$1:" "$2" | head -1 | sed "s|^$1: *||"; }

# ---------------------------------------------------------------------------
# Cross-lane mutex for RAMC scoring (identical to run_5day_queue.sh) -- RAMC
# whole-recording inference needs ~120-130 GB RAM on the CPU path
# (diaper_ramc_infer_hardware_limit), so at most one runs at a time.
# ---------------------------------------------------------------------------
ramc_lock_acquire () {
    local waited=0
    while true; do
        if mkdir "$RAMC_LOCK" 2>/dev/null; then
            echo $$ > "$RAMC_LOCK/pid"; return 0
        fi
        local owner; owner=$(cat "$RAMC_LOCK/pid" 2>/dev/null)
        if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
            log "RAMC lock held by dead pid $owner -- reclaiming"; rm -rf "$RAMC_LOCK"; continue
        fi
        sleep 60
        if stop_requested; then
            log "STOP requested while waiting for the RAMC lock -- giving up"; return 1
        fi
        waited=$((waited + 60))
        if [ "$waited" -ge "$RAMC_LOCK_TIMEOUT" ]; then
            log "RAMC lock still held after ${waited}s -- SKIPPING this scoring"; return 1
        fi
    done
}
ramc_lock_release () { rm -rf "$RAMC_LOCK"; }

# ---------------------------------------------------------------------------
# preflight_resume <label> <output_path>
# The only thing that matters here: is there something to resume FROM. If
# not, train.py will happily start from the adapt checkpoint via
# init_model_path instead -- silently turning a "run a bit more" stage into
# a full fresh finetune, which blows the time budget and answers a different
# question. Advisory by default, same as the 5-day queue.
# ---------------------------------------------------------------------------
preflight_resume () {
    local label="$1" out="$2" n
    n=$(find "$out/models" -maxdepth 1 -name 'checkpoint_*.tar' 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        log "  MISSING checkpoints to resume: $label -> $out/models has none"
        log "  *** this stage would train FROM SCRATCH, not resume. Did the"
        log "  *** weights tar (scripts/pack_resume_weights_for_36h_queue.sh)"
        log "  *** get extracted under experiments/10attractors on this box?"
        if [ "${STRICT_PREFLIGHT:-0}" = "1" ]; then
            return 1
        fi
        return 0
    fi
    log "  ok      $label -- $n checkpoint(s) on disk, will resume"
    return 0
}

# ---------------------------------------------------------------------------
# resume_stage <label> <gpu> <cfg> <output_path> <logfile> <timeout_min> [extra args...]
# Like run_5day_queue.sh's train_stage, but time-boxed. A timeout kill (rc
# 124) is treated as a normal, expected end of this stage -- NOT a failure --
# since the point is exactly to stop after a bounded amount of extra
# training, whether or not raised patience fires first.
#
# init_model_path override -- WHY THIS MATTERS FOR A RESUME, NOT JUST A
# FRESH RUN: diaper/train.py runs `if args.init_model_path != '':
# average_checkpoints(...)` UNCONDITIONALLY, before it ever checks
# output_path for existing checkpoints to resume from -- so it always tries
# to load the config's init_model_path (the ADAPT stage's checkpoints) first,
# even though a successful resume immediately overwrites that model anyway
# a few lines later. Every one of these 9 finetune configs points
# init_model_path at its adapt-stage output, which is NOT part of what
# scripts/pack_resume_weights_for_36h_queue.sh uploads and is not guaranteed
# to still be on the server (these were downloaded locally; per
# run_5day_queue.sh's own audit, anything absent from the local checkout has
# no server copy). Left alone, a missing adapt directory would crash EVERY
# stage on startup before it ever reaches the resume logic.
#
# So: only when $output_path/models already has a checkpoint to resume from
# do we override --init-model-path '' (empty string satisfies train.py's
# `!= ''` gate, default value, so this simply skips average_checkpoints
# entirely -- harmless since its result would be discarded by the resume a
# moment later anyway). If $output_path has NO checkpoint (upload missed
# this run, or KEEP=0 on the packer), we deliberately do NOT override --
# falling back to the config's own init_model_path (a fresh finetune from
# the adapt checkpoint, if that's actually on the server) is a far better
# failure mode than silently training from random-init weights, which is
# what forcing '' unconditionally would risk.
# ---------------------------------------------------------------------------
resume_stage () {
    local label="$1" gpu="$2" cfg="$3" output_path="$4" logfile="$5" timeout_min="$6"; shift 6
    if [ ! -f "$cfg" ]; then
        log "SKIP $label -- config not found: $cfg"; return 1
    fi
    if stop_requested; then
        log "STOP requested -- not starting $label"; return 1
    fi

    local init_args=()
    if [ "$(find "$output_path/models" -maxdepth 1 -name 'checkpoint_*.tar' 2>/dev/null | wc -l)" -gt 0 ]; then
        init_args=(--init-model-path '')
    else
        log "WARN  $label -- no checkpoint at $output_path/models, NOT overriding"
        log "      init-model-path -- this will fall back to the adapt checkpoint"
        log "      (or crash there too if that's also missing on this server)"
    fi

    log "START $label gpu=$gpu cfg=$cfg cap=${timeout_min}min"
    timeout "${timeout_min}m" env CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/train.py -c "$cfg" \
        --gpu 1 "${init_args[@]}" "$@" >> "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        log "DONE  $label (patience fired or max_epochs reached before the time cap)"
        return 0
    elif [ $rc -eq 124 ]; then
        log "CAPPED $label -- hit its ${timeout_min}min wall-clock cap, moving on"
        log "       (checkpoint saved; safe to extend further later, same command)"
        return 0
    else
        log "FAIL  $label (exit $rc) -- see $logfile"
        if stop_requested; then
            log "      not retrying (STOP requested)"; return 1
        fi
        log "RETRY $label once (transient blips are nearly free -- train.py resumes)"
        timeout "${timeout_min}m" env CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/train.py -c "$cfg" \
            --gpu 1 "${init_args[@]}" "$@" >> "$logfile" 2>&1
        rc=$?
        if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then
            log "DONE  $label (after retry)"; return 0
        fi
        log "FAIL  $label (exit $rc, after retry) -- see $logfile"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# infer_stage -- same as run_5day_queue.sh's (auto-detects checkpoints,
# averages the last $MAX_CHECKPOINTS_TO_AVERAGE, scores with dscore).
# ---------------------------------------------------------------------------
infer_stage () {
    local label="$1" device="$2" cfg="$3" models_path="$4" rttms_dir="$5"
    local collar="$6" median="$7" logfile="$8"; shift 8

    if [ ! -f "$cfg" ]; then log "SKIP infer $label -- config not found: $cfg"; return 1; fi
    if [ ! -d "$models_path" ]; then log "SKIP infer $label -- no models dir at $models_path"; return 1; fi
    if stop_requested; then log "STOP requested -- not starting infer $label"; return 1; fi

    local infer_data_dir ref_rttm
    infer_data_dir=$(yaml_get infer_data_dir "$cfg")
    ref_rttm="${infer_data_dir}/rttm"

    mapfile -t ckpt_epochs < <(find "$models_path" -maxdepth 1 -name 'checkpoint_*.tar' \
        -exec basename {} \; | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n)
    if [ "${#ckpt_epochs[@]}" -eq 0 ]; then
        log "SKIP infer $label -- $models_path has no checkpoint_*.tar"; return 1
    fi
    local last_idx last_epoch start_idx first_epoch epochs_range
    last_idx=$(( ${#ckpt_epochs[@]} - 1 )); last_epoch="${ckpt_epochs[$last_idx]}"
    start_idx=$(( ${#ckpt_epochs[@]} > MAX_CHECKPOINTS_TO_AVERAGE \
                  ? ${#ckpt_epochs[@]} - MAX_CHECKPOINTS_TO_AVERAGE : 0 ))
    first_epoch="${ckpt_epochs[$start_idx]}"
    epochs_range="$(( first_epoch - 1 ))-${last_epoch}"

    local dev_args=() visible=""
    if [ "$device" = "cpu" ]; then
        dev_args=(--gpu 0 --num-threads "$RAMC_CPU_THREADS"); visible=""
    else
        dev_args=(--gpu 1 --num-threads 4); visible="$device"
    fi

    log "START infer $label device=$device epochs=$epochs_range median=$median"
    CUDA_VISIBLE_DEVICES="$visible" "${PY[@]}" diaper/infer.py -c "$cfg" \
        --models-path "$models_path" --rttms-dir "$rttms_dir" \
        --median-window-length "$median" --epochs "$epochs_range" \
        "${dev_args[@]}" "$@" >> "$logfile" 2>&1
    if [ $? -ne 0 ]; then log "FAIL  infer $label -- see $logfile"; return 1; fi

    mapfile -t sys_rttms < <(find "$rttms_dir/epochs${epochs_range}" \
        -path "*/median${median}/*/rttms/*.rttm" -type f 2>/dev/null)
    if [ "${#sys_rttms[@]}" -eq 0 ]; then
        log "WARN  no RTTMs under $rttms_dir/epochs${epochs_range} (median${median}) -- not scored"
        return 1
    fi
    local collar_args=() collar_label="0"
    if [ -n "$collar" ]; then collar_args=(--collar "$collar"); collar_label="$collar"; fi
    local score_log="${rttms_dir}/dscore_collar${collar_label}_resumed.log"
    "${DSCORE_PY[@]}" "$DSCORE_SRC/score.py" "${collar_args[@]}" \
        -r "$ref_rttm" -s "${sys_rttms[@]}" > "$score_log" 2>&1
    log "SCORED $label (${#sys_rttms[@]} files) -> $score_log"
    grep -h "OVERALL" "$score_log" || log "WARN no OVERALL line in $score_log"
    return 0
}

ramc_infer_bg () {
    local label="$1" cfg="$2" models_path="$3" rttms_dir="$4" median="$5" logfile="$6" jobfile="$7"
    if stop_requested; then log "STOP requested -- not queueing RAMC scoring: $label"; return 1; fi
    (
        if ! ramc_lock_acquire; then exit 1; fi
        trap 'ramc_lock_release' EXIT
        infer_stage "$label" "$RAMC_INFER_DEVICE" "$cfg" "$models_path" "$rttms_dir" "" "$median" "$logfile"
    ) &
    echo $! >> "$jobfile"
    log "QUEUED RAMC scoring: $label (pid $!, device=$RAMC_INFER_DEVICE)"
}

# ===========================================================================
# LANE A -- conformer_k31 (gpu $LANE_A_GPU)
# ===========================================================================
lane_A () {
    local GPU="$1" P="${LOG_DIR}/laneA" JOBS
    mkdir -p "$P"; JOBS="$P/.ramc_jobs"; : > "$JOBS"

    local out_ramc out_msd out_lr1e5 out_lr3e6 out_sub5
    out_ramc=$(yaml_get output_path "$CNF_DIR/finetune_ramc_10spks.yaml")
    out_msd=$(yaml_get output_path "$CNF_DIR/finetune_msdwild_10spks.yaml")
    out_lr1e5="${out_ramc}_lr1e-5"
    out_lr3e6="${out_ramc}_lr3e-6"
    out_sub5="${out_ramc}_sub5"

    log "LANE A on gpu $GPU (conformer_k31)"
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        preflight_resume "A2 RAMC"        "$out_ramc"  || { log "LANE A ABORTED"; return 1; }
        preflight_resume "A5 RAMC_sub5"   "$out_sub5"  || { log "LANE A ABORTED"; return 1; }
        preflight_resume "A1 MSDWild"     "$out_msd"   || { log "LANE A ABORTED"; return 1; }
        preflight_resume "A3 RAMC_lr1e-5" "$out_lr1e5" || { log "LANE A ABORTED"; return 1; }
        preflight_resume "A4 RAMC_lr3e-6" "$out_lr3e6" || { log "LANE A ABORTED"; return 1; }
    fi

    # 1. RAMC base -- highest value, first in the lane
    resume_stage "A2 RAMC (base)" "$GPU" "$CNF_DIR/finetune_ramc_10spks.yaml" "$out_ramc" \
        "$P/1_ramc.log" "$A2_TIMEOUT_MIN" --early-stopping-patience 150
    ramc_infer_bg "A2 RAMC" "$CNF_DIR/infer_ramc.yaml" \
        "$out_ramc/models" "$out_ramc/ramc_test_pred" 1 "$P/1_ramc_infer.log" "$JOBS"

    # 2. RAMC @ subsampling5 -- the proven-biggest RAMC lever elsewhere
    resume_stage "A5 RAMC @ subsampling5" "$GPU" "$CNF_DIR/finetune_ramc_10spks.yaml" "$out_sub5" \
        "$P/2_ramc_sub5.log" "$A5_TIMEOUT_MIN" --early-stopping-patience 150 \
        --subsampling 5 --num-frames 1200 --train-batchsize 16 --output-path "$out_sub5"
    ramc_infer_bg "A5 RAMC sub5" "$CNF_DIR/infer_ramc.yaml" \
        "$out_sub5/models" "$out_sub5/ramc_test_pred" 1 "$P/2_ramc_sub5_infer.log" "$JOBS"

    # 3. MSDWild base -- fits one GPU, scored inline
    resume_stage "A1 MSDWild (base)" "$GPU" "$CNF_DIR/finetune_msdwild_10spks.yaml" "$out_msd" \
        "$P/3_msdwild.log" "$A1_TIMEOUT_MIN" --early-stopping-patience 150
    infer_stage "A1 MSDWild" "$GPU" "$CNF_DIR/infer_msdwild.yaml" \
        "$out_msd/models" "$out_msd/msdwild_test_pred" "0.25" 11 "$P/3_msdwild_infer.log"

    # 4-5. RAMC LR-sweep fillers -- lowest value, last in the lane
    resume_stage "A3 RAMC @ lr1e-5" "$GPU" "$CNF_DIR/finetune_ramc_10spks.yaml" "$out_lr1e5" \
        "$P/4_ramc_lr1e-5.log" "$A3_TIMEOUT_MIN" --early-stopping-patience 150 \
        --lr 1e-5 --output-path "$out_lr1e5"
    ramc_infer_bg "A3 RAMC lr1e-5" "$CNF_DIR/infer_ramc.yaml" \
        "$out_lr1e5/models" "$out_lr1e5/ramc_test_pred" 1 "$P/4_ramc_lr1e-5_infer.log" "$JOBS"

    resume_stage "A4 RAMC @ lr3e-6" "$GPU" "$CNF_DIR/finetune_ramc_10spks.yaml" "$out_lr3e6" \
        "$P/5_ramc_lr3e-6.log" "$A4_TIMEOUT_MIN" --early-stopping-patience 150 \
        --lr 3e-6 --output-path "$out_lr3e6"
    ramc_infer_bg "A4 RAMC lr3e-6" "$CNF_DIR/infer_ramc.yaml" \
        "$out_lr3e6/models" "$out_lr3e6/ramc_test_pred" 1 "$P/5_ramc_lr3e-6_infer.log" "$JOBS"

    local jp
    while read -r jp; do [ -n "$jp" ] && wait "$jp" 2>/dev/null; done < "$JOBS"
    log "LANE A COMPLETE"
}

# ===========================================================================
# LANE B -- ebf / paperlr (gpu $LANE_B_GPU)
# ===========================================================================
lane_B () {
    local GPU="$1" P="${LOG_DIR}/laneB" JOBS
    mkdir -p "$P"; JOBS="$P/.ramc_jobs"; : > "$JOBS"

    local out_ple_ramc out_ebf_ramc out_ebf_lr1e5 out_ebf_sub5 out_plr_msd
    out_ple_ramc=$(yaml_get output_path "$PLE_DIR/finetune_ramc_10spks.yaml")
    out_ebf_ramc=$(yaml_get output_path "$EBF_DIR/finetune_ramc_10spks.yaml")
    out_ebf_lr1e5="${out_ebf_ramc}_lr1e-5"
    out_ebf_sub5="${out_ebf_ramc}_sub5"
    out_plr_msd=$(yaml_get output_path "$PLR_DIR/finetune_msdwild_10spks.yaml")

    log "LANE B on gpu $GPU (ebf / paperlr)"
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        preflight_resume "B4 paperlr_ebf RAMC"    "$out_ple_ramc"  || { log "LANE B ABORTED"; return 1; }
        preflight_resume "B2 fixednoam_ebf sub5"  "$out_ebf_sub5"  || { log "LANE B ABORTED"; return 1; }
        preflight_resume "B3 paperlr MSDWild"     "$out_plr_msd"   || { log "LANE B ABORTED"; return 1; }
        preflight_resume "B1 fixednoam_ebf lr1e-5" "$out_ebf_lr1e5" || { log "LANE B ABORTED"; return 1; }
    fi

    # 1. paperlr_ebf RAMC base -- architecture-matched-to-paper lineage
    resume_stage "B4 paperlr_ebf RAMC (base)" "$GPU" "$PLE_DIR/finetune_ramc_10spks.yaml" "$out_ple_ramc" \
        "$P/1_ple_ramc.log" "$B4_TIMEOUT_MIN" --early-stopping-patience 200
    ramc_infer_bg "B4 paperlr_ebf RAMC" "$PLE_DIR/infer_ramc.yaml" \
        "$out_ple_ramc/models" "$out_ple_ramc/ramc_test_pred" 1 "$P/1_ple_ramc_infer.log" "$JOBS"

    # 2. fixednoam_ebf RAMC @ subsampling5 -- the proven-biggest RAMC lever
    resume_stage "B2 fixednoam_ebf RAMC @ subsampling5" "$GPU" "$EBF_DIR/finetune_ramc_10spks.yaml" "$out_ebf_sub5" \
        "$P/2_ebf_ramc_sub5.log" "$B2_TIMEOUT_MIN" --early-stopping-patience 200 \
        --subsampling 5 --num-frames 1200 --train-batchsize 16 --output-path "$out_ebf_sub5"
    ramc_infer_bg "B2 ebf RAMC sub5" "$EBF_DIR/infer_ramc.yaml" \
        "$out_ebf_sub5/models" "$out_ebf_sub5/ramc_test_pred" 1 "$P/2_ebf_ramc_sub5_infer.log" "$JOBS"

    # 3. paperlr vanilla MSDWild base -- NO patience override: this run's own
    #    yaml already raised early_stopping_patience to 350 for exactly this
    #    reason (see finetune_msdwild_10spks.yaml's own comment). Will very
    #    likely be capped by the timeout, not by patience -- that's expected.
    resume_stage "B3 paperlr MSDWild (base)" "$GPU" "$PLR_DIR/finetune_msdwild_10spks.yaml" "$out_plr_msd" \
        "$P/3_plr_msdwild.log" "$B3_TIMEOUT_MIN"
    infer_stage "B3 paperlr MSDWild" "$GPU" "$PLR_DIR/infer_msdwild.yaml" \
        "$out_plr_msd/models" "$out_plr_msd/msdwild_test_pred" "0.25" 11 "$P/3_plr_msdwild_infer.log"

    # 4. fixednoam_ebf RAMC @ lr1e-5 -- lowest value, last in the lane
    resume_stage "B1 fixednoam_ebf RAMC @ lr1e-5" "$GPU" "$EBF_DIR/finetune_ramc_10spks.yaml" "$out_ebf_lr1e5" \
        "$P/4_ebf_ramc_lr1e-5.log" "$B1_TIMEOUT_MIN" --early-stopping-patience 200 \
        --lr 1e-5 --output-path "$out_ebf_lr1e5"
    ramc_infer_bg "B1 ebf RAMC lr1e-5" "$EBF_DIR/infer_ramc.yaml" \
        "$out_ebf_lr1e5/models" "$out_ebf_lr1e5/ramc_test_pred" 1 "$P/4_ebf_ramc_lr1e-5_infer.log" "$JOBS"

    local jp
    while read -r jp; do [ -n "$jp" ] && wait "$jp" 2>/dev/null; done < "$JOBS"
    log "LANE B COMPLETE"
}

# ---------------------------------------------------------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _cvd <<< "$CUDA_VISIBLE_DEVICES"
    if [ -n "${_cvd[0]:-}" ]; then LANE_A_GPU="${_cvd[0]}"; fi
    LANE_B_GPU="${_cvd[1]:-${_cvd[0]}}"
    log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> lane A gpu $LANE_A_GPU, lane B gpu $LANE_B_GPU"
    if [ "${#_cvd[@]}" -lt 2 ] && [ -z "$ONLY_LANE" ]; then
        log "WARNING: only one device listed but both lanes will run and SHARE it."
        log "WARNING: pass ONLY_LANE=A (or B), or list two devices."
    fi
    unset CUDA_VISIBLE_DEVICES
fi

log "36h resume queue starting"
log "  lane A (conformer_k31) gpu=$LANE_A_GPU  budget ~$(( (A1_TIMEOUT_MIN+A2_TIMEOUT_MIN+A3_TIMEOUT_MIN+A4_TIMEOUT_MIN+A5_TIMEOUT_MIN) / 60 ))h"
log "  lane B (ebf/paperlr)   gpu=$LANE_B_GPU  budget ~$(( (B1_TIMEOUT_MIN+B2_TIMEOUT_MIN+B3_TIMEOUT_MIN+B4_TIMEOUT_MIN) / 60 ))h"
log "  RAMC scoring device=$RAMC_INFER_DEVICE (serialized across lanes)"
log "  logs: $LOG_DIR"

rm -rf "$RAMC_LOCK"
rm -f "$STOP_FILE"

pids=()
if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "A" ]; then
    lane_A "$LANE_A_GPU" & pids+=($!)
fi
if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "B" ]; then
    lane_B "$LANE_B_GPU" & pids+=($!)
fi
for pid in "${pids[@]}"; do wait "$pid"; done

log "36h resume queue finished."
log "Pack results with scripts/pack_36h_resume_results.sh and compare each"
log "stage's new DER against its pre-resume baseline before trusting a win --"
log "RAMC resolves 2-3 DER, pooled MSDWild resolves nothing under ~1.5."
