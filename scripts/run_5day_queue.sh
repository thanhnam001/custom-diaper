#!/bin/bash
# Five-day, 2-GPU experiment queue for the conformer / E-Branchformer
# lineages. Start it once and leave it:
#
#   mkdir -p logs/5day_queue
#   nohup ./scripts/run_5day_queue.sh > logs/5day_queue/driver.log 2>&1 &
#
# (the mkdir matters: the shell opens that redirect before the script runs,
# so without it the redirect fails and nothing starts)
#
# Deliberately NO `set -e`. This runs unattended for days; a stage that
# fails must log and let its lane continue to the next item, not take the
# remaining four days down with it. Every training stage is retried once,
# and train.py auto-resumes from its own latest checkpoint, so a retry (or
# a re-run of this whole script after a crash) picks up where it stopped.
#
#
# WHAT THIS QUEUE IS FOR
# ----------------------
# The conformer and E-Branchformer lineages (results.csv rows 21-25) differ
# from the winning paperlr baseline (row 20) in SIX ways, not one -- so
# "the conv variants lose to vanilla" has never been a statement about
# frame encoders. Two of those six are things the PAPER'S OWN Table VIII
# measures as harmful:
#
#   intermediate_loss_perceiver: False   8.43 vs 7.96 DER  = +0.47
#     (turned off from the adapt stage onward in both conv lineages; their
#      own pretrains, and every paperlr stage, have it True)
#   l2a entropy loss Le effectively off  8.02 vs 7.96 DER  = +0.06
#     (not a decision -- latents2attractors: mlp hard-zeros the term at
#      models.py:1300; it is only computed for weighted_average)
#
# So before the encoder is even considered, the conv variants were carrying
# roughly 0.5 DER of self-inflicted handicap. This queue removes what can
# be removed without retraining the pretrain -- intermediate_loss_perceiver
# back on, plus the corrected Noam schedule -- and measures the result.
#
# The Noam correction is included as a CONTROL, not as the expected fix.
# The E-Branchformer already beats paperlr on RAMC dev BCE (0.353 vs 0.492)
# and dev attractor-existence loss (0.095 vs 0.155) while losing DER, so it
# is not underfit and "more optimizer steps" is not the lever. It is in
# here because leaving it out would leave the comparison uninterpretable.
#
# NOT in this queue, and why:
#   SpecAugment          -- the paper tried it: "no improvement" (Sec. VI).
#   speaker-id loss      -- the paper tried it: "slightly worse results".
#   length normalization -- the paper tried it: "worse performance".
#   overlap_loss_weight 3 -- its evidence rests on a results.csv mislabel;
#                            row 24 (the 23.71/18.12 run) has no overlap
#                            loss at all, and the runs that did use it kept
#                            the counting head on and scored worst.
#   noise/reverb augmentation -- the SC data already has MUSAN background
#                            noise at random SNR plus reverberation.
#
#
# PREREQUISITE WEIGHTS AND DATA -- AUDITED 2026-08-28
# -----------------------------------------------------
# Anything absent from the local Windows checkout is GONE -- no server copy,
# and no pretrain or adapt stage can be resumed from it. Every dependency
# below was checked locally for existence, epoch coverage, and torch.load.
#
# CHECKPOINTS -- ALL FIVE PRESENT AND INTACT. Nothing needs a rerun.
#   warm-start A   ..._2500h_paperlr_ebf_pretrain2500h_mlp/models
#                  30 ckpts ep 71-100, 903 MB (301 MB pruned to last 10),
#                  covers init_epochs 90-100, loads OK
#   warm-start B   ..._conformer_kernel31_mlp_fresh_2500h_spkcounting_pretrain2500h/models
#                  10 ckpts ep 91-100, 269 MB, covers init_epochs, loads OK
#   D1 lane A      ..._paperlr_ebf_adapted.../models_finetuneRAMC/models
#                  30 ckpts ep 178-207, 2,708 MB (903 MB pruned), loads OK
#   D1 lane B      ..._headoff/models_finetuneRAMC/models
#                  10 ckpts ep 273-282, 805 MB, loads OK
#   D1 control     ..._paperlr_adapted.../models_finetuneRAMC/models
#                  10 ckpts ep 322-331, 524 MB, loads OK
#   -> ~2.8 GB total to upload if each is pruned to its last 10 checkpoints.
#
# FINETUNE DATA -- PRESENT LOCALLY, and verifiably the caches the published
# runs used (their dev chunk counts match the TensorBoard step spacing):
#   ramc_precomputed_6000frames      9,180 train / 607 dev   (9.3 GB, 7.9 GB zipped)
#   msdwild_precompute_6000frames    5,202 train / 331 dev   (3.5 GB zipped)
#   ramc/kaldi/test, msdwild/kaldi/test -- complete, for inference
#
# SC DATA -- confirmed by the user as always present on the server, so the
# 2500h caches are not a staging concern. Worth knowing anyway: they are NOT
# on the local machine (E:/datasets holds only the 300h/500h equivalents),
# so the server is the only copy apart from the S3 tars
# (s3-b200:ttnt-data/ocr/namvt17/*.tar). The preflight still counts the
# adapt cache's chunks -- not to check it exists, but because
# noam_warmup_steps: 66749 is derived from 19,888 chunks/epoch and a cache
# that differs would silently change the realized schedule.
#
# Because nothing is recoverable, do not let a stage overwrite an
# output_path that already holds checkpoints you still want: every stage
# below writes to a NEW _fixednoam_* experiment dir for exactly that reason.
#
# LANE LAYOUT -- TWO SINGLE-GPU LANES, NOT ONE DDP JOB
# -----------------------------------------------------
# Lane A (E-Branchformer) on one GPU, lane B (conformer k31) on the other.
#   1. The Noam step counts in the configs are exact only when the yaml's
#      train_batchsize IS the effective batch. Under DDP the effective
#      batch multiplies by the rank count and the schedule silently drifts
#      -- which is the original bug. (If you switch to DDP anyway, halve
#      noam_warmup_steps per doubling of ranks; noam_model_size stays 1534.)
#   2. A crash takes out one lane, not both.
#   3. The V100 memory ceiling (batch 96 at 600 frames, 16 at 2400) is a
#      per-device number, so DDP buys throughput, not batch size.
#
#
# RAMC INFERENCE RUNS ON CPU, AND ONLY ONE AT A TIME
# ---------------------------------------------------
# Whole-recording RAMC inference (num_frames: -1 at subsampling 5, ~31-min
# median recordings) does not fit a V100 and needs ~120-130 GB of RAM on
# the CPU path. Measured: ~50 minutes for all 43 test files. So the cost
# is RAM, not time -- but two concurrent runs would ask for ~250 GB and
# take the box down, not just the job. Every RAMC inference here therefore
# takes a cross-lane lock and they queue behind each other.
#
# They also run in the BACKGROUND: a RAMC scoring job uses no GPU, so its
# lane moves straight on to the next training stage instead of idling.
# Each lane waits for its own outstanding jobs before reporting complete.
# At ~50 min each and 4 per lane, they never become the critical path.
#
# Note infer.py's --gpu is a DEVICE FLAG, not a count like train.py's:
# >= 1 means CUDA, anything lower means CPU. Hence --gpu 0 below.
#
# MSDWild inference does fit on one GPU and runs inline on the lane's GPU.
#
# Not used here, on purpose: --fallback-cpu-oom. Its own help text warns
# the CPU retry has no memory ceiling and can trigger an OS-level OOM that
# kills unrelated processes (sshd, the other lane) rather than just itself.
# If you want a runaway guard instead, set RAMC_MAX_INPUT_FRAMES -- but be
# aware that skipping a recording changes the scored file set, so the DER
# is then not comparable to any other row.
#
#
# QUEUE ORDER (per lane, strictly by value -- a slow box truncates the
# tail rather than leaving two half-finished experiments)
# ------------------------------------------------------------------
#   bg. D1  re-score the EXISTING checkpoint at subsampling 10 / median 11
#   1.  adapt: intermediate_loss_perceiver ON + corrected Noam   ~58 h
#   2.  finetune RAMC   -> background CPU scoring                ~10 h
#   3.  finetune MSDWild + inline GPU inference                  ~14 h
#   4.  F1 finetune RAMC at subsampling 5 (train/infer match)    ~10 h
#   5.  F2 finetune RAMC at lr 3e-6 (the never-swept constant)   ~10 h
#   6.  F2 finetune RAMC at lr 1e-5                              ~10 h
#
# The RAMC finetune comes before the MSDWild one on purpose. RAMC (43
# files) resolves the 2-3 DER effects at stake here; MSDWild pooled DER
# cannot resolve anything below ~1.5 DER, so its whole current leaderboard
# ordering is inside the noise band. When you do score MSDWild, read it
# bucketed by reference speaker count (2 / 3 / 4), never pooled.
#
#
# CALIBRATION GATE -- CHECK THIS AT HOUR 2, DO NOT SKIP
# ------------------------------------------------------
# After three adapt epochs, read the real minutes-per-epoch off the
# checkpoint mtimes:
#   ls -l --time-style=full-iso <adapt output_path>/models
#   <= 20 min/ep  -> everything below fits comfortably.
#   20-35 min/ep  -> the queue as written; the fillers may not finish.
#   > 35 min/ep   -> RUN_FILLERS=0, and consider RUN_MSDWILD=0.
# Never shorten the adapt stage itself -- a truncated schedule reintroduces
# the exact confound this queue exists to remove.
#
#
# ENV KNOBS
# ---------
#   LANE_A_GPU / LANE_B_GPU   physical device ids       (default 0 / 1)
#                             An exported CUDA_VISIBLE_DEVICES is honoured
#                             too -- CUDA_VISIBLE_DEVICES=2,3 puts lane A on
#                             gpu 2 and lane B on gpu 3. LANE_*_GPU wins if
#                             both are given.
#   RUN_DIAGNOSTICS=0         skip the D1 re-scores     (default 1)
#   RUN_PRETRAIN=1            fresh 2-spk pretrains first; adds ~68 h PER
#                             LANE and is NOT the default -- the default
#                             warm-starts adapt from the existing 2500h
#                             pretrain checkpoints
#   RUN_MSDWILD=0             RAMC only                 (default 1)
#   RUN_FILLERS=0             skip F1/F2                (default 1)
#   ONLY_LANE=A|B             run a single lane
#   STRICT_PREFLIGHT=1        abort a lane if a dependency looks wrong.
#                             Default is ADVISORY: problems are logged and
#                             the lane starts anyway, so a false positive
#                             can never cost an unattended five-day run.
#   SKIP_PREFLIGHT=1          don't run the dependency check at all
#   ADAPT_EXPECTED_CHUNKS     chunks/epoch the Noam steps assume (19888)
#   RAMC_INFER_DEVICE=gpu     try RAMC inference on the GPU anyway
#                             (default cpu). Worth trying at subsampling
#                             10, where the token count halves and the
#                             attention matrices shrink ~4x.
#   RAMC_CPU_THREADS          torch threads for CPU inference (default 8),
#                             kept below the core count so the training
#                             lanes' dataloader workers are not starved
#   RAMC_MAX_INPUT_FRAMES     runaway guard, -1 = off (default)
#   LOG_DIR                   default logs/5day_queue

# Remember whether the caller pinned the lanes explicitly, before defaulting.
_LANE_A_EXPLICIT="${LANE_A_GPU:-}"
_LANE_B_EXPLICIT="${LANE_B_GPU:-}"
LANE_A_GPU="${LANE_A_GPU:-0}"
LANE_B_GPU="${LANE_B_GPU:-1}"
LOG_DIR="${LOG_DIR:-logs/5day_queue}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
RUN_PRETRAIN="${RUN_PRETRAIN:-0}"
RUN_MSDWILD="${RUN_MSDWILD:-1}"
RUN_FILLERS="${RUN_FILLERS:-1}"
ONLY_LANE="${ONLY_LANE:-}"

RAMC_INFER_DEVICE="${RAMC_INFER_DEVICE:-cpu}"
RAMC_CPU_THREADS="${RAMC_CPU_THREADS:-8}"
RAMC_MAX_INPUT_FRAMES="${RAMC_MAX_INPUT_FRAMES:--1}"
RAMC_LOCK_TIMEOUT="${RAMC_LOCK_TIMEOUT:-21600}"    # 6 h; jobs take ~50 min

DIAPER_ENV="${DIAPER_ENV:-/data/ocr/namvt17/custom-diaper/.venv}"
DSCORE_SRC="${DSCORE_SRC:-/data/ocr/namvt17/custom-diaper/dscore}"
DSCORE_ENV="${DSCORE_ENV:-/data/ocr/namvt17/custom-diaper/dscore/.dscore}"
MAX_CHECKPOINTS_TO_AVERAGE="${MAX_CHECKPOINTS_TO_AVERAGE:-10}"

EBF_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_ebf
CNF_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_conformer_k31

# Existing (old-recipe) finetuned lineages, used only by diagnostic D1.
EBF_OLD_INFER_RAMC=models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer/infer_ramc.yaml
CNF_OLD_INFER_RAMC=models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff/infer_ramc_mlp.yaml

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

yaml_get () {  # yaml_get <key> <file>
    grep "^$1:" "$2" | head -1 | sed "s|^$1: *||"
}

# ---------------------------------------------------------------------------
# Cross-lane mutex for RAMC inference. mkdir is atomic on every filesystem
# that matters here, so no flock/util-linux dependency. A lock whose owner
# PID is gone is treated as stale and reclaimed -- otherwise a lane killed
# mid-inference would block the other lane for the rest of the run.
# ---------------------------------------------------------------------------
ramc_lock_acquire () {
    local waited=0
    while true; do
        if mkdir "$RAMC_LOCK" 2>/dev/null; then
            echo $$ > "$RAMC_LOCK/pid"
            return 0
        fi
        local owner
        owner=$(cat "$RAMC_LOCK/pid" 2>/dev/null)
        if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then
            log "RAMC lock held by dead pid $owner -- reclaiming"
            rm -rf "$RAMC_LOCK"
            continue
        fi
        sleep 60
        waited=$((waited + 60))
        if [ "$waited" -ge "$RAMC_LOCK_TIMEOUT" ]; then
            log "RAMC lock still held after ${waited}s -- SKIPPING this scoring"
            return 1
        fi
    done
}

ramc_lock_release () { rm -rf "$RAMC_LOCK"; }

# ---------------------------------------------------------------------------
# preflight -- verify every path a lane will read, BEFORE burning GPU hours.
#
# Audited against the local Windows checkout on 2026-08-28: all five required
# checkpoint directories exist, cover the epochs their configs ask for, and
# torch.load cleanly. What is NOT on the local machine, and therefore cannot
# be restored from it if the server has lost it, is the 2500h SC data:
#   diaper_precompute_2500h_maximum_10spks_24000frames  (adapt -- REQUIRED)
#   diaper_precompute_2500h_fixed_2spks                 (only if RUN_PRETRAIN=1)
# E:/datasets holds only the 300h/500h equivalents. If the server has lost
# the 2500h caches they must come from S3 (s3-b200:ttnt-data/ocr/namvt17/...,
# see scripts/extract_tar_subset_from_s3.py) -- there is no local copy.
#
# The adapt chunk count is checked, not just the directory's existence:
# noam_warmup_steps: 66749 is derived from 19,888 chunks/epoch at batch 16.
# A partial cache would silently produce a different realized schedule, which
# is the exact class of bug this whole lineage exists to fix.
# ---------------------------------------------------------------------------
ADAPT_EXPECTED_CHUNKS="${ADAPT_EXPECTED_CHUNKS:-19888}"

preflight () {
    local cfgdir="$1" label="$2" fail=0 p n
    log "PREFLIGHT $label"

    for spec in \
        "adapt-init:$(yaml_get init_model_path "$cfgdir/train_10spks.yaml")" \
        "adapt-train:$(yaml_get train_precomputed_dir "$cfgdir/train_10spks.yaml")" \
        "adapt-valid:$(yaml_get valid_precomputed_dir "$cfgdir/train_10spks.yaml")" \
        "ramc-train:$(yaml_get train_precomputed_dir "$cfgdir/finetune_ramc_10spks.yaml")" \
        "ramc-dev:$(yaml_get valid_precomputed_dir "$cfgdir/finetune_ramc_10spks.yaml")" \
        "msd-train:$(yaml_get train_precomputed_dir "$cfgdir/finetune_msdwild_10spks.yaml")" \
        "msd-dev:$(yaml_get valid_precomputed_dir "$cfgdir/finetune_msdwild_10spks.yaml")" \
        "ramc-infer:$(yaml_get infer_data_dir "$cfgdir/infer_ramc.yaml")" \
        "msd-infer:$(yaml_get infer_data_dir "$cfgdir/infer_msdwild.yaml")" ; do
        p="${spec#*:}"
        if [ -d "$p" ]; then
            log "  ok      ${spec%%:*}"
        else
            log "  MISSING ${spec%%:*} -> $p"
            fail=1
        fi
    done

    # the warm-start must actually contain the epochs init_epochs asks for
    p=$(yaml_get init_model_path "$cfgdir/train_10spks.yaml")
    if [ -d "$p" ]; then
        n=$(find "$p" -maxdepth 1 -name 'checkpoint_*.tar' | wc -l)
        log "  warm-start holds $n checkpoint(s); init_epochs=$(yaml_get init_epochs "$cfgdir/train_10spks.yaml")"
        [ "$n" -lt 10 ] && { log "  WARNING: fewer than 10 checkpoints to average"; }
    fi

    # a partial adapt cache would silently change the realized Noam schedule
    p=$(yaml_get train_precomputed_dir "$cfgdir/train_10spks.yaml")
    if [ -d "$p" ]; then
        n=$(find "$p" -maxdepth 1 -type f | wc -l)
        log "  adapt cache: $n chunks (expected ~$ADAPT_EXPECTED_CHUNKS)"
        if [ "$n" -lt $(( ADAPT_EXPECTED_CHUNKS * 98 / 100 )) ] || \
           [ "$n" -gt $(( ADAPT_EXPECTED_CHUNKS * 102 / 100 )) ]; then
            log "  *** adapt cache size is off by >2%. noam_warmup_steps: 66749"
            log "  *** assumes $ADAPT_EXPECTED_CHUNKS chunks / 1,243 steps per epoch at batch 16."
            log "  *** Recompute with diaper/common_utils/noam_lr_calc.py before running,"
            log "  *** or the schedule will not be the one this lineage documents."
            fail=1
        fi
    fi

    if [ "$fail" -ne 0 ]; then
        # ADVISORY BY DEFAULT. This runs unattended for five days; a false
        # positive here (a symlinked data dir, a legitimately different chunk
        # count) must never cost the whole run. Everything above is logged so
        # the driver log says what was wrong, and the lane starts anyway --
        # a genuinely missing path just makes its own stage fail, and the
        # lane moves on to the next item as it would for any other failure.
        # Set STRICT_PREFLIGHT=1 to abort the lane instead.
        log "PREFLIGHT: $label has problems above."
        if [ "${STRICT_PREFLIGHT:-0}" = "1" ]; then
            log "PREFLIGHT FAILED for $label (STRICT_PREFLIGHT=1) -- lane not started."
            return 1
        fi
        log "PREFLIGHT: continuing anyway (STRICT_PREFLIGHT=1 would stop here)."
        return 0
    fi
    log "PREFLIGHT OK for $label"
    return 0
}

# ---------------------------------------------------------------------------
# train_stage <label> <gpu> <config> <logfile> [extra train.py args...]
# ---------------------------------------------------------------------------
train_stage () {
    local label="$1" gpu="$2" cfg="$3" logfile="$4"; shift 4
    local attempt rc
    if [ ! -f "$cfg" ]; then
        log "SKIP $label -- config not found: $cfg"
        return 1
    fi
    for attempt in 1 2; do
        log "START $label (attempt $attempt/2) gpu=$gpu cfg=$cfg"
        CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/train.py -c "$cfg" \
            --gpu 1 "$@" >> "$logfile" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            log "DONE  $label"
            return 0
        fi
        log "FAIL  $label (exit $rc) -- see $logfile"
        # A second attempt is nearly free: train.py resumes from the latest
        # checkpoint, so a transient OOM/filesystem blip costs minutes. A
        # config or data error fails identically twice and the lane moves on.
    done
    return 1
}

# ---------------------------------------------------------------------------
# infer_stage <label> <gpu-or-cpu> <config> <models_path> <rttms_dir>
#             <collar> <median> <score_suffix> <logfile> [extra args...]
#
# device: a physical gpu id, or the literal "cpu".
# collar: "" means no --collar flag (0 s, RAMC). "0.25" for MSDWild.
# Mirrors scripts/run_infer_*.sh: auto-detects checkpoints on disk and
# averages the last $MAX_CHECKPOINTS_TO_AVERAGE, then scores with dscore.
# ---------------------------------------------------------------------------
infer_stage () {
    local label="$1" device="$2" cfg="$3" models_path="$4" rttms_dir="$5"
    local collar="$6" median="$7" suffix="$8" logfile="$9"; shift 9

    if [ ! -f "$cfg" ]; then
        log "SKIP infer $label -- config not found: $cfg"
        return 1
    fi
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

    local dev_args=() visible=""
    if [ "$device" = "cpu" ]; then
        dev_args=(--gpu 0 --num-threads "$RAMC_CPU_THREADS")
        visible=""
    else
        dev_args=(--gpu 1 --num-threads 4)
        visible="$device"
    fi
    if [ "$RAMC_MAX_INPUT_FRAMES" != "-1" ] && [ "$device" = "cpu" ]; then
        dev_args+=(--max-input-frames "$RAMC_MAX_INPUT_FRAMES")
    fi

    log "START infer $label device=$device epochs=$epochs_range median=$median"
    CUDA_VISIBLE_DEVICES="$visible" "${PY[@]}" diaper/infer.py -c "$cfg" \
        --models-path "$models_path" \
        --rttms-dir "$rttms_dir" \
        --median-window-length "$median" \
        --epochs "$epochs_range" \
        "${dev_args[@]}" "$@" >> "$logfile" 2>&1
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

    log "SCORED $label (${#sys_rttms[@]} files) -> $score_log"
    grep -h "OVERALL" "$score_log" || log "WARN no OVERALL line in $score_log"
    return 0
}

# ---------------------------------------------------------------------------
# ramc_infer_bg -- same signature as infer_stage minus the device, but takes
# the cross-lane RAMC lock and runs detached so the lane's GPU keeps working.
# Appends the child PID to the caller's RAMC_JOBS array via a namefile.
# ---------------------------------------------------------------------------
ramc_infer_bg () {
    local label="$1" cfg="$2" models_path="$3" rttms_dir="$4"
    local median="$5" suffix="$6" logfile="$7" jobfile="$8"; shift 8
    (
        if ! ramc_lock_acquire; then exit 1; fi
        trap 'ramc_lock_release' EXIT
        infer_stage "$label" "$RAMC_INFER_DEVICE" "$cfg" "$models_path" \
            "$rttms_dir" "" "$median" "$suffix" "$logfile" "$@"
    ) &
    echo $! >> "$jobfile"
    log "QUEUED RAMC scoring: $label (pid $!, device=$RAMC_INFER_DEVICE)"
}

# ---------------------------------------------------------------------------
# lane <letter> <config dir> <gpu> <old-lineage RAMC infer cfg for D1>
# ---------------------------------------------------------------------------
lane () {
    local L="$1" CFG="$2" GPU="$3" D1_CFG="$4"
    local P="${LOG_DIR}/lane${L}"
    mkdir -p "$P"
    local JOBS="$P/.ramc_jobs"
    : > "$JOBS"

    local adapt_out ft_ramc_out ft_msd_out
    adapt_out=$(yaml_get output_path "$CFG/train_10spks.yaml")
    ft_ramc_out=$(yaml_get output_path "$CFG/finetune_ramc_10spks.yaml")
    ft_msd_out=$(yaml_get output_path "$CFG/finetune_msdwild_10spks.yaml")

    log "LANE $L on gpu $GPU -- adapt output: $adapt_out"

    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        if ! preflight "$CFG" "lane $L"; then
            log "LANE $L ABORTED before any GPU work. Fix the paths above and re-run."
            return 1
        fi
    fi
    # Note: preflight only returns non-zero under STRICT_PREFLIGHT=1. By
    # default it reports and the lane proceeds -- see its header.

    # -- D1 diagnostic, in the background from the start --------------------
    # Re-score the EXISTING old-recipe checkpoint at MSDWild's resolution
    # settings (subsampling 10 / median 11) instead of RAMC's (5 / 1).
    # The conv branches carry a fixed-size depthwise kernel, so their
    # receptive field in SECONDS halves at subsampling 5, while plain
    # self-attention (use_posenc: False) has no fixed temporal kernel.
    # results.csv's third column already shows every conv variant losing
    # 6-10 DER when MSDWild is re-scored at subsampling 5 against plain
    # self-attention's 5.26 -- this asks the same question on RAMC, where
    # subsampling 5 is the standard protocol.
    # If the conv variants gain >= 2 DER here and paperlr gains < 1, the
    # eval protocol has been taxing them all along. Report it as an EXTRA
    # results.csv column, never a replacement: it departs from the paper's
    # prescribed collar-0 protocol (Section IV.D).
    if [ "$RUN_DIAGNOSTICS" = "1" ] && [ -f "$D1_CFG" ]; then
        ramc_infer_bg "D1-lane${L} subsampling10" "$D1_CFG" \
            "$(yaml_get models_path "$D1_CFG")" \
            "$(yaml_get rttms_dir "$D1_CFG")" \
            11 "_subsampling10" "$P/0_d1_subsampling10.log" "$JOBS" \
            --subsampling 10
    fi

    # -- optional fresh pretrain --------------------------------------------
    local adapt_init_args=()
    if [ "$RUN_PRETRAIN" = "1" ]; then
        train_stage "L${L} pretrain (2 spk, 2500h, batch 96)" "$GPU" \
            "$CFG/train.yaml" "$P/1_pretrain.log"
        adapt_init_args=(--init-model-path "$(yaml_get output_path "$CFG/train.yaml")/models")
    fi

    # -- 1. adapt: restored perceiver loss + corrected Noam ------------------
    train_stage "L${L} adapt (interm. perceiver loss ON, noam 1534/66749)" \
        "$GPU" "$CFG/train_10spks.yaml" "$P/2_adapt.log" "${adapt_init_args[@]}"

    # -- 2. finetune RAMC  << THE DECIDING NUMBER >> ------------------------
    train_stage "L${L} finetune RAMC" "$GPU" \
        "$CFG/finetune_ramc_10spks.yaml" "$P/3_ft_ramc.log"
    ramc_infer_bg "L${L} RAMC" "$CFG/infer_ramc.yaml" \
        "${ft_ramc_out}/models" "${ft_ramc_out}/ramc_test_pred" \
        1 "" "$P/3_ft_ramc_infer.log" "$JOBS"

    # -- 3. finetune MSDWild (fits one GPU, scored inline) ------------------
    if [ "$RUN_MSDWILD" = "1" ]; then
        train_stage "L${L} finetune MSDWild" "$GPU" \
            "$CFG/finetune_msdwild_10spks.yaml" "$P/4_ft_msdwild.log"
        infer_stage "L${L} MSDWild" "$GPU" "$CFG/infer_msdwild.yaml" \
            "${ft_msd_out}/models" "${ft_msd_out}/msdwild_test_pred" \
            "0.25" 11 "" "$P/4_ft_msdwild_infer.log"
    fi

    # -- 4. F1: RAMC finetune at the resolution it is EVALUATED at ----------
    # RAMC trains at subsampling 10 and infers at 5. This closes half of
    # that mismatch (num_frames 1200 holds the 60 s sequence length; batch
    # drops to 16 for the doubled token count). It does NOT close the other
    # half: RAMC recordings are ~31 min at inference against 60 s training
    # chunks, a ~37x length extrapolation no finetune setting fixes.
    if [ "$RUN_FILLERS" = "1" ]; then
        train_stage "L${L} F1 finetune RAMC @ subsampling 5" "$GPU" \
            "$CFG/finetune_ramc_10spks.yaml" "$P/5_f1_ft_ramc_sub5.log" \
            --subsampling 5 --num-frames 1200 --train-batchsize 16 \
            --output-path "${ft_ramc_out}_sub5"
        ramc_infer_bg "L${L} F1 RAMC sub5" "$CFG/infer_ramc.yaml" \
            "${ft_ramc_out}_sub5/models" "${ft_ramc_out}_sub5/ramc_test_pred" \
            1 "" "$P/5_f1_ft_ramc_sub5_infer.log" "$JOBS"

        # -- 5. F2: the finetune LR nobody has ever swept -------------------
        # Every finetune in this repo uses Adam at a flat 1e-6, chosen for a
        # model with 73% fewer parameters than these two.
        train_stage "L${L} F2 finetune RAMC @ lr 3e-6" "$GPU" \
            "$CFG/finetune_ramc_10spks.yaml" "$P/6_f2_ft_ramc_lr3e-6.log" \
            --lr 3e-6 --output-path "${ft_ramc_out}_lr3e-6"
        ramc_infer_bg "L${L} F2 RAMC lr3e-6" "$CFG/infer_ramc.yaml" \
            "${ft_ramc_out}_lr3e-6/models" "${ft_ramc_out}_lr3e-6/ramc_test_pred" \
            1 "" "$P/6_f2_ft_ramc_lr3e-6_infer.log" "$JOBS"

        # Second LR point. Last in the queue on purpose -- if the box is
        # slower than the calibration gate assumed, this is the item that
        # should fall off the end.
        train_stage "L${L} F2 finetune RAMC @ lr 1e-5" "$GPU" \
            "$CFG/finetune_ramc_10spks.yaml" "$P/7_f2_ft_ramc_lr1e-5.log" \
            --lr 1e-5 --output-path "${ft_ramc_out}_lr1e-5"
        ramc_infer_bg "L${L} F2 RAMC lr1e-5" "$CFG/infer_ramc.yaml" \
            "${ft_ramc_out}_lr1e-5/models" "${ft_ramc_out}_lr1e-5/ramc_test_pred" \
            1 "" "$P/7_f2_ft_ramc_lr1e-5_infer.log" "$JOBS"
    fi

    # -- wait for this lane's background RAMC scorings ----------------------
    local jp
    while read -r jp; do
        [ -n "$jp" ] && wait "$jp" 2>/dev/null
    done < "$JOBS"

    log "LANE $L COMPLETE"
}

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Honour an inherited CUDA_VISIBLE_DEVICES.
#
# Every stage runs as `CUDA_VISIBLE_DEVICES="$gpu" python ...`, and a
# per-command assignment overrides an exported one -- so without this block,
#   CUDA_VISIBLE_DEVICES=2,3 ./scripts/run_5day_queue.sh
# would silently discard the 2,3 and run both lanes on physical 0 and 1,
# very possibly on top of somebody else's job. Rather than ignore it, take
# the list as the lane device assignment, which is what anyone typing it
# means (and matches the GPUS= convention of the other pipeline scripts).
# An explicit LANE_A_GPU / LANE_B_GPU always wins over it.
# ---------------------------------------------------------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _cvd <<< "$CUDA_VISIBLE_DEVICES"
    _from_cvd=""
    if [ -z "$_LANE_A_EXPLICIT" ] && [ -n "${_cvd[0]:-}" ]; then
        LANE_A_GPU="${_cvd[0]}"; _from_cvd="yes"
    fi
    if [ -z "$_LANE_B_EXPLICIT" ]; then
        # With only one device listed, lane B takes that same device rather
        # than keeping its default of 1 -- otherwise it would quietly train
        # on a GPU the caller never listed, which is the exact failure this
        # block exists to prevent.
        LANE_B_GPU="${_cvd[1]:-${_cvd[0]}}"; _from_cvd="yes"
    fi
    if [ -n "$_from_cvd" ]; then
        log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> lane A gpu $LANE_A_GPU, lane B gpu $LANE_B_GPU"
    fi
    if [ "${#_cvd[@]}" -lt 2 ] && [ -z "$ONLY_LANE" ]; then
        log "WARNING: CUDA_VISIBLE_DEVICES lists only ${#_cvd[@]} device, but both"
        log "WARNING: lanes will run and now SHARE gpu $LANE_A_GPU. At batch 16 over"
        log "WARNING: 2400 frames they will contend and probably both OOM."
        log "WARNING: Fix: pass ONLY_LANE=A (or B), or list two devices."
    fi
    # Consumed. Leaving it exported is harmless (each stage overrides it),
    # but unsetting keeps `nvidia-smi`-style debugging from being confusing.
    unset CUDA_VISIBLE_DEVICES
fi

log "5-day queue starting"
log "  lane A (E-Branchformer) gpu=$LANE_A_GPU  cfg=$EBF_DIR"
log "  lane B (conformer k31)  gpu=$LANE_B_GPU  cfg=$CNF_DIR"
log "  diagnostics=$RUN_DIAGNOSTICS pretrain=$RUN_PRETRAIN msdwild=$RUN_MSDWILD fillers=$RUN_FILLERS"
log "  RAMC inference device=$RAMC_INFER_DEVICE (serialized across lanes)"
log "  logs: $LOG_DIR"

rm -rf "$RAMC_LOCK"

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
log "(their own selves with the perceiver loss off and the old schedule),"
log "and 20.80 (paperlr vanilla). Before recording any win, run the paired"
log "bootstrap -- RAMC resolves 2-3 DER, MSDWild pooled resolves nothing"
log "under ~1.5, and MSDWild must be read bucketed by speaker count."
