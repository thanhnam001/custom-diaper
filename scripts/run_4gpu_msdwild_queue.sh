#!/bin/bash
# 4-GPU MSDWild-only queue (2026-09-11). One lane per GPU, all independent --
# no RAMC anywhere, so none of run_36h_resume_queue.sh's RAMC CPU/RAM locking
# applies and every lane can run flat out.
#
#   mkdir -p logs/4gpu_msdwild_queue
#   nohup ./scripts/run_4gpu_msdwild_queue.sh > logs/4gpu_msdwild_queue/driver.log 2>&1 &
#
# Upload the weights first with scripts/pack_weights_for_4gpu_msdwild_queue.sh
# (~594 MB). The preflight below refuses to start a lane whose init/resume
# checkpoints are missing rather than silently training from random init.
#
#
# WHY THESE FOUR LANES
# =====================
# Established before building this queue (2026-09-11), on MSDWild test
# (collar 0.25 / median 11 / subsampling 10, 490 files, dscore pooled):
#
#   PAPER's own checkpoint          15.47
#   A1 conformer_k31  ep646-656     17.07   <- our best
#   ebf paperlr       ep740-750     17.15   <- statistically tied with A1
#   paperlr vanilla   ep551-561     18.31   <- our closest paper reproduction
#
# The residual gap to the paper is +1.77 (paired bootstrap 95% CI
# [-2.48,-1.09], so it is real and large). Two things were measured that
# determine what is worth running:
#
# 1. THE GAP IS BROAD, NOT A FIXABLE SUBSET. Median per-file delta +0.98,
#    we lose on 62% of files, and the RELATIVE gap is flat at +8.6%..+16.3%
#    across every stratum (speaker count, overlap fraction, duration, and
#    the 420 files where neither model breaks). No file property predicts
#    the per-file delta (|r| <= 0.12). The 420 "both fine" files carry 86%
#    of the gap; catastrophic divergence nets to only +0.16 (~9%). So there
#    is nothing to triage -- a fix has to lift the model broadly.
#
# 2. OUR FINETUNE CONFIG IS BYTE-IDENTICAL TO THE PAPER'S. Diffed
#    models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr/finetune_msdwild_10spks.yaml
#    against the paper's shipped
#    ../Master/repos/DiaPer/models/10attractors/SC_LibriSpeech_2spk_adapted1-10_finetuneMSDWILD/train.yaml:
#    identical on lr 1e-6, adam, train_batchsize 32, num_frames 600,
#    gradclip 5, norm_loss_per_spk True, max_epochs 750, init_epochs 90-100,
#    seed 3, dropout 0.1/0.1, both intermediate losses, self_attention +
#    weighted_average. The only diffs anywhere are our deliberate lineage
#    changes (conformer/ebranchformer encoder, mlp latents2attractors,
#    diversity loss) -- and those HELP (18.31 -> 17.06).
#
# So the deficit is not a hyperparameter. That leaves a clean 2x2 with one
# cell missing, which lane A fills:
#
#                  |  our finetune  |  paper's finetune
#   our adapt      |     18.31      |  18.31 (config-identical)
#   PAPER's adapt  |   LANE A  <--  |     15.47 (published)
#
#
# LANE LAYOUT
# ============
#   A (gpu 0)  THE SWAP -- the decisive one. Fresh MSDWild finetune from the
#              PAPER's adapt checkpoint through our exact pipeline.
#              ~15.5 => our finetune stage is faithful and 100% of the
#                       remaining gap is the adapt/SC stage. Stop tuning
#                       finetuning; spend everything on adapt.
#              ~18   => our finetune EXECUTION is defective despite an
#                       identical config. Suspects a config diff cannot see:
#                       the precompute cache, checkpoint averaging, or early
#                       stopping being driven by MSDWild dev (many.val, 97%
#                       5-10 speakers) while train/test are both 2-4 speakers.
#              Diagnostic, NOT a record attempt -- it runs the paper's
#              architecture, so it will not beat 17.07 either way.
#
#   B (gpu 1)  Extend ebf paperlr MSDWild past its 750 cap. Proven lever,
#              expect ~-0.3, zero risk. CAP-bound (ended exactly at 750).
#
#   C (gpu 2)  Extend A1 conformer_k31 MSDWild past 656. NOTE: this one is
#              PATIENCE-bound, not cap-bound -- it stopped at 656 with 94
#              epochs still under its own max_epochs 750, so raising
#              --max-epochs alone is a no-op; patience must go up too.
#
#   D (gpu 3)  SEED REPLICATE of A1: same config, same adapt init, only the
#              seed differs. Calibrates the whole results table. Two of our
#              models disagree by mean |per-file DER| 6.5-7.6, and this has
#              never been split into architecture vs seed. If seed alone is
#              worth ~1 DER then the 17.07-vs-17.15 "tie" and much of
#              results.csv's architecture ordering is noise.
#
# Deliberately NOT here: overlap loss (the stratification in point 1 killed
# its motivation -- the LOWEST-overlap bucket has the LARGEST relative gap,
# +16.3%), attractor-existence pos_weight (run 2026-09-11, confirmed dead,
# flat dose-response between 3.0 and 5.0), and RAMC (parked by the user).
#
#
# ENV KNOBS
#   LANE_{A,B,C,D}_GPU   physical device ids        (default 0/1/2/3)
#   ONLY_LANE=A|B|C|D    run a single lane (repeatable, comma-separated)
#   SEED_D               lane D's seed              (default 7)
#   LR_ARM               lane C stage-2 finetune LR (default 1e-5, vs the
#                        config/paper's 1e-6; use 3e-6 if 1e-5 destabilises)
#   EXTEND_MAX_EPOCHS    lanes B/C new cap          (default 900)
#   SCORE_ONLY=1         skip training, score what is on disk now
#   SKIP_PREFLIGHT=1     don't verify checkpoints first
#   LOG_DIR              default logs/4gpu_msdwild_queue

set -u

LANE_A_GPU="${LANE_A_GPU:-0}"
LANE_B_GPU="${LANE_B_GPU:-1}"
LANE_C_GPU="${LANE_C_GPU:-2}"
LANE_D_GPU="${LANE_D_GPU:-3}"
ONLY_LANE="${ONLY_LANE:-}"
SEED_D="${SEED_D:-7}"
LR_ARM="${LR_ARM:-1e-5}"
EXTEND_MAX_EPOCHS="${EXTEND_MAX_EPOCHS:-900}"
SCORE_ONLY="${SCORE_ONLY:-0}"
LOG_DIR="${LOG_DIR:-logs/4gpu_msdwild_queue}"

DIAPER_ENV="${DIAPER_ENV:-/data/ocr/namvt17/custom-diaper/.venv}"
DSCORE_SRC="${DSCORE_SRC:-/data/ocr/namvt17/custom-diaper/dscore}"
DSCORE_ENV="${DSCORE_ENV:-/data/ocr/namvt17/custom-diaper/dscore/.dscore}"
EXP_ROOT="${EXP_ROOT:-/data/ocr/namvt17/custom-diaper/experiments/10attractors}"
MAX_CHECKPOINTS_TO_AVERAGE="${MAX_CHECKPOINTS_TO_AVERAGE:-10}"

PLR_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr
CNF_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_conformer_k31
EBF_DIR=models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer

# Where scripts/pack_weights_for_4gpu_msdwild_queue.sh puts the paper's own
# adapt checkpoints (epochs 91-100). Kept in a PAPER_-prefixed directory so
# it can never be confused with one of our own lineages.
PAPER_ADAPT="${PAPER_ADAPT:-$EXP_ROOT/PAPER_SC_LibriSpeech_2spk_adapted1-10/models}"

mkdir -p "$LOG_DIR"
STOP_FILE="${LOG_DIR}/STOP"

if [ "${USE_CONDA_RUN:-1}" = "1" ]; then
    PY=(conda run -p "$DIAPER_ENV" --no-capture-output python)
    DSCORE_PY=(conda run -p "$DSCORE_ENV" --no-capture-output python -u)
else
    PY=(python)
    DSCORE_PY=(python -u)
fi

log () { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
yaml_get () { grep "^$1:" "$2" | head -1 | sed "s|^$1: *||"; }
stop_requested () { [ -f "$STOP_FILE" ]; }
lane_enabled () { [ -z "$ONLY_LANE" ] && return 0; case ",$ONLY_LANE," in *",$1,"*) return 0;; esac; return 1; }

# Same escalation as run_36h_resume_queue.sh: a background lane subshell
# ignores a bare SIGINT, so tear down the whole process group.
on_signal () {
    trap - INT TERM
    log "caught SIG$1 -- stopping the queue"
    : > "$STOP_FILE"
    local mypgid
    mypgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
    if [ -n "$mypgid" ] && [ "$mypgid" = "$$" ]; then
        kill -TERM 0 2>/dev/null; sleep 5; kill -KILL 0 2>/dev/null
    else
        pkill -TERM -P $$ 2>/dev/null
    fi
    exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

# ---------------------------------------------------------------------------
# guard_output <output_path> <config> <label>
# Refuse to write into a config's OWN output_path. train.py auto-resumes from
# the latest checkpoint in output_path BEFORE it considers init_model_path, so
# a fresh-run lane pointed at an existing run's directory would silently warm
# start from that run AND overwrite its checkpoints. That is how you destroy a
# baseline; run_posweight_queue.sh added this check for the same reason.
# ---------------------------------------------------------------------------
guard_output () {
    local out="$1" cfg="$2" label="$3" cfg_out
    cfg_out="$(yaml_get output_path "$cfg")"
    if [ "$out" = "$cfg_out" ]; then
        log "FATAL $label -- output_path equals the config's own run ($cfg_out)."
        log "      That would overwrite an existing baseline. Refusing."
        return 1
    fi
    return 0
}

preflight_init () {   # fresh-finetune lanes: the warm-start weights must exist
    local dir="$1" label="$2" n
    n=$(find "$dir" -maxdepth 1 -name 'checkpoint_*.tar' 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        log "FATAL $label -- no init checkpoints at $dir"
        log "      Upload them with scripts/pack_weights_for_4gpu_msdwild_queue.sh,"
        log "      or this trains from random init and answers a different question."
        return 1
    fi
    log "  ok $label -- init has $n checkpoint(s) at $dir"
    return 0
}

preflight_resume () { # resume lanes: there must be something to resume FROM
    local out="$1" label="$2" n
    n=$(find "$out/models" -maxdepth 1 -name 'checkpoint_*.tar' 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        log "FATAL $label -- $out/models has no checkpoint to resume from."
        log "      Without it this stage trains FROM SCRATCH instead of extending."
        return 1
    fi
    local last
    last=$(find "$out/models" -maxdepth 1 -name 'checkpoint_*.tar' -exec basename {} \; \
           | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n | tail -1)
    log "  ok $label -- $n checkpoint(s), latest epoch $last, will resume"
    return 0
}

# ---------------------------------------------------------------------------
# train_fresh <label> <gpu> <cfg> <out> <log> [extra args...]
# Warm-starts from init_model_path (overridden per lane). Does NOT pass
# --init-model-path '' -- a fresh finetune is SUPPOSED to average the adapt
# checkpoints. train.py checkpoints every epoch and auto-resumes from
# output_path, so re-running after an interruption is safe and cheap.
# ---------------------------------------------------------------------------
train_fresh () {
    local label="$1" gpu="$2" cfg="$3" out="$4" logfile="$5"; shift 5
    if stop_requested; then log "STOP requested -- not starting $label"; return 1; fi
    log "START $label gpu=$gpu cfg=$cfg out=$out"
    env CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/train.py -c "$cfg" \
        --gpu 1 --output-path "$out" "$@" >> "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then log "DONE  $label"; return 0; fi
    log "FAIL  $label (exit $rc) -- see $logfile"
    stop_requested && return 1
    log "RETRY $label once (train.py resumes from its own checkpoints)"
    env CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/train.py -c "$cfg" \
        --gpu 1 --output-path "$out" "$@" >> "$logfile" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { log "DONE  $label (after retry)"; return 0; }
    log "FAIL  $label (exit $rc, after retry)"; return 1
}

# ---------------------------------------------------------------------------
# train_resume <label> <gpu> <cfg> <out> <log> [extra args...]
# THE init_model_path TRAP (learned the hard way, see memory
# diaper-36h-resume-queue): train.py runs
#   if args.init_model_path != '': average_checkpoints(...)
# UNCONDITIONALLY, before it ever checks output_path for a resumable
# checkpoint -- so even a successful resume first tries to load the config's
# init_model_path (the ADAPT stage) and only overwrites that model moments
# later. If the adapt directory is absent server-side the stage crashes on
# startup even though the resume never needed it. So when a real resume
# checkpoint exists we pass --init-model-path '' (empty satisfies the != ''
# gate, skipping average_checkpoints entirely -- harmless, its result would
# be discarded by the resume anyway).
# ---------------------------------------------------------------------------
train_resume () {
    local label="$1" gpu="$2" cfg="$3" out="$4" logfile="$5"; shift 5
    if stop_requested; then log "STOP requested -- not starting $label"; return 1; fi
    local init_args=()
    if [ "$(find "$out/models" -maxdepth 1 -name 'checkpoint_*.tar' 2>/dev/null | wc -l)" -gt 0 ]; then
        init_args=(--init-model-path '')
    else
        log "WARN  $label -- no resume checkpoint; NOT overriding init-model-path"
    fi
    log "START $label gpu=$gpu (resume, cap $EXTEND_MAX_EPOCHS)"
    env CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/train.py -c "$cfg" \
        --gpu 1 --output-path "$out" "${init_args[@]}" "$@" >> "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then log "DONE  $label"; return 0; fi
    log "FAIL  $label (exit $rc) -- see $logfile"
    stop_requested && return 1
    log "RETRY $label once"
    env CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/train.py -c "$cfg" \
        --gpu 1 --output-path "$out" "${init_args[@]}" "$@" >> "$logfile" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { log "DONE  $label (after retry)"; return 0; }
    log "FAIL  $label (exit $rc, after retry)"; return 1
}

# ---------------------------------------------------------------------------
# score <label> <gpu> <infer_cfg> <out> <range|auto> <log>
# MSDWild inference is GPU-and-inline (490 files, ~2 min) -- none of RAMC's
# locking applies. 'auto' averages the last $MAX_CHECKPOINTS_TO_AVERAGE
# checkpoints; an explicit "A-B" scores exactly that range.
#
# The dscore log filename EMBEDS THE EPOCH RANGE on purpose: the 36h queue
# wrote a fixed dscore_*_resumed.log every call, so after a later round added
# more rttms nobody could tell which epochs a score belonged to and all 9
# stages had to be rescored by hand. Do not "simplify" this back.
# ---------------------------------------------------------------------------
score () {
    local label="$1" gpu="$2" cfg="$3" out="$4" want="$5" logfile="$6"
    local models_path="$out/models" rttms_dir="$out/msdwild_test_pred"

    if [ ! -f "$cfg" ]; then log "SKIP score $label -- no config $cfg"; return 1; fi
    if [ ! -d "$models_path" ]; then log "SKIP score $label -- no models dir $models_path"; return 1; fi
    if stop_requested; then log "STOP requested -- not scoring $label"; return 1; fi

    local range
    if [ "$want" = "auto" ]; then
        mapfile -t ck < <(find "$models_path" -maxdepth 1 -name 'checkpoint_*.tar' \
            -exec basename {} \; | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n)
        if [ "${#ck[@]}" -eq 0 ]; then log "SKIP score $label -- no checkpoints"; return 1; fi
        local li=$(( ${#ck[@]} - 1 )) si
        si=$(( ${#ck[@]} > MAX_CHECKPOINTS_TO_AVERAGE \
               ? ${#ck[@]} - MAX_CHECKPOINTS_TO_AVERAGE : 0 ))
        range="$(( ${ck[$si]} - 1 ))-${ck[$li]}"
    else
        range="$want"
        # an explicit range is useless if those checkpoints were never kept
        local lo=${range%-*} hi=${range#*-}
        if [ ! -f "$models_path/checkpoint_${hi}.tar" ]; then
            log "SKIP score $label -- checkpoint_${hi}.tar absent (range $range not on disk)"
            return 1
        fi
    fi

    local infer_data_dir ref_rttm
    infer_data_dir=$(yaml_get infer_data_dir "$cfg")
    ref_rttm="${infer_data_dir}/rttm"

    log "START score $label epochs=$range"
    env CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" diaper/infer.py -c "$cfg" \
        --models-path "$models_path" --rttms-dir "$rttms_dir" \
        --epochs "$range" --median-window-length 11 --subsampling 10 \
        --gpu 1 --num-threads 4 >> "$logfile" 2>&1
    if [ $? -ne 0 ]; then log "FAIL  score $label -- see $logfile"; return 1; fi

    mapfile -t sys_rttms < <(find "$rttms_dir/epochs${range}" \
        -path "*/median11/*/rttms/*.rttm" -type f 2>/dev/null)
    if [ "${#sys_rttms[@]}" -eq 0 ]; then
        log "WARN  no RTTMs under $rttms_dir/epochs${range} -- not scored"; return 1
    fi
    local score_log="${rttms_dir}/dscore_collar0.25_epochs${range}.log"
    "${DSCORE_PY[@]}" "$DSCORE_SRC/score.py" --collar 0.25 \
        -r "$ref_rttm" -s "${sys_rttms[@]}" > "$score_log" 2>&1
    log "SCORED $label (${#sys_rttms[@]} files) -> $score_log"
    grep -h "OVERALL" "$score_log" || log "WARN no OVERALL line in $score_log"
    return 0
}

# ===========================================================================
# LANE A -- THE SWAP (gpu $LANE_A_GPU)
# Paper's adapt checkpoint -> our finetune pipeline, paperlr config (which is
# config-identical to the paper's finetune yaml, and is self_attention +
# weighted_average so the paper's weights actually load).
#
# Scored at TWO ranges on purpose:
#   515-525  matches the paper's own published FT checkpoint budget, so it is
#            the apples-to-apples cell of the 2x2 against its 15.47
#   auto     wherever this run actually ends, for the best-effort number
# ===========================================================================
lane_A () {
    local GPU="$1" P="${LOG_DIR}/laneA"; mkdir -p "$P"
    local cfg="$PLR_DIR/finetune_msdwild_10spks.yaml"
    local out="$EXP_ROOT/PAPER_SC_LibriSpeech_2spk_adapted1-10/models_finetuneMSDWILD_ourrecipe"

    log "LANE A on gpu $GPU -- THE SWAP (paper adapt + our finetune)"
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        guard_output "$out" "$cfg" "lane A" || { log "LANE A ABORTED"; return 1; }
        preflight_init "$PAPER_ADAPT" "lane A (paper adapt)" || { log "LANE A ABORTED"; return 1; }
    fi

    if [ "$SCORE_ONLY" != "1" ]; then
        # patience 400: this lane must NOT be cut short by early stopping on
        # MSDWild dev, which is 97% 5-10-speaker while train/test are 2-4 --
        # a plateau there says little about test. max_epochs 750 is the
        # config's (and the paper's) own budget, left alone deliberately.
        train_fresh "A swap (paper adapt -> our FT)" "$GPU" "$cfg" "$out" "$P/train.log" \
            --init-model-path "$PAPER_ADAPT" --init-epochs 90-100 \
            --early-stopping-patience 400
    fi
    score "A swap @paper-budget" "$GPU" "$PLR_DIR/infer_msdwild.yaml" "$out" "515-525" "$P/infer.log"
    score "A swap @final"        "$GPU" "$PLR_DIR/infer_msdwild.yaml" "$out" "auto"    "$P/infer.log"
    log "LANE A COMPLETE"
}

# ===========================================================================
# LANE B -- extend ebf paperlr MSDWild (gpu $LANE_B_GPU)
# Ended at exactly max_epochs 750 => CAP-bound, a bare resume would be a
# silent no-op (train.py's `for epoch in range(init_epoch, args.max_epochs)`).
# ===========================================================================
lane_B () {
    local GPU="$1" P="${LOG_DIR}/laneB"; mkdir -p "$P"
    local cfg="$EBF_DIR/finetune_msdwild_10spks.yaml"
    local out; out="$(yaml_get output_path "$cfg")"

    log "LANE B on gpu $GPU -- extend ebf paperlr MSDWild past its 750 cap"
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        preflight_resume "$out" "lane B (ebf MSDWild)" || { log "LANE B ABORTED"; return 1; }
    fi
    if [ "$SCORE_ONLY" != "1" ]; then
        train_resume "B ebf MSDWild extend" "$GPU" "$cfg" "$out" "$P/train.log" \
            --max-epochs "$EXTEND_MAX_EPOCHS" --early-stopping-patience 300
    fi
    score "B ebf MSDWild" "$GPU" "$EBF_DIR/infer_msdwild.yaml" "$out" "auto" "$P/infer.log"
    log "LANE B COMPLETE"
}

# ===========================================================================
# LANE C -- extend A1 conformer_k31 MSDWild (gpu $LANE_C_GPU)
# Stopped at 656 with max_epochs 750 still unreached => PATIENCE-bound, not
# cap-bound. Raising --max-epochs alone does nothing here; patience is the
# binding constraint and must go up as well. (The 36h queue ran it with
# --early-stopping-patience 150.)
# ===========================================================================
lane_C () {
    local GPU="$1" P="${LOG_DIR}/laneC"; mkdir -p "$P"
    local cfg="$CNF_DIR/finetune_msdwild_10spks.yaml"
    local out; out="$(yaml_get output_path "$cfg")"

    log "LANE C on gpu $GPU -- extend A1 conformer_k31 MSDWild past 656"
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        preflight_resume "$out" "lane C (conformer MSDWild)" || { log "LANE C ABORTED"; return 1; }
    fi
    if [ "$SCORE_ONLY" != "1" ]; then
        train_resume "C conformer MSDWild extend" "$GPU" "$cfg" "$out" "$P/train.log" \
            --max-epochs "$EXTEND_MAX_EPOCHS" --early-stopping-patience 300
    fi
    score "C conformer MSDWild" "$GPU" "$CNF_DIR/infer_msdwild.yaml" "$out" "auto" "$P/infer.log"

    # ---- stage 2: THE LR ARM -------------------------------------------
    # Extending to 900 only costs ~11h at ~2.8 min/epoch, so this lane has
    # room for a second, more interesting run.
    #
    # WHY: the finetune LR is a CONSTANT 1e-6 with no scheduler (verified
    # from tensorboard: 8078 logged points, distinct=1, min=max=1.000e-06;
    # train.py's noam path at line ~1083 only fires for optimizer: noam and
    # these configs use adam). And nothing converges within the budget --
    # train_DER is still descending near-linearly in the final 5% of the
    # run, while the only components still moving in the second half are
    # confusion (-10.8% conformer / -23.2% ebf) and attractor-existence
    # loss (-35.6% ebf, accuracy 96.4 -> 98.0). miss and FA plateau by
    # mid-run. That is truncated optimization, not convergence.
    #
    # This arm is a FRESH finetune from the same adapt init as A1 and lane
    # D, changing ONLY the learning rate -- so A1 (lr 1e-6), lane D (seed),
    # and this (lr $LR_ARM) form three single-knob variations of one
    # baseline, comparable at matched epochs.
    #
    # 1e-5 is a deliberate 10x probe rather than a cautious 3x: a 3x change
    # risks landing inside the noise floor and telling us nothing. If it
    # destabilises, that shows up in the first few tens of epochs in
    # train.log -- rerun with LR_ARM=3e-6.
    #
    # NOTE this is NOT a deviation-from-the-paper fix: 1e-6 is the paper's
    # own finetune LR. The paper reached 15.47 in ~525 epochs with it, so
    # the LR is not what separates us from them -- this arm tests whether
    # OUR starting point simply needs more optimization to get there.
    local out_lr="${out}_lr${LR_ARM}"
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        guard_output "$out_lr" "$cfg" "lane C stage 2" || { log "LANE C stage 2 ABORTED"; return 1; }
    fi
    if [ "$SCORE_ONLY" != "1" ]; then
        train_fresh "C2 conformer MSDWild lr$LR_ARM" "$GPU" "$cfg" "$out_lr" "$P/train_lr.log" \
            --lr "$LR_ARM" --early-stopping-patience 400
    fi
    score "C2 lr$LR_ARM @494-504" "$GPU" "$CNF_DIR/infer_msdwild.yaml" "$out_lr" "494-504" "$P/infer_lr.log"
    score "C2 lr$LR_ARM @final"   "$GPU" "$CNF_DIR/infer_msdwild.yaml" "$out_lr" "auto"    "$P/infer_lr.log"
    log "LANE C COMPLETE"
}

# ===========================================================================
# LANE D -- SEED REPLICATE of A1 (gpu $LANE_D_GPU)
# Identical config, identical adapt init, ONLY --seed differs. Scored at
# several matched epoch ranges so it can be compared against A1 at the same
# budget rather than only at its endpoint -- A1's own history is a series of
# resumes at different patience settings, so endpoint-vs-endpoint would
# confound seed with schedule.
# ===========================================================================
lane_D () {
    local GPU="$1" P="${LOG_DIR}/laneD"; mkdir -p "$P"
    local cfg="$CNF_DIR/finetune_msdwild_10spks.yaml"
    local base; base="$(yaml_get output_path "$cfg")"
    local out="${base}_seed${SEED_D}"
    local init; init="$(yaml_get init_model_path "$cfg")"

    log "LANE D on gpu $GPU -- seed replicate of A1 (seed $SEED_D)"
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        guard_output "$out" "$cfg" "lane D" || { log "LANE D ABORTED"; return 1; }
        preflight_init "$init" "lane D (conformer adapt)" || { log "LANE D ABORTED"; return 1; }
    fi
    if [ "$SCORE_ONLY" != "1" ]; then
        train_fresh "D conformer MSDWild seed$SEED_D" "$GPU" "$cfg" "$out" "$P/train.log" \
            --seed "$SEED_D" --early-stopping-patience 400
    fi
    # matched-epoch comparison points against A1 (17.39 @494-504, 17.07 @646-656)
    score "D seed$SEED_D @494-504" "$GPU" "$CNF_DIR/infer_msdwild.yaml" "$out" "494-504" "$P/infer.log"
    score "D seed$SEED_D @646-656" "$GPU" "$CNF_DIR/infer_msdwild.yaml" "$out" "646-656" "$P/infer.log"
    score "D seed$SEED_D @final"   "$GPU" "$CNF_DIR/infer_msdwild.yaml" "$out" "auto"    "$P/infer.log"
    log "LANE D COMPLETE"
}

# ---------------------------------------------------------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _cvd <<< "$CUDA_VISIBLE_DEVICES"
    [ -n "${_cvd[0]:-}" ] && LANE_A_GPU="${_cvd[0]}"
    LANE_B_GPU="${_cvd[1]:-${_cvd[0]}}"
    LANE_C_GPU="${_cvd[2]:-${_cvd[0]}}"
    LANE_D_GPU="${_cvd[3]:-${_cvd[0]}}"
    log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> lanes A/B/C/D on $LANE_A_GPU/$LANE_B_GPU/$LANE_C_GPU/$LANE_D_GPU"
    if [ "${#_cvd[@]}" -lt 4 ] && [ -z "$ONLY_LANE" ]; then
        log "WARNING: fewer than 4 devices listed -- lanes will SHARE a GPU."
        log "WARNING: pass ONLY_LANE=... or list four devices."
    fi
    unset CUDA_VISIBLE_DEVICES
fi

for f in "$PLR_DIR/finetune_msdwild_10spks.yaml" "$CNF_DIR/finetune_msdwild_10spks.yaml" \
         "$EBF_DIR/finetune_msdwild_10spks.yaml"; do
    [ -f "$f" ] || { echo "ERROR: $f not found -- run from the repo root." >&2; exit 1; }
done

log "4-GPU MSDWild queue starting (no wall-clock caps -- each lane runs to its own patience/cap)"
log "  A gpu=$LANE_A_GPU  the swap (paper adapt + our finetune)"
log "  B gpu=$LANE_B_GPU  extend ebf paperlr MSDWild   (cap -> $EXTEND_MAX_EPOCHS)"
log "  C gpu=$LANE_C_GPU  extend A1 conformer MSDWild  (patience-bound: patience -> 300)"
log "                     then stage 2: fresh conformer finetune at lr=$LR_ARM (vs 1e-6)"
log "  D gpu=$LANE_D_GPU  seed replicate of A1         (seed $SEED_D)"
log "  paper adapt: $PAPER_ADAPT"
log "  score_only=$SCORE_ONLY   logs: $LOG_DIR"
rm -f "$STOP_FILE"

pids=()
lane_enabled A && { lane_A "$LANE_A_GPU" & pids+=($!); }
lane_enabled B && { lane_B "$LANE_B_GPU" & pids+=($!); }
lane_enabled C && { lane_C "$LANE_C_GPU" & pids+=($!); }
lane_enabled D && { lane_D "$LANE_D_GPU" & pids+=($!); }
for pid in "${pids[@]}"; do wait "$pid"; done

log "4-GPU MSDWild queue finished."
log ""
log "READ THE RESULTS IN THIS ORDER:"
log "  1. Lane A @515-525 vs the paper's 15.47. That single number decides"
log "     whether the remaining gap is the ADAPT stage or our finetune"
log "     EXECUTION, and therefore where every future run should go."
log "  2. Lane D vs A1 at the SAME epoch range. That is the noise floor for"
log "     every architecture comparison in results.csv -- interpret lanes"
log "     B and C only after you know it."
log "  3. Lanes B/C deltas. Use a PAIRED bootstrap, not the pooled point"
log "     estimate: same-lineage comparisons resolve ~0.3 DER, while the"
log "     often-quoted ~1.5 MSDWild floor only applies across architectures."
