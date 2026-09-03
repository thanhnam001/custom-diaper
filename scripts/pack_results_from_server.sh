#!/bin/bash
# Pack everything scripts/run_5day_queue.sh PRODUCED into one archive, to
# bring back from the server. This is the return leg of
# scripts/pack_weights_for_server.sh (which sends warm-starts the other way).
#
# Run it ON THE SERVER, from the repo root:
#
#   ./scripts/pack_results_from_server.sh                  # ~0.5 GB, no weights
#   DRY_RUN=1 ./scripts/pack_results_from_server.sh        # just size it, pack nothing
#   MODE=min ./scripts/pack_results_from_server.sh         # ~5 MB, scores + logs only
#   MODE=avg ./scripts/pack_results_from_server.sh         # + the averaged checkpoints
#   MODE=all ./scripts/pack_results_from_server.sh         # + every checkpoint on disk
#
# ALWAYS RUN DRY_RUN=1 FIRST. A finished two-lane queue leaves ~30 GB of
# checkpoints on disk; MODE=all packs all of it and you probably do not want
# that over scp.
#
# MODES
#   min      MANIFEST + dscore score logs + the queue's own stage logs + the
#            configs that produced them. Answers "did it work, what are the
#            numbers" and nothing else. Few MB -- send this first while the
#            big one uploads.
#   results  (default) min + TensorBoard event files + every predicted RTTM.
#            Enough to plot the training curves, re-score at another collar,
#            and do per-file DER analysis locally. No weights, so ~0.5 GB.
#   avg      results + the last $KEEP (default 10) checkpoints of each output
#            directory -- exactly the window infer.py averaged, so you can
#            reproduce or re-run inference locally. ~90 MB per checkpoint.
#   all      results + every checkpoint still on disk (train.py keeps the
#            last 30 by default). Tens of GB. Use only to migrate a run.
#
# Nothing here is required to *finish* the queue -- it only reads. A stage
# that never ran is reported as missing and skipped, so this is safe to run
# mid-queue to check on progress, and safe to re-run afterwards.
#
# The archive has two top-level directories:
#   experiments/...  relative to the server's experiments/10attractors
#   repo/...         relative to this repo root (logs/, the configs used)
# plus MANIFEST.txt at the root. The script prints the exact extract
# commands when it finishes.

set -eu

MODE="${MODE:-results}"
OUT="${OUT:-diaper_5day_results.tar}"
KEEP="${KEEP:-10}"
COMPRESS="${COMPRESS:-0}"
SPLIT="${SPLIT:-}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_LANE="${ONLY_LANE:-}"
INCLUDE_D1="${INCLUDE_D1:-1}"

# Same defaults as run_5day_queue.sh -- override together if you moved things.
LOG_DIR="${LOG_DIR:-logs/5day_queue}"
EBF_DIR="${EBF_DIR:-models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_ebf}"
CNF_DIR="${CNF_DIR:-models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_conformer_k31}"
EBF_OLD_INFER_RAMC="${EBF_OLD_INFER_RAMC:-models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer/infer_ramc.yaml}"
CNF_OLD_INFER_RAMC="${CNF_OLD_INFER_RAMC:-models/10attractors/SC_LibriSpeech_2spk_conformer_kernel31_mlp_fresh_2500h_spkcounting_headoff/infer_ramc_mlp.yaml}"

case "$MODE" in
    min|results|avg|all) ;;
    *) echo "ERROR: MODE must be one of: min results avg all (got '$MODE')" >&2; exit 1 ;;
esac

# Identical to run_5day_queue.sh's helper, on purpose: if that one resolves a
# path, so does this one, including when a config uses odd spacing.
yaml_get () {  # yaml_get <key> <file>
    grep "^$1:" "$2" | head -1 | sed "s|^$1: *||"
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
LIST_EXP="$WORK/list_exp"
LIST_LOG="$WORK/list_log"
STAGE="$WORK/stage"
MANIFEST="$WORK/MANIFEST.txt"
: > "$LIST_EXP"; : > "$LIST_LOG"; : > "$MANIFEST"
mkdir -p "$STAGE/repo"

BYTES=0
NFILES=0
MISSING=0
ADDED_N=0
ADDED_B=0

say () { printf '%s\n' "$*"; printf '%s\n' "$*" >> "$MANIFEST"; }
note () { printf '%s\n' "$*" >&2; }   # progress only, never in the manifest

human () {  # bytes -> human, no bc dependency
    awk -v b="$1" 'BEGIN{
        if (b >= 1073741824) printf "%.1f GB", b/1073741824;
        else if (b >= 1048576) printf "%.0f MB", b/1048576;
        else printf "%.0f KB", b/1024;
    }'
}

# ---------------------------------------------------------------------------
# Discover the experiments root from the configs rather than hardcoding it:
# the adapt output_path's parent is experiments/10attractors on the server.
# Everything the queue writes -- including the D1 diagnostic, which lands in
# the OLD lineages' directories -- lives under it.
# ---------------------------------------------------------------------------
_probe=""
for d in "$EBF_DIR" "$CNF_DIR"; do
    [ -f "$d/train_10spks.yaml" ] || continue
    _probe=$(yaml_get output_path "$d/train_10spks.yaml")
    [ -n "$_probe" ] && break
done
if [ -z "$_probe" ]; then
    echo "ERROR: could not read output_path from any train_10spks.yaml." >&2
    echo "       Run from the repo root, or set EBF_DIR/CNF_DIR." >&2
    exit 1
fi
EXP_ROOT="${EXP_ROOT:-$(dirname "$_probe")}"

# ---------------------------------------------------------------------------
# add_files <find-root> [find args...]
# Adds every regular file found, rebased onto $EXP_ROOT. Anything that
# escapes $EXP_ROOT is dropped with a warning rather than silently producing
# an archive that will not extract where the header says it does.
# Sets ADDED_N / ADDED_B for the caller to report.
# ---------------------------------------------------------------------------
add_files () {
    local root="$1"; shift
    local f n=0 b=0 sz
    ADDED_N=0; ADDED_B=0
    [ -e "$root" ] || return 0
    while IFS= read -r f; do
        case "$f" in
            "$EXP_ROOT"/*) ;;
            *) note "  WARN outside EXP_ROOT, skipped: $f"; continue ;;
        esac
        printf '%s\n' "${f#"$EXP_ROOT"/}" >> "$LIST_EXP"
        sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
        b=$(( b + sz )); n=$(( n + 1 ))
    done < <(find "$root" "$@" -type f 2>/dev/null)
    BYTES=$(( BYTES + b )); NFILES=$(( NFILES + n ))
    ADDED_N=$n; ADDED_B=$b
}

# ---------------------------------------------------------------------------
# stage_repo_file <src> <path under repo/ in the archive>
# The repo-side extras (configs, the driver) are a handful of small text
# files, so they are COPIED into a staging tree that already has the final
# archive layout. That costs nothing at this size and removes a whole class
# of bug: EBF_DIR and friends can be absolute, relative, or point outside
# the repo entirely, and the archive still comes out identical.
# ---------------------------------------------------------------------------
stage_repo_file () {
    local src="$1" dst="$STAGE/repo/$2" sz
    [ -f "$src" ] || return 0
    mkdir -p "$(dirname "$dst")"
    cp -p "$src" "$dst"
    sz=$(stat -c %s "$src" 2>/dev/null || echo 0)
    BYTES=$(( BYTES + sz )); NFILES=$(( NFILES + 1 ))
    ADDED_N=$(( ADDED_N + 1 )); ADDED_B=$(( ADDED_B + sz ))
}

# ---------------------------------------------------------------------------
# The log directory gets its own tar root so that an absolute LOG_DIR (or one
# outside the repo) still lands at repo/logs/<name>/... instead of being
# stored with a stripped leading slash. Entries are relative to the log
# directory's PARENT, so the default LOG_DIR=logs/5day_queue and an absolute
# /somewhere/5day_queue both come out as repo/logs/5day_queue/...
# ---------------------------------------------------------------------------
LOG_PARENT=""
add_log_files () {
    local root="$1"; shift
    local f n=0 b=0 sz rel
    ADDED_N=0; ADDED_B=0
    [ -d "$root" ] || return 0
    LOG_PARENT="$(cd "$(dirname "$root")" && pwd)"
    while IFS= read -r f; do
        rel="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
        rel="${rel#"$LOG_PARENT"/}"
        printf '%s\n' "$rel" >> "$LIST_LOG"
        sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
        b=$(( b + sz )); n=$(( n + 1 ))
    done < <(find "$root" "$@" -type f 2>/dev/null)
    BYTES=$(( BYTES + b )); NFILES=$(( NFILES + n ))
    ADDED_N=$n; ADDED_B=$b
}

# ---------------------------------------------------------------------------
# pack_stage <lane> <stage label> <output dir>
# One training stage's worth of output: checkpoints (mode-dependent),
# TensorBoard, and any inference predictions written beside them.
#
# The adapt directory is the PARENT of the finetune directories, so this
# never recurses blindly -- it names models/, tensorboard/ and *_test_pred/
# explicitly, and each finetune dir is passed in as its own stage.
# ---------------------------------------------------------------------------
pack_stage () {
    local lane="$1" label="$2" dir="$3"
    local ck_n=0 ck_first="" ck_last="" tb=0 rt=0 sc=0 b0=$BYTES n_packed=0

    if [ ! -d "$dir" ]; then
        say "  [$lane] $label"
        say "        NOT RUN -- no directory at $dir"
        MISSING=$(( MISSING + 1 ))
        return 0
    fi

    # -- checkpoints ------------------------------------------------------
    local epochs=()
    if [ -d "$dir/models" ]; then
        mapfile -t epochs < <(
            find "$dir/models" -maxdepth 1 -name 'checkpoint_*.tar' -printf '%f\n' 2>/dev/null \
            | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n
        )
    fi
    ck_n=${#epochs[@]}
    if [ "$ck_n" -gt 0 ]; then
        ck_first="${epochs[0]}"; ck_last="${epochs[$(( ck_n - 1 ))]}"
        local picked=()
        case "$MODE" in
            all) picked=("${epochs[@]}") ;;
            avg) mapfile -t picked < <(printf '%s\n' "${epochs[@]}" | tail -n "$KEEP") ;;
            *)   picked=() ;;
        esac
        n_packed=${#picked[@]}
        local e sz
        for e in ${picked[@]+"${picked[@]}"}; do
            printf '%s\n' "${dir#"$EXP_ROOT"/}/models/checkpoint_${e}.tar" >> "$LIST_EXP"
            sz=$(stat -c %s "$dir/models/checkpoint_${e}.tar" 2>/dev/null || echo 0)
            BYTES=$(( BYTES + sz )); NFILES=$(( NFILES + 1 ))
        done
    fi

    # -- TensorBoard ------------------------------------------------------
    if [ "$MODE" != "min" ] && [ -d "$dir/tensorboard" ]; then
        add_files "$dir/tensorboard"
        tb=$ADDED_N
    fi

    # -- predictions and scores ------------------------------------------
    local p
    for p in "$dir"/*_test_pred; do
        [ -d "$p" ] || continue
        # dscore logs are the numbers themselves -- always take them.
        add_files "$p" -maxdepth 1 -name 'dscore*.log'
        sc=$(( sc + ADDED_N ))
        if [ "$MODE" != "min" ]; then
            add_files "$p" -name '*.rttm'
            rt=$(( rt + ADDED_N ))
        fi
    done

    say "  [$lane] $label"
    if [ "$ck_n" -gt 0 ]; then
        say "        checkpoints  $ck_n on disk (ep ${ck_first}..${ck_last}), packing $n_packed"
    else
        say "        checkpoints  none -- stage started but wrote nothing, or was skipped"
    fi
    say "        tensorboard  $tb file(s)   rttms $rt   dscore logs $sc"
    local ov
    while IFS= read -r ov; do
        say "        $ov"
    done < <(grep -h -- '\*\*\* OVERALL \*\*\*' "$dir"/*_test_pred/dscore*.log 2>/dev/null \
             | awk '{printf "DER %s  JER %s\n", $4, $5}')
    say "        $(human $(( BYTES - b0 ))) added"
}

# ---------------------------------------------------------------------------
# pack_lane <letter> <config dir> <old infer config, for D1>
# ---------------------------------------------------------------------------
pack_lane () {
    local L="$1" CFG="$2" D1_CFG="$3"

    if [ ! -d "$CFG" ]; then
        say "LANE $L -- config dir missing: $CFG (skipped)"
        return 0
    fi

    local pre_out adapt_out ft_ramc_out ft_msd_out
    pre_out=$(yaml_get output_path "$CFG/train.yaml")
    adapt_out=$(yaml_get output_path "$CFG/train_10spks.yaml")
    ft_ramc_out=$(yaml_get output_path "$CFG/finetune_ramc_10spks.yaml")
    ft_msd_out=$(yaml_get output_path "$CFG/finetune_msdwild_10spks.yaml")

    say ""
    say "LANE $L  ($(basename "$CFG"))"
    say "  adapt output: $adapt_out"

    # Only if RUN_PRETRAIN=1 was used -- detected on disk, not assumed.
    if [ -n "$pre_out" ] && [ -d "$pre_out" ]; then
        pack_stage "$L" "pretrain (2 spk, 2500h)" "$pre_out"
    fi

    pack_stage "$L" "adapt (10 spk, 2500h)"           "$adapt_out"
    pack_stage "$L" "finetune RAMC"                   "$ft_ramc_out"
    pack_stage "$L" "finetune MSDWild"                "$ft_msd_out"
    pack_stage "$L" "F1 finetune RAMC @ subsampling5" "${ft_ramc_out}_sub5"
    pack_stage "$L" "F2 finetune RAMC @ lr 3e-6"      "${ft_ramc_out}_lr3e-6"
    pack_stage "$L" "F2 finetune RAMC @ lr 1e-5"      "${ft_ramc_out}_lr1e-5"

    # -- D1 diagnostic ----------------------------------------------------
    # Re-scores an EXISTING old-recipe checkpoint at subsampling 10, and
    # writes into that old lineage's rttms_dir. Its weights came FROM the
    # local machine (pack_weights_for_server.sh sent them), so take the
    # outputs only and never the checkpoints, in any mode. Taking the whole
    # rttms_dir also brings back that lineage's original subsampling-5
    # RTTMs, which is exactly what the new run is compared against.
    if [ "$INCLUDE_D1" = "1" ] && [ -f "$D1_CFG" ]; then
        local d1_rttms n_sc n_rt b1=$BYTES
        d1_rttms=$(yaml_get rttms_dir "$D1_CFG")
        say "  [$L] D1 diagnostic (subsampling 10 re-score, outputs only)"
        if [ -d "$d1_rttms" ]; then
            add_files "$d1_rttms" -maxdepth 1 -name 'dscore*.log'
            n_sc=$ADDED_N; n_rt=0
            if [ "$MODE" != "min" ]; then
                add_files "$d1_rttms" -name '*.rttm'; n_rt=$ADDED_N
            fi
            say "        $d1_rttms"
            say "        dscore logs $n_sc   rttms $n_rt   $(human $(( BYTES - b1 ))) added"
            local ov
            while IFS= read -r ov; do
                say "        $ov"
            done < <(grep -h -- '\*\*\* OVERALL \*\*\*' "$d1_rttms"/dscore*.log 2>/dev/null \
                     | awk '{printf "DER %s  JER %s\n", $4, $5}')
        else
            say "        NOT RUN -- no directory at $d1_rttms"
            MISSING=$(( MISSING + 1 ))
        fi
    fi
}

# ---------------------------------------------------------------------------

say "DiaPer 5-day queue -- results pack"
say "  packed   : $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname 2>/dev/null || echo unknown-host)"
if [ "$MODE" = "avg" ]; then
    say "  mode     : avg (KEEP=$KEEP checkpoints per stage)"
else
    say "  mode     : $MODE"
fi
say "  exp root : $EXP_ROOT"
say "  repo     : $(pwd)"
say "  commit   : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
say ""
say "A stage marked NOT RUN never started, or was disabled (RUN_MSDWILD=0,"
say "RUN_FILLERS=0, RUN_DIAGNOSTICS=0). A stage with checkpoints but no DER"
say "line finished training but was not scored -- most likely the queue was"
say "stopped before its RAMC inference reached the front of the cross-lane"
say "lock. Its weights are in the archive under MODE=avg or all, so it can"
say "still be scored later."
say ""

if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "A" ]; then
    pack_lane A "$EBF_DIR" "$EBF_OLD_INFER_RAMC"
fi
if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "B" ]; then
    pack_lane B "$CNF_DIR" "$CNF_OLD_INFER_RAMC"
fi

# ---------------------------------------------------------------------------
# Repo-side: the queue's own logs, and the configs that produced everything
# above. The configs are in git, but a run is only reproducible against the
# exact YAML it used, and the mid-run overrides (--lr, --subsampling,
# --output-path) live in the driver -- so ship the driver too.
# ---------------------------------------------------------------------------
say ""
say "REPO SIDE"
b0=$BYTES
if [ -d "$LOG_DIR" ]; then
    # Skip the lock directory: it is a mutex, not a result.
    add_log_files "$LOG_DIR" -not -path '*/.ramc_infer.lock/*'
    say "  logs     $LOG_DIR -- $ADDED_N file(s), $(human $ADDED_B)"
else
    say "  logs     $LOG_DIR MISSING -- was the queue started from this directory?"
    MISSING=$(( MISSING + 1 ))
fi
for d in "$EBF_DIR" "$CNF_DIR"; do
    [ -d "$d" ] || continue
    ADDED_N=0; ADDED_B=0
    for y in "$d"/*.yaml; do
        [ -f "$y" ] || continue
        stage_repo_file "$y" "configs/$(basename "$d")/$(basename "$y")"
    done
    say "  configs  $d -- $ADDED_N yaml"
done
ADDED_N=0; ADDED_B=0
stage_repo_file scripts/run_5day_queue.sh "run_5day_queue.sh"
stage_repo_file scripts/pack_results_from_server.sh "pack_results_from_server.sh"
say "  driver   $ADDED_N script(s) (the per-stage --lr/--subsampling overrides live here)"
say "  $(human $(( BYTES - b0 ))) added"

say ""
say "TOTAL: $NFILES files, $(human "$BYTES")"
if [ "$MISSING" -gt 0 ]; then
    say "       $MISSING expected item(s) missing -- see NOT RUN above."
fi
say ""
say "Extract on the local machine:"
say "  mkdir -p ~/diaper_5day && tar -xf $(basename "$OUT") -C ~/diaper_5day"
say "Then, to drop the experiment outputs into a local checkout:"
say "  tar -xf $(basename "$OUT") -C experiments/10attractors --strip-components=1 experiments"

if [ "$DRY_RUN" = "1" ]; then
    printf '\n%s\n' "DRY_RUN=1 -- nothing written. Re-run without it to pack $(human "$BYTES")."
    exit 0
fi

if [ ! -s "$LIST_EXP" ]; then
    # The configs and driver alone would still make a non-empty archive, and
    # calling that "results" would be actively misleading. Every stage above
    # said NOT RUN, so either this is not the machine the queue ran on, or
    # EXP_ROOT is wrong.
    echo "ERROR: not one experiment output was found under" >&2
    echo "         $EXP_ROOT" >&2
    echo "       Every stage reported NOT RUN, so there are no results to pack." >&2
    echo "       Run this on the SERVER, from the repo root the queue ran in." >&2
    echo "       If the outputs moved, point EXP_ROOT at their parent directory." >&2
    echo "       (Set FORCE=1 to pack the configs and logs anyway.)" >&2
    [ "${FORCE:-0}" = "1" ] || exit 1
fi
if [ "$NFILES" -eq 0 ]; then
    echo "ERROR: nothing to pack at all. Wrong directory?" >&2
    exit 1
fi

printf '\n%s\n' "packing $NFILES files, $(human "$BYTES") -> $OUT"

rm -f "$OUT"
# Two roots in one archive: create from the experiments side, then append the
# repo side. --transform prefixes each so they cannot collide. Append (-r)
# only works on an uncompressed archive, which is why compression, if asked
# for, happens afterwards rather than via -z.
if [ -s "$LIST_EXP" ]; then
    tar -cf "$OUT" -C "$EXP_ROOT" --transform="s,^,experiments/," -T "$LIST_EXP"
else
    tar -cf "$OUT" --files-from=/dev/null
fi
if [ -s "$LIST_LOG" ]; then
    tar -rf "$OUT" -C "$LOG_PARENT" --transform="s,^,repo/logs/," -T "$LIST_LOG"
fi
if [ -d "$STAGE/repo" ]; then
    tar -rf "$OUT" -C "$STAGE" repo
fi
tar -rf "$OUT" -C "$WORK" MANIFEST.txt

if [ "$COMPRESS" = "1" ]; then
    printf '%s\n' "compressing (checkpoints are float tensors and barely shrink; logs and RTTMs do)"
    gzip -f "$OUT"
    OUT="${OUT}.gz"
fi

printf '%s\n' "done: $OUT ($(du -h "$OUT" | cut -f1))"

SPLIT_OK=0
if [ -n "$SPLIT" ]; then
    printf '%s\n' "splitting into ${SPLIT} parts for transfer"
    split -b "$SPLIT" -d -a 3 "$OUT" "${OUT}.part"
    whole=$(stat -c %s "$OUT")
    parts=$(cat "${OUT}.part"* | wc -c)
    if [ "$whole" = "$parts" ]; then
        # Only now is it safe to drop the original: a 30 GB archive plus its
        # parts would otherwise need 60 GB of scratch on the server.
        rm -f "$OUT"
        SPLIT_OK=1
        printf '%s\n' "  $(ls -1 "${OUT}.part"* | wc -l) parts, original removed after size check"
    else
        printf '%s\n' "  WARNING: parts total $parts != archive $whole -- keeping the original," >&2
        printf '%s\n' "  WARNING: do not trust the parts." >&2
        rm -f "${OUT}.part"*
    fi
fi

case "$OUT" in
    /*) OUT_ABS="$OUT" ;;
    *)  OUT_ABS="$(pwd)/$OUT" ;;
esac
OUT_BASE="$(basename "$OUT")"

printf '\n%s\n' "Copy it home:"
if [ "$SPLIT_OK" = "1" ]; then
    printf '%s\n' "  scp '<user>@<server>:${OUT_ABS}.part*' ."
    printf '%s\n' "  cat ${OUT_BASE}.part* > ${OUT_BASE}"
else
    printf '%s\n' "  scp <user>@<server>:${OUT_ABS} ."
fi

cat <<EOF

Then, locally:
  mkdir -p ~/diaper_5day && tar -xf ${OUT_BASE} -C ~/diaper_5day
  cat ~/diaper_5day/MANIFEST.txt

MANIFEST.txt lists every stage, whether it ran, how many checkpoints it
wrote, and its dscore DER -- read it before unpacking anything else.

To merge the experiment outputs into a local checkout instead:
  tar -xf ${OUT_BASE} -C experiments/10attractors --strip-components=1 experiments
EOF
