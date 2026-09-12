#!/bin/bash
# Pack everything scripts/run_4gpu_msdwild_queue.sh produced, to bring back
# from the server. Run ON THE SERVER, from the repo root:
#
#   ./scripts/pack_4gpu_msdwild_results.sh                  # no weights
#   DRY_RUN=1 ./scripts/pack_4gpu_msdwild_results.sh        # just size it
#   MODE=avg ./scripts/pack_4gpu_msdwild_results.sh         # + last $KEEP checkpoints
#
# MODES: min (scores+logs only) / results (default, +tensorboard+rttms) /
# avg (+ last $KEEP checkpoints per stage) / all (+ every checkpoint on disk).
# Same semantics as pack_36h_resume_results.sh.
#
# CHECKPOINT PRUNING -- read before choosing a MODE. train.py's
# --keep-last-n-checkpoints defaults to 30 (train.py:794) and none of these
# configs override it, so prune_checkpoints() (backend/models.py:56) unlinks
# by mtime after every save. At most the last 30 epochs of any stage still
# exist, which means MODE=all is at most 30 checkpoints here, not the full
# history -- unlike the 36h queue, where some stages were short enough to
# survive intact. It also means a stage's EARLIER scored ranges are gone
# from disk even though their dscore logs remain; this script flags that
# case per stage as "scored range PRUNED" so a range you cannot re-score
# locally is never mistaken for one you can.
#
# Pruning only runs inside save_checkpoint, so once training stops nothing
# further is deleted -- there is no rush to pack a finished lane.
#
# A stage marked NOT RUN never started (dependency missing, or the lane
# aborted preflight). A stage with checkpoints but no DER line trained but
# was not scored -- its weights are in the archive under MODE=avg/all, so
# score it locally afterward.

set -eu

MODE="${MODE:-results}"
OUT="${OUT:-diaper_4gpu_msdwild_results.tar}"
KEEP="${KEEP:-10}"
COMPRESS="${COMPRESS:-0}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_LANE="${ONLY_LANE:-}"

# Must match the values the queue ran with -- they are baked into directory
# names (..._lr$LR_ARM, ..._seed$SEED_D), so a mismatch silently reports the
# stage as NOT RUN.
SEED_D="${SEED_D:-7}"
LR_ARM="${LR_ARM:-1e-5}"

LOG_DIR="${LOG_DIR:-logs/4gpu_msdwild_queue}"
PLR_DIR="${PLR_DIR:-models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr}"
CNF_DIR="${CNF_DIR:-models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_conformer_k31}"
EBF_DIR="${EBF_DIR:-models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer}"

case "$MODE" in
    min|results|avg|all) ;;
    *) echo "ERROR: MODE must be one of: min results avg all (got '$MODE')" >&2; exit 1 ;;
esac

yaml_get () { grep "^$1:" "$2" | head -1 | sed "s|^$1: *||"; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
LIST_EXP="$WORK/list_exp"; LIST_LOG="$WORK/list_log"; MANIFEST="$WORK/MANIFEST.txt"
: > "$LIST_EXP"; : > "$LIST_LOG"; : > "$MANIFEST"
mkdir -p "$WORK/stage/repo"

BYTES=0; NFILES=0; MISSING=0; ADDED_N=0; ADDED_B=0; PRUNED=0

say () { printf '%s\n' "$*"; printf '%s\n' "$*" >> "$MANIFEST"; }
note () { printf '%s\n' "$*" >&2; }
human () { awk -v b="$1" 'BEGIN{
    if (b>=1073741824) printf "%.1f GB", b/1073741824;
    else if (b>=1048576) printf "%.0f MB", b/1048576;
    else printf "%.0f KB", b/1024;}'; }

_probe=$(yaml_get output_path "$CNF_DIR/finetune_msdwild_10spks.yaml")
if [ -z "$_probe" ]; then
    echo "ERROR: could not read output_path from $CNF_DIR/finetune_msdwild_10spks.yaml." >&2
    echo "       Run from the repo root, or set CNF_DIR." >&2
    exit 1
fi
# output_path is .../experiments/10attractors/<run>/models_finetuneMSDWILD ->
# EXP_ROOT is .../experiments/10attractors
EXP_ROOT="${EXP_ROOT:-$(dirname "$(dirname "$_probe")")}"

# Lane A writes beside the paper's adapt weights, in a PAPER_-prefixed
# directory the queue constructs directly rather than reading from a yaml
# (its config's own output_path belongs to the paperlr baseline, which
# guard_output() refuses to overwrite).
PAPER_FT="${PAPER_FT:-$EXP_ROOT/PAPER_SC_LibriSpeech_2spk_adapted1-10/models_finetuneMSDWILD_ourrecipe}"

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
        sz=$(stat -c %s "$f" 2>/dev/null || echo 0); b=$(( b + sz )); n=$(( n + 1 ))
    done < <(find "$root" "$@" -type f 2>/dev/null)
    BYTES=$(( BYTES + b )); NFILES=$(( NFILES + n )); ADDED_N=$n; ADDED_B=$b
}

stage_repo_file () {
    local src="$1" dst="$WORK/stage/repo/$2" sz
    [ -f "$src" ] || return 0
    mkdir -p "$(dirname "$dst")"; cp -p "$src" "$dst"
    sz=$(stat -c %s "$src" 2>/dev/null || echo 0)
    BYTES=$(( BYTES + sz )); NFILES=$(( NFILES + 1 ))
    ADDED_N=$(( ADDED_N + 1 )); ADDED_B=$(( ADDED_B + sz ))
}

LOG_PARENT=""
add_log_files () {
    local root="$1"; shift
    local f n=0 b=0 sz rel
    ADDED_N=0; ADDED_B=0
    [ -d "$root" ] || return 0
    LOG_PARENT="$(cd "$(dirname "$root")" && pwd)"
    while IFS= read -r f; do
        rel="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"; rel="${rel#"$LOG_PARENT"/}"
        printf '%s\n' "$rel" >> "$LIST_LOG"
        sz=$(stat -c %s "$f" 2>/dev/null || echo 0); b=$(( b + sz )); n=$(( n + 1 ))
    done < <(find "$root" "$@" -type f 2>/dev/null)
    BYTES=$(( BYTES + b )); NFILES=$(( NFILES + n )); ADDED_N=$n; ADDED_B=$b
}

# pack_stage <lane> <label> <output-dir>
pack_stage () {
    local lane="$1" label="$2" dir="$3"
    local ck_n=0 ck_first="" ck_last="" tb=0 rt=0 sc=0 b0=$BYTES n_packed=0

    if [ ! -d "$dir" ]; then
        say "  [$lane] $label"; say "        NOT RUN -- no directory at $dir"
        MISSING=$(( MISSING + 1 )); return 0
    fi

    local epochs=()
    if [ -d "$dir/models" ]; then
        mapfile -t epochs < <(find "$dir/models" -maxdepth 1 -name 'checkpoint_*.tar' -printf '%f\n' 2>/dev/null \
            | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n)
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

    if [ "$MODE" != "min" ] && [ -d "$dir/tensorboard" ]; then
        add_files "$dir/tensorboard"; tb=$ADDED_N
    fi
    local p
    for p in "$dir"/*_test_pred; do
        [ -d "$p" ] || continue
        add_files "$p" -maxdepth 1 -name 'dscore*.log'; sc=$(( sc + ADDED_N ))
        if [ "$MODE" != "min" ]; then
            add_files "$p" -name '*.rttm'; rt=$(( rt + ADDED_N ))
        fi
    done

    say "  [$lane] $label"
    if [ "$ck_n" -gt 0 ]; then
        say "        checkpoints  $ck_n on disk (ep ${ck_first}..${ck_last}), packing $n_packed"
    else
        say "        checkpoints  none -- stage started but wrote nothing, or was skipped"
    fi
    say "        tensorboard  $tb file(s)   rttms $rt   dscore logs $sc"

    # DER per scored range, and whether that range can still be re-scored.
    # score() embeds the epoch range in the dscore log name deliberately.
    local lg base rng hi der jer
    for lg in "$dir"/*_test_pred/dscore*.log; do
        [ -f "$lg" ] || continue
        base=$(basename "$lg")
        rng=$(printf '%s' "$base" | sed -nE 's/.*epochs([0-9]+-[0-9]+)\.log/\1/p')
        der=$(grep -h -- '\*\*\* OVERALL \*\*\*' "$lg" 2>/dev/null | awk '{print $4}' | head -1)
        jer=$(grep -h -- '\*\*\* OVERALL \*\*\*' "$lg" 2>/dev/null | awk '{print $5}' | head -1)
        if [ -z "$der" ]; then
            say "        epochs ${rng:-?}  no OVERALL line -- scoring failed"
            continue
        fi
        if [ -n "$rng" ] && [ "$ck_n" -gt 0 ]; then
            hi="${rng#*-}"
            if [ "$hi" -lt "$ck_first" ] 2>/dev/null; then
                say "        epochs $rng  DER $der  JER $jer   <-- scored range PRUNED, cannot re-score"
                PRUNED=$(( PRUNED + 1 ))
                continue
            fi
        fi
        say "        epochs ${rng:-?}  DER $der  JER $jer"
    done
    say "        $(human $(( BYTES - b0 ))) added"
}

say "DiaPer 4-GPU MSDWild queue -- results pack"
say "  packed   : $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname 2>/dev/null || echo unknown-host)"
say "  mode     : $MODE$( [ "$MODE" = "avg" ] && echo " (KEEP=$KEEP)")"
say "  arms     : LR_ARM=$LR_ARM  SEED_D=$SEED_D"
say "  exp root : $EXP_ROOT"
say "  repo     : $(pwd)"
say "  commit   : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
say ""

cnf_msd=$(yaml_get output_path "$CNF_DIR/finetune_msdwild_10spks.yaml")
ebf_msd=$(yaml_get output_path "$EBF_DIR/finetune_msdwild_10spks.yaml")

if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "A" ]; then
    say "LANE A -- the swap (paper adapt + our finetune)"
    pack_stage A "A  paper adapt -> our MSDWild FT" "$PAPER_FT"
fi

if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "B" ]; then
    say ""
    say "LANE B -- extend ebf paperlr MSDWild past its 750 cap"
    pack_stage B "B  ebf paperlr MSDWild (extended)" "$ebf_msd"
fi

if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "C" ]; then
    say ""
    say "LANE C -- extend A1, then the LR arm"
    pack_stage C "C1 conformer_k31 MSDWild (extended)" "$cnf_msd"
    pack_stage C "C2 conformer_k31 MSDWild @ lr$LR_ARM" "${cnf_msd}_lr${LR_ARM}"
fi

if [ -z "$ONLY_LANE" ] || [ "$ONLY_LANE" = "D" ]; then
    say ""
    say "LANE D -- seed replicate of A1"
    pack_stage D "D  conformer_k31 MSDWild @ seed$SEED_D" "${cnf_msd}_seed${SEED_D}"
fi

say ""
say "REPO SIDE"
b0=$BYTES
if [ -d "$LOG_DIR" ]; then
    # No RAMC in this queue, so no .ramc_infer.lock to exclude. STOP is a
    # sentinel, not a result -- packing it would make a local re-run of the
    # queue refuse to score anything.
    add_log_files "$LOG_DIR" -not -name 'STOP'
    say "  logs     $LOG_DIR -- $ADDED_N file(s), $(human $ADDED_B)"
else
    say "  logs     $LOG_DIR MISSING -- was the queue started from this directory?"
    MISSING=$(( MISSING + 1 ))
fi
ADDED_N=0; ADDED_B=0
stage_repo_file scripts/run_4gpu_msdwild_queue.sh "run_4gpu_msdwild_queue.sh"
stage_repo_file scripts/pack_weights_for_4gpu_msdwild_queue.sh "pack_weights_for_4gpu_msdwild_queue.sh"
say "  driver   $ADDED_N script(s) (per-lane patience/LR overrides live here)"
say "  $(human $(( BYTES - b0 ))) added"

say ""
say "TOTAL: $NFILES files, $(human "$BYTES")"
[ "$MISSING" -gt 0 ] && say "       $MISSING expected item(s) missing -- see NOT RUN above."
if [ "$PRUNED" -gt 0 ]; then
    say "       $PRUNED scored range(s) no longer on disk -- their DER stands, but"
    say "       those checkpoints are gone and the range cannot be re-scored."
fi
say ""
say "Extract locally: mkdir -p ~/diaper_4gpu && tar -xf $(basename "$OUT") -C ~/diaper_4gpu"
say "Merge into checkout: tar -xf $(basename "$OUT") -C experiments/10attractors --strip-components=1 experiments"

if [ "$DRY_RUN" = "1" ]; then
    printf '\n%s\n' "DRY_RUN=1 -- nothing written. Re-run without it to pack $(human "$BYTES")."
    exit 0
fi
if [ ! -s "$LIST_EXP" ]; then
    echo "ERROR: nothing found under $EXP_ROOT -- run this on the server, from the repo root." >&2
    [ "${FORCE:-0}" = "1" ] || exit 1
fi

printf '\n%s\n' "packing $NFILES files, $(human "$BYTES") -> $OUT"
rm -f "$OUT"
if [ -s "$LIST_EXP" ]; then
    tar -cf "$OUT" -C "$EXP_ROOT" --transform="s,^,experiments/," -T "$LIST_EXP"
else
    tar -cf "$OUT" --files-from=/dev/null
fi
if [ -s "$LIST_LOG" ]; then
    tar -rf "$OUT" -C "$LOG_PARENT" --transform="s,^,repo/logs/," -T "$LIST_LOG"
fi
tar -rf "$OUT" -C "$WORK/stage" repo
tar -rf "$OUT" -C "$WORK" MANIFEST.txt
if [ "$COMPRESS" = "1" ]; then gzip -f "$OUT"; OUT="${OUT}.gz"; fi
printf '%s\n' "done: $OUT ($(du -h "$OUT" | cut -f1))"

case "$OUT" in /*) OUT_ABS="$OUT" ;; *) OUT_ABS="$(pwd)/$OUT" ;; esac
printf '\n%s\n' "Copy it home:  scp <user>@<server>:${OUT_ABS} ."
