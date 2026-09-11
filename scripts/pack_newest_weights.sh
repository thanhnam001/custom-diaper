#!/bin/bash
# Tar the newest checkpoint(s) of every run in the 36h resume queue and/or the
# pos_weight queue, to bring the trained weights back off the server.
#
#   DRY_RUN=1 ./scripts/pack_newest_weights.sh      # size it first (do this)
#   ./scripts/pack_newest_weights.sh                # both queues, KEEP=10
#   QUEUE=posw KEEP=1 ./scripts/pack_newest_weights.sh
#
# Run it ON THE SERVER, from the repo root. This is the counterpart to
# scripts/pack_resume_weights_for_36h_queue.sh (which pushes weights TO the
# server); this one pulls the results back.
#
#
# HOW MANY CHECKPOINTS -- KEEP defaults to 10, ON PURPOSE
# --------------------------------------------------------
# Every DER number in this project comes from averaging the last 10
# checkpoints (infer.py + MAX_CHECKPOINTS_TO_AVERAGE=10 in both queue
# scripts). A KEEP=1 archive therefore CANNOT reproduce any number that is
# comparable to results.csv -- scoring a single checkpoint is a different
# protocol. So:
#
#   KEEP=10  (default)  you want to score these locally / keep them.  ~800 MB
#                       per run.
#   KEEP=1              you only want to RESUME training later. train.py
#                       sorts output_path/models by mtime and loads only the
#                       single latest one, so 1 is genuinely enough for that
#                       (see pack_resume_weights_for_36h_queue.sh's header).
#
# dscore logs and the MANIFEST are always included -- they are a few KB and
# are what tells you which epoch each number came from.
#
#
# WHICH RUNS GET PACKED
# ----------------------
# Output paths are read from each lineage's own yaml, not hardcoded, so this
# follows the configs if the server root moves.
#
#   QUEUE=36h    exactly the three runs run_36h_resume_queue.sh actually
#                runs: A5 (conformer RAMC @ sub5), A1 (same lineage,
#                MSDWild), B2 (fixed-Noam ebf RAMC @ sub5). That queue was
#                narrowed to these on 2026-09-10; A2/A3/A4/B1/B3/B4 were
#                dropped but their directories still exist from earlier
#                rounds, so this does NOT glob for variants -- doing so
#                would pack 13 runs instead of 3.
#   QUEUE=posw   the paperlr-ebranchformer MSDWild *_posw* arms, found by
#                globbing so it picks up whatever POSW_A/POSW_B you ran.
#   QUEUE=all    both (default)
#
#   ALL_VARIANTS=1  with QUEUE=36h|all, pack every finetune variant under
#                   those lineages instead of the narrowed three -- the
#                   historical superset, for a final bring-everything-home.
#
# A run that has not started yet is reported NOT RUN and skipped, not fatal
# -- the pos_weight queue may legitimately not have produced anything when
# you first run this.

set -u

QUEUE="${QUEUE:-all}"
KEEP="${KEEP:-10}"
OUT="${OUT:-diaper_newest_weights.tar}"
COMPRESS="${COMPRESS:-0}"
DRY_RUN="${DRY_RUN:-0}"

case "$QUEUE" in
    36h|posw|all) ;;
    *) echo "ERROR: QUEUE must be 36h, posw or all (got '$QUEUE')" >&2; exit 1 ;;
esac

CNF_DIR="${CNF_DIR:-models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_conformer_k31}"
EBF_DIR="${EBF_DIR:-models/10attractors/SC_LibriSpeech_2spk_2500h_fixednoam_ebf}"
PLR_DIR="${PLR_DIR:-models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr}"
PLE_DIR="${PLE_DIR:-models/10attractors/SC_LibriSpeech_2spk_2500h_paperlr_ebranchformer}"

yaml_get () { grep "^$1:" "$2" | head -1 | sed "s|^$1: *||"; }

_probe=$(yaml_get output_path "$PLE_DIR/finetune_msdwild_10spks.yaml" 2>/dev/null)
if [ -z "$_probe" ]; then
    echo "ERROR: could not read output_path from $PLE_DIR/finetune_msdwild_10spks.yaml." >&2
    echo "       Run from the repo root, or set PLE_DIR." >&2
    exit 1
fi
# output_path is .../experiments/10attractors/<run>/models_finetuneXXX
EXP_ROOT="${EXP_ROOT:-$(dirname "$(dirname "$_probe")")}"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
LIST="$WORK/list"; MANIFEST="$WORK/MANIFEST.txt"
: > "$LIST"; : > "$MANIFEST"

BYTES=0; NFILES=0; NRUNS=0; NMISSING=0

say () { printf '%s\n' "$*"; printf '%s\n' "$*" >> "$MANIFEST"; }
human () { awk -v b="$1" 'BEGIN{
    if (b>=1073741824) printf "%.1f GB", b/1073741824;
    else if (b>=1048576) printf "%.0f MB", b/1048576;
    else printf "%.0f KB", b/1024;}'; }

# add_path <absolute-file> -> append to tar list, relative to EXP_ROOT
# Returns 0 if added. NOTE: never call this via $( ) -- it mutates BYTES and
# NFILES, and command substitution would run it in a subshell and silently
# drop those (which made DRY_RUN under-report the archive size by ~1000x).
add_path () {
    local f="$1" sz
    case "$f" in
        "$EXP_ROOT"/*) ;;
        *) echo "  WARN outside EXP_ROOT, skipped: $f" >&2; return 1 ;;
    esac
    printf '%s\n' "${f#"$EXP_ROOT"/}" >> "$LIST"
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    BYTES=$(( BYTES + sz )); NFILES=$(( NFILES + 1 ))
    return 0
}

# out_of <cfg> -> that config's output_path, rebased onto EXP_ROOT.
# Rebasing by the last two components, <run>/models_finetuneXXX, is a no-op
# on the server but lets you point EXP_ROOT at a local checkout and DRY_RUN
# the whole thing against real directories before trusting it.
out_of () {
    local abs
    abs=$(yaml_get output_path "$1" 2>/dev/null) || return 1
    [ -n "$abs" ] || return 1
    printf '%s/%s/%s\n' "$EXP_ROOT" \
        "$(basename "$(dirname "$abs")")" "$(basename "$abs")"
}

# pack_run <run-dir> [queue-label]
# Packs the newest $KEEP checkpoints plus every dscore log under the run's
# *_test_pred directories.
pack_run () {
    local dir="$1" tag="${2:-}" label="${1#"$EXP_ROOT"/}"
    local b0=$BYTES n_ck=0 n_log=0 sz
    [ -n "$tag" ] && label="$tag  $label"

    if [ ! -d "$dir/models" ]; then
        say "  NOT RUN   $label  (no models/ dir)"
        NMISSING=$(( NMISSING + 1 )); return 0
    fi

    mapfile -t eps < <(find "$dir/models" -maxdepth 1 -name 'checkpoint_*.tar' \
        -printf '%f\n' 2>/dev/null \
        | sed -E 's/checkpoint_([0-9]+)\.tar/\1/' | sort -n)
    if [ "${#eps[@]}" -eq 0 ]; then
        say "  EMPTY     $label  (models/ exists but holds no checkpoint_*.tar)"
        NMISSING=$(( NMISSING + 1 )); return 0
    fi

    mapfile -t picked < <(printf '%s\n' "${eps[@]}" | tail -n "$KEEP")
    local e
    for e in "${picked[@]}"; do
        add_path "$dir/models/checkpoint_${e}.tar" && n_ck=$(( n_ck + 1 ))
    done

    local p f
    for p in "$dir"/*_test_pred; do
        [ -d "$p" ] || continue
        while IFS= read -r f; do
            add_path "$f" && n_log=$(( n_log + 1 ))
        done < <(find "$p" -maxdepth 1 -name 'dscore*.log' -type f 2>/dev/null)
    done

    NRUNS=$(( NRUNS + 1 ))
    say "  ok        $label"
    say "            epochs on disk ${eps[0]}..${eps[-1]} (${#eps[@]}), packing $n_ck newest (ep ${picked[0]}..${picked[-1]}), $n_log dscore log(s), $(human $(( BYTES - b0 )))"

    # Surface the scores we already have, so the manifest says what each
    # archived run was actually worth without unpacking anything.
    local ov
    while IFS= read -r ov; do say "            $ov"; done < <(
        grep -h -- '\*\*\* OVERALL \*\*\*' "$dir"/*_test_pred/dscore*.log 2>/dev/null \
        | awk '{printf "DER %s  JER %s\n", $4, $5}' | sort -u)
}

# scan_config <cfg> <mode>
#   mode base  -> the config's own output_path plus every *_suffix variant,
#                 excluding *_posw* (those belong to the pos_weight queue)
#   mode posw  -> only the *_posw* variants
scan_config () {
    local cfg="$1" mode="$2" base d
    [ -f "$cfg" ] || { say "  (no config: $cfg)"; return 0; }
    base=$(out_of "$cfg") || { say "  (no output_path in $cfg)"; return 0; }

    if [ "$mode" = "posw" ]; then
        local found=0
        for d in "${base}"_posw*; do
            [ -d "$d" ] || continue
            pack_run "$d" "posw"; found=1
        done
        [ "$found" -eq 1 ] || say "  NOT RUN   ${base#"$EXP_ROOT"/}_posw*  (queue has not produced anything yet)"
        return 0
    fi

    [ -d "$base" ] && pack_run "$base"
    for d in "${base}"_*; do
        [ -d "$d" ] || continue
        case "$d" in *_posw*) continue ;; esac
        pack_run "$d"
    done
}

# ---------------------------------------------------------------------------
# scan_36h -- mirrors run_36h_resume_queue.sh's lane definitions EXACTLY.
#
# That queue was narrowed on 2026-09-10 to three runs: lane A does A5 (the
# conformer RAMC finetune at subsampling 5) then A1 (the same lineage's
# MSDWild finetune), and lane B does B2 (the fixed-Noam E-Branchformer RAMC
# finetune at subsampling 5). A2/A3/A4/B1/B3/B4 were dropped -- they are all
# above the paper's ~21 RAMC reference, or out of headroom.
#
# Those six dropped runs still have directories on disk from earlier rounds,
# so globbing <base>_* would pack 13 runs instead of 3 and quadruple the
# archive with weights this queue never touched. Set ALL_VARIANTS=1 when you
# deliberately want that historical superset.
# ---------------------------------------------------------------------------
scan_36h () {
    local cnf_ramc cnf_msd ebf_ramc

    if [ "${ALL_VARIANTS:-0}" = "1" ]; then
        say "  (ALL_VARIANTS=1 -- every finetune variant under these lineages,"
        say "   not just the three the narrowed queue runs)"
        scan_config "$CNF_DIR/finetune_ramc_10spks.yaml"    base
        scan_config "$CNF_DIR/finetune_msdwild_10spks.yaml" base
        scan_config "$EBF_DIR/finetune_ramc_10spks.yaml"    base
        scan_config "$EBF_DIR/finetune_msdwild_10spks.yaml" base
        scan_config "$PLR_DIR/finetune_msdwild_10spks.yaml" base
        scan_config "$PLE_DIR/finetune_ramc_10spks.yaml"    base
        return 0
    fi

    cnf_ramc=$(out_of "$CNF_DIR/finetune_ramc_10spks.yaml") || cnf_ramc=""
    cnf_msd=$(out_of "$CNF_DIR/finetune_msdwild_10spks.yaml") || cnf_msd=""
    ebf_ramc=$(out_of "$EBF_DIR/finetune_ramc_10spks.yaml") || ebf_ramc=""

    if [ -n "$cnf_ramc" ]; then pack_run "${cnf_ramc}_sub5" "A5"
    else say "  (no config: $CNF_DIR/finetune_ramc_10spks.yaml)"; fi
    if [ -n "$cnf_msd" ]; then pack_run "$cnf_msd" "A1"
    else say "  (no config: $CNF_DIR/finetune_msdwild_10spks.yaml)"; fi
    if [ -n "$ebf_ramc" ]; then pack_run "${ebf_ramc}_sub5" "B2"
    else say "  (no config: $EBF_DIR/finetune_ramc_10spks.yaml)"; fi
}

say "DiaPer newest-weights pack"
say "  packed   : $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname 2>/dev/null || echo unknown-host)"
say "  queue    : $QUEUE      KEEP=$KEEP checkpoint(s) per run"
say "  exp root : $EXP_ROOT"
say "  repo     : $(pwd)"
say "  commit   : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
say ""

if [ "$QUEUE" = "36h" ] || [ "$QUEUE" = "all" ]; then
    say "36h RESUME QUEUE (A5, A1, B2 -- the narrowed lane set)"
    scan_36h
    say ""
fi

if [ "$QUEUE" = "posw" ] || [ "$QUEUE" = "all" ]; then
    say "POS_WEIGHT QUEUE"
    scan_config "$PLE_DIR/finetune_msdwild_10spks.yaml" posw
    say ""
fi

say "TOTAL: $NRUNS run(s), $NFILES files, $(human "$BYTES")"
[ "$NMISSING" -gt 0 ] && say "       $NMISSING run(s) skipped -- see NOT RUN / EMPTY above."
say ""
say "Extract at home, from the repo root:"
say "  tar -xf $(basename "$OUT") -C experiments/10attractors"

if [ "$DRY_RUN" = "1" ]; then
    printf '\n%s\n' "DRY_RUN=1 -- nothing written. Re-run without it to pack $(human "$BYTES")."
    exit 0
fi
if [ ! -s "$LIST" ]; then
    echo "ERROR: nothing to pack -- no run produced checkpoints. Run this on the" >&2
    echo "       server, from the repo root, after a queue has started." >&2
    exit 1
fi

printf '\n%s\n' "packing $NFILES files, $(human "$BYTES") -> $OUT"
rm -f "$OUT"
tar -cf "$OUT" -C "$EXP_ROOT" -T "$LIST"
tar -rf "$OUT" -C "$WORK" MANIFEST.txt
if [ "$COMPRESS" = "1" ]; then gzip -f "$OUT"; OUT="${OUT}.gz"; fi
printf '%s\n' "done: $OUT ($(du -h "$OUT" | cut -f1))"

case "$OUT" in /*) OUT_ABS="$OUT" ;; *) OUT_ABS="$(pwd)/$OUT" ;; esac
printf '\n%s\n' "Copy it home:  scp <user>@<server>:${OUT_ABS} ."
