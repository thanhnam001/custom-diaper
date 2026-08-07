# MSDWild DER Gap Analysis

Investigation into why local MSDWild-finetuned checkpoints score notably higher
(worse) DER than the paper's published numbers, and what's actually driving
per-file errors. Conducted 2026-08-06 on the full MSDWild test set (490 files,
`database/msdwild/kaldi/test`), comparing two locally-available checkpoints:

- **original_model**: `SC_LibriSpeech_2spk_adapted1-10`, vanilla (self-attention,
  `latents2attractors: weighted_average`), epochs 66-76. Matches `results.csv`'s
  "Diaper (10 attractors) (vanila)" row (published-style collar DER 21.08%).
- **mlp_unmaskeddiv**: `SC_LibriSpeech_2spk_conformer_kernel31`, conformer
  kernel-31 + `latents2attractors: mlp` + unmasked diversity loss, epochs
  156-166. The best-scoring local config in `results.csv` (collar DER 19.17%).

## New tooling built for this investigation

`diaper/infer.py` previously only wrote RTTMs; there was no way to compute
diarization metrics without a separate `dscore` run, and no per-file
breakdown for weakness analysis. Added `--compute-metrics` (opt-in, default
off):

- Computes whole-file, frame-level DER/miss/FA/confusion, VAD/OSD error
  rates, and `attractor_accuracy` per recording, using the *same* decision
  process that produces the RTTM (existence gating + threshold + median
  filter) — so the numbers are informative about what dscore would score,
  just without dscore's forgiveness collar (expect these numbers to read
  higher than collar-based dscore DER for the same run).
- Writes `metrics_per_file.csv` (one row per recording — the artifact for
  weakness analysis) and `metrics_summary.txt` (macro-average across files,
  i.e. the *average of individuals*, not pooled-then-divided like the
  existing training-time dev metric) into the run's output directory.
- `attractor_accuracy` replicates `pit_loss_multispk`'s Hungarian-matched
  existence ground truth (via `get_exists_mask` in `infer.py`) so it's
  comparable to the training-time metric of the same name.

**Bug found and fixed along the way**: `calculate_metrics`
(`common_utils/metrics.py`) normalizes `VAD_FA`/`VAD_miss`/`OSD_FA`/`OSD_miss`
by a population (real speech/overlap frame count) disjoint from what their
false-alarm numerator counts. Guarded only by `epsilon=1e-6`, this is
harmless pooled across a training batch but blows up to values in the
hundreds of millions of percent per-file (e.g. any single false-positive
overlap frame in a file with zero real overlap — common, since most
MSDWild files aren't simultaneously multi-speaker throughout). Fixed via an
optional `return_denominators` return from `calculate_metrics` plus
NaN-marking + per-key-count-aware averaging in `infer.py`, so degenerate
per-file rates are excluded from the macro-average instead of corrupting it.

## Baseline results (median=11, threshold=0.5, existence-thr=0.5 — current defaults)

No-collar, macro-averaged over all 490 test files:

| Metric | original_model | mlp_unmaskeddiv (best) |
|---|---|---|
| DER | 36.49% | 35.21% |
| miss / FA / conf | 19.42 / 7.28 / 9.81 | 15.95 / 8.85 / 10.42 |
| VAD FA / miss | 5.73 / 11.18 | 6.31 / 8.97 |
| OSD FA / miss (n=470/490) | 30.24 / 83.17 | 40.26 / 69.35 |
| spk qty (ref/pred) | 1.01 / 0.88 | 1.01 / 0.93 |
| attractor accuracy | 95.27% | 95.55% |

Cross-checked against dscore (0.25s collar, the repo's MSDWild convention)
on the single longest test file (`02470`, 1320s): dscore gave 15.12% DER
vs our no-collar 25.00% for the same predictions — confirms the
no-collar/collar relationship is behaving as expected, not a bug.

mlp_unmaskeddiv wins on DER (as expected, matches the published ordering),
driven mainly by lower miss (15.95 vs 19.42) and better speaker-count
calibration, at the cost of slightly more FA/confusion and more OSD false
alarms.

## Question: does the paper's much longer training schedule (~516-525
epochs vs our 76/166) explain the gap, especially via an undertrained
attractor branch?

**No — checked directly against TensorBoard, not just reasoned about.**

We'd already established (prior session) that local finetune runs plateau
in far fewer epochs than the paper's schedule because of a *metric fix*:
`compute_loss_and_metrics` used to score dev-DER without existence-gating
(optimistic/noisy), which needed the paper's much longer schedule to show a
genuine plateau. The corrected (gated) metric detects convergence faster —
this alone isn't evidence of undertraining.

The sharper version of that question — does the *attractor branch
specifically* keep improving after early stopping fires on the gated
dev-DER, on a slower timescale the metric doesn't show? Checked directly:
`dev_attractor_existence_loss` in both local MSDWild-FT TensorBoard runs is
**flat-to-slightly-worse in the last 20% of training** (+0.55% / +1.80%
relative change) at the exact point `dev_DER` also plateaus. The attractor
branch converges in lockstep with everything else; early stopping isn't
cutting it off mid-improvement. `dev_DER_miss` is flat-to-rising in the
last 20% of training while `dev_DER_FA` keeps slowly improving — more
epochs of the current recipe would not have fixed the dominant miss error.

## Chasing the miss-dominant error: three inference-time levers, one small win

**1. Existence-gating threshold (`--estimate-spk-qty-thr`, default 0.5) — swept to 0.3 and 0.1.**

| model | thr | DER | miss | FA | conf | spk_qty(ref/pred) | attractor_acc |
|---|---|---|---|---|---|---|---|
| original_model | 0.5/0.3/0.1 | 36.49 (all three) | 19.42 (all three) | 7.28 (all three) | 9.81 (all three) | 1.01/0.88 (all three) | 95.27→94.76→92.31 |
| mlp_unmaskeddiv | 0.5/0.3/0.1 | 35.21 (all three) | 15.95 (all three) | 8.85 (all three) | 10.42 (all three) | 1.01/0.93 (all three) | 95.55→95.37→94.45 |

**Zero effect.** DER/miss/FA/conf/spk_qty are bit-for-bit identical across
all three thresholds for both models — the existence-probability head is
already sharply, confidently separated (real speakers well above 0.5,
non-speakers well below 0.1). `attractor_accuracy` even *drops slightly* at
lower thresholds (marginal true-negatives flipping to false-positive-active).
Miss is not coming from whole speakers/attractor-slots being wrongly gated
off.

**2. Per-frame activation threshold (`--threshold`, default 0.5) — swept to 0.3 and 0.2.**

| model | thr | DER | miss | FA | conf | spk_qty(ref/pred) |
|---|---|---|---|---|---|---|
| original_model | 0.5 | 36.49 | 19.42 | 7.28 | 9.81 | 1.01/0.88 |
| original_model | 0.3 | 41.61 | 8.77 | 22.15 | 10.69 | 1.01/1.13 |
| original_model | 0.2 | 53.19 | 5.04 | 38.92 | 9.23 | 1.01/1.34 |
| mlp_unmaskeddiv | 0.5 | 35.21 | 15.95 | 8.85 | 10.42 | 1.01/0.93 |
| mlp_unmaskeddiv | 0.3 | 41.07 | 6.29 | 24.33 | 10.42 | 1.01/1.18 |
| mlp_unmaskeddiv | 0.2 | 51.12 | 3.71 | 38.89 | 8.52 | 1.01/1.35 |

Opposite pattern: this one moves a lot. Miss drops sharply, but FA more
than compensates and **DER gets worse at every step**, overshooting into
over-prediction. Confirms the frame-activation branch (not the existence
branch) is where the under-prediction signal lives, but the default 0.5 is
already close to a local DER-optimum on this axis — a real tradeoff, not
free miscalibration.

**3. Median filter window (`--median-window-length`, default 11) — swept to 1 and 5.**

| model | median | DER | miss | FA | conf | pred_spk |
|---|---|---|---|---|---|---|
| original_model | 11 | 36.49 | 19.42 | 7.28 | 9.81 | 0.88 |
| original_model | 5 | **35.76** | 18.69 | 6.69 | 10.38 | 0.88 |
| original_model | 1 | 37.88 | 19.12 | 7.34 | 11.44 | 0.88 |
| mlp_unmaskeddiv | 11 | 35.21 | 15.95 | 8.85 | 10.42 | 0.93 |
| mlp_unmaskeddiv | 5 | **34.28** | 15.77 | 8.14 | 10.39 | 0.92 |
| mlp_unmaskeddiv | 1 | 34.32 | 15.87 | 8.06 | 10.39 | 0.92 |

Small but real win: `median=5` beats `median=11` for both models (-0.73 /
-0.93 DER points), via FA/confusion, not miss (miss barely moves, <1
point). `median=1` overshoots for original_model (confusion spikes from
fragmented turn flicker). **Worth adopting `median=5`** as a free win, but
it doesn't touch the core miss gap.

**All three cheap inference-time levers are now exhausted.**

## Root cause found: DER scales sharply with the number of distinct speakers

Computed distinct-speaker-count per recording directly from
`database/msdwild/kaldi/test/rttm` (not `avg_ref_spk_qty`, which measures
concurrent-overlap density, not speaker identity count):

| n distinct ref speakers | count (test set) | mean DER_miss | mean DER |
|---|---|---|---|
| 2 | 274 (56%) | ~11-13 | ~25-26% |
| 3 | 138 (28%) | ~20-24 | ~44-45% |
| 4 | 78 (16%) | ~28-34 | ~56% |

`corr(n_speakers, DER_miss)` = 0.50-0.56, `corr(n_speakers, DER)` = 0.57,
`corr(n_speakers, DER_FA)` ≈ 0 (-0.08 to +0.02) for both models — this is a
pure miss effect, not general noise. On 2-speaker files both models are
already close to the paper's ballpark; on 4-speaker files DER roughly
doubles. The worst-miss files pulled by hand (`00250`, `00804`, `01763`,
`02650`, appearing in both models' worst-15 lists) all had 3-4 distinct
reference speakers and captured roughly half the reference total speech
duration; near-median-miss files (`00208`, `00809`) had predicted duration
closely matching reference despite similar nominal speaker counts.

## Ruled out: training-data imbalance, checked at both the recording level and the actual per-chunk level

Hypothesis: maybe 3-4-speaker examples are underrepresented somewhere in
the training pipeline. Checked every stage:

**MSDWild finetune train set** (`database/msdwild/kaldi/train/rttm`, 2,476
files): 60.5% 2-spk / 25.8% 3-spk / 13.7% 4-spk — nearly identical to the
test set's 56%/28%/16%. Proportionate exposure (977/2476 files, ~40%, have
3-4 speakers); not underrepresented.

**Adapt stage** (`adapted1-10`, `300h_maximum10spks`,
`E:/datasets/v1_300hours_max10spks`), recording-level roster: 25.5% 2-spk
down to 3.5% 10-spk, majority 3+ speakers — looked fine at first glance.

**Adapt stage, actual per-training-chunk level** (the number that matters —
loaded all 4,712 precomputed chunks directly from
`E:/datasets/diaper_precompute_300h_maximum_10spks_24000frames/train/*.pkl`,
each ~240s, and counted simultaneously-active speaker columns in each
chunk's label matrix `T`): only **13.4%** of chunks have just 2 active
speakers — the remaining **86.6%** have 3-10, roughly evenly spread. The
adapt stage massively *over*-represents multi-speaker scenes relative to
both the finetune data and the test distribution — the opposite of the
hypothesized bias.

**Pretrain stage** (`SC_LibriSpeech_2spk_pretrain500h`,
`E:/datasets/v1_500hours`, 1,386 recordings): confirmed **100% 2-speaker**
by construction — a real, strong initial bias, but one the subsequent
adapt stage clearly works hard to counteract (86.6% multi-speaker chunk
exposure).

**Conclusion: no training stage under-exposes the model to multi-speaker
scenarios.** The speaker-count-vs-DER gradient is therefore most likely
**intrinsic problem difficulty** (more speakers → more possible confusions
and overlap combinations, a pattern well-documented across diarization
systems generally), not a fixable data-curriculum artifact.

## Cross-check against the paper's own shipped checkpoint (not just our reproductions)

Everything above compared two *local* reproductions against each other. To
find out how much of the remaining gap is real vs. an artifact of our own
training recipe, we ran `diaper/infer.py --compute-metrics` against the
paper's own shipped checkpoint (`../Master/repos/DiaPer/models/10attractors/
SC_LibriSpeech_2spk_adapted1-10_finetuneMSDWILD`, epochs 515-525, i.e.
averaging checkpoints 516-525 — all 10 shipped), on the same 490-file test
set with the same postprocessing (median=11, subsampling=10, threshold=0.5).
Config: `models/10attractors/paper_original_checkpoints/infer_msdwild.yaml`.

**Validated against dscore first**: pooled DER = 15.47% vs. the paper's own
published 15.46% (`../Master/repos/DiaPer/results/DiaPer/10attractors/
MSDWild/withFT/few.val/result_collar0.25`, collar 0.25) — essentially an
exact match. Confirms our `infer.py` + dscore pipeline correctly reproduces
the paper's official result end-to-end when fed their checkpoint.

### Why `--compute-metrics`'s own macro DER (31.02%) reads so much higher than dscore's pooled 15.47%

Investigated because the gap is much larger than expected. Two independent
components, both confirmed:

1. **Macro vs. duration-weighted aggregation (~5 points, expected/benign).**
   Even using dscore's *own* per-file DER numbers: unweighted (macro) average
   = 20.58%, duration-weighted average ≈ 17.36%, true pooled OVERALL =
   15.47%. MSDWild file durations are heavily right-skewed (median 39.4s,
   mean 72.4s) — short files count equally in a macro average but get
   diluted by duration in the pooled number.

2. **`calculate_metrics` has no forgiveness collar at all (~10 more points,
   the dominant factor).** Confirmed by reading `common_utils/metrics.py` —
   there is no collar logic anywhere in it; every frame is scored, including
   the frames immediately around every reference speaker-turn boundary that
   dscore's 0.25s collar would forgive entirely. This makes our own DER a
   strictly harsher metric definition than the one used for the published
   numbers, and it hits short, turn-dense files hardest. Worst example:
   file `00066` (25.2s, **25 reference speaker turns** — a turn roughly
   every second, many segments <0.5s) scores 34.0% on our no-collar metric
   vs. 3.08% on dscore. At `median_window_length=11`/`subsampling=10` (100ms
   frames, ~1.1s smoothing — tuned to work *with* the 0.25s collar per the
   paper's recipe), the model architecturally cannot resolve turns finer
   than about a second; dscore's collar exists precisely to forgive that,
   our raw frame metric doesn't. Dropping the 50 worst-by-our-DER files only
   moves the macro average from 31.02%→26.88%, confirming this is a
   systematic bias across most short files, not a few pathological outliers.

**Implication (superseded below)**: `--compute-metrics` output is
self-consistent for comparing our own checkpoints against each other (its
intended use), but was **not directly comparable in absolute terms** to
dscore/paper-published DER — it read systematically higher, especially on
short/turn-dense files.

### Fix: collar support added to `calculate_metrics`/`--compute-metrics`

Rather than leaving that as a standing caveat, added an optional forgiveness
collar so `--compute-metrics` can be made directly comparable to dscore when
needed, reusable in future sessions:

- `diaper/common_utils/metrics.py::calculate_metrics` — new `collar_frames`
  parameter (default `0`, fully backward-compatible: `train.py`'s existing
  call is untouched). Per sequence, detects reference speaker-turn
  boundaries (any frame where the reference activity row changes vs. the
  previous frame, plus frame 0) and excludes frames within `collar_frames`
  of any boundary from both the error numerators and the scored-frame
  denominators — the same idea as dscore/md-eval.pl's collar, implemented
  as a `max_pool1d` dilation over a boundary indicator (vectorized, and
  edge-clipped for free via the implicit padding). Guards the degenerate
  case where an entire sequence gets collared away (would otherwise crash
  on `torch.round` receiving a plain Python float instead of a tensor).
- `diaper/infer.py` — new `--collar` CLI flag (seconds, default `0.0`).
  `compute_file_metrics` converts seconds → frames using the model's native
  (subsampled) frame period (`subsampling * frame_shift / sampling_rate`)
  before calling `calculate_metrics`. `metrics_summary.txt` now records
  which collar was used on its summary line. Help text warns that changing
  only `--collar` against an `--rttms-dir` that already has RTTMs from a
  prior run will skip inference for every file (RTTMs themselves don't
  depend on collar) — point at a fresh `--rttms-dir` to get a complete
  re-scored `metrics_per_file.csv`.

**Validated end-to-end** by rerunning the paper's MSDWild checkpoint
(490 files) with `--collar 0.25`:
- Macro DER: **31.02% → 20.59%**, landing almost exactly on dscore's own
  macro-of-per-file number (20.58%) computed earlier in this section —
  essentially a match.
- File `00066` (the worst no-collar outlier: 25 speaker turns in 25.2s):
  **34.0% → 2.0%**, close to dscore's 3.08% for the same file.

It's a frame-level approximation of md-eval.pl's exact algorithm, not
bit-identical, but close enough to trust for fast local iteration. **Use
`--collar 0.25` for MSDWild, leave it at the default `0.0` for RAMC**,
matching each dataset's dscore convention (see "Evaluation" in
`CLAUDE.md`). The numbers in the rest of this document (all no-collar,
predating this fix) remain valid for *relative* comparison as already
established, and don't need to be recomputed unless absolute comparability
to dscore is specifically needed going forward.

### Three-way comparison (paper's official checkpoint vs. our two local reproductions)

Same `--compute-metrics` basis for all three (so the *absolute* numbers all
read high per the caveat above, but are comparable to each other):

| Checkpoint | DER | miss / FA / conf | attractor_acc | spk_qty (ref/pred) |
|---|---|---|---|---|
| **Paper's official** (adapted1-10→finetuneMSDWILD) | **31.02%** | 13.16 / 9.48 / 8.38 | 95.06% | 1.01 / 0.96 |
| Our `original_model` (vanilla recipe repro) | 36.49% | 19.42 / 7.28 / 9.81 | 95.27% | 1.01 / 0.88 |
| Our `mlp_unmaskeddiv` (current best) | 35.21% | 15.95 / 8.85 / 10.42 | 95.55% | 1.01 / 0.93 |

The relative gap here (paper 31.02% vs. our best 35.21%, ~4.2 pts) tracks
the dscore-validated relative gap (15.46% vs. ~19.17%, ~3.7 pts) closely —
good confirmation that `--compute-metrics`, despite the absolute mismatch
above, is trustworthy for *relative* checkpoint comparison, which is what
all the sweeps earlier in this doc rely on.

### Sharpened root cause: the gap to the paper is concentrated in 3-4-speaker files

Bucketing all three checkpoints' per-file results by reference speaker count
(same buckets as the "Root cause found" section above):

| n_spk | paper miss / DER | our best (mlp_unmaskeddiv) miss / DER | miss gap |
|---|---|---|---|
| 2 (274 files) | 9.14 / 22.72 | 10.22 / 24.70 | +1.08 |
| 3 (138 files) | 16.18 / 38.35 | 20.46 / 44.43 | +4.28 |
| 4 (78 files) | 21.95 / 47.21 | 28.08 / 55.85 | +6.13 |

All three checkpoints (including the paper's own) show the same underlying
speaker-count-vs-miss correlation (r=0.48-0.57) — confirming that scaling
itself is task-intrinsic/architecture-independent, not a defect specific to
our reproduction (consistent with the "Root cause found" section above).
**But our degradation curve is steeper than the paper's**: the gap is nearly
closed at 2 speakers (+1.08 miss) and widens substantially by 4 speakers
(+6.13 miss). This tracks with speaker-count *detection* specifically, not
just general frame noise: the paper's checkpoint under-predicts speaker
count on 193/490 files (39.4%), vs. our best at 218/490 (44.5%) and vanilla
at 303/490 (61.8%) — `mlp_unmaskeddiv` already closed most of the
under-prediction gap over vanilla, but not all of it, and the residual
tracks exactly where the remaining DER gap lives (3-4-speaker files).

This directly sharpens the "Not yet tried" actionables below: oversampling
3-4-speaker finetune examples is now the best-evidenced lever, since we can
see precisely where and how (speaker-count under-prediction on multi-speaker
files) the paper's checkpoint pulls ahead — not just infer generically from
the correlation.

## Where this leaves things

Established, don't re-chase without new evidence:
- Fewer local FT epochs than the paper ≠ undertraining (metric-fix artifact).
- Attractor branch is not cut off early; converges in lockstep with everything else.
- Existence-gating threshold: no effect on DER at any value tested.
- Frame-activation threshold: real tradeoff, but default 0.5 is near-optimal.
- Data-curriculum imbalance across all three training stages: ruled out, checked at the chunk level where it matters.
- **Confirmed against the paper's own shipped checkpoint, not just our reproductions** — dscore validates our pipeline reproduces their published number almost exactly (15.47% vs. 15.46%), and the relative gap tracks between the dscore view and our own `--compute-metrics` view.
- `--compute-metrics` used to have no forgiveness collar, so its numbers weren't directly comparable in absolute terms to dscore (read systematically higher, worst on short/turn-dense files); still trustworthy for *relative* comparison between our own checkpoints, which is all the sweeps above rely on. **Fixed**: `--collar` (seconds) is now a supported flag — `--collar 0.25` on the paper's MSDWild checkpoint reproduced dscore's macro number almost exactly (20.59% vs. 20.58%). All the tables in this document predate the fix and are no-collar (still valid for relative comparison); rerun with `--collar 0.25` (MSDWild) if absolute comparability to dscore is needed for future work.
- The gap to the paper is **not uniform across the test set** — it's concentrated in 3-4-speaker files (miss gap +1.08 at 2 spk vs. +6.13 at 4 spk) and correlates with residual speaker-count under-prediction (44.5% of our files under-predict vs. 39.4% for the paper's checkpoint).

Actionable:
- **Adopt `--median-window-length 5`** instead of 11 for MSDWild inference — small free DER win (~0.7-0.9 points), no retraining needed.
- **Best-evidenced next step**: oversample/upweight 3-4-speaker MSDWild finetune examples beyond their natural proportion. Previously "not yet tried, temper expectations" — now better-evidenced: the paper's own checkpoint achieves closer speaker-count calibration and dramatically smaller miss on exactly the 3-4-speaker subset where we lag, so the lever is targeted at the right axis (speaker-count detection precision), not just general miss noise.
- **Not yet tried**: class-imbalance-aware loss weighting (`vad_loss_weight`/`attractor_existence_loss_weight`), reframed toward speaker-count difficulty rather than raw speech/silence balance.
- If neither of those moves the needle, this may simply be the honest difficulty ceiling of the current architecture/recipe on >2-speaker real conversations — worth accepting and reporting as a characterized limitation.

## Artifacts

- Code: `diaper/infer.py` (`--compute-metrics`, `--collar`, `compute_file_metrics`, `get_exists_mask`, `get_active_attractor_mask`), `diaper/common_utils/metrics.py` (`return_denominators`, `collar_frames`).
- Full-test-set results: `results/full_msdwild_test/{original_model,mlp_unmaskeddiv}/.../metrics_per_file.csv` + `metrics_summary.txt`.
- Threshold/median sweeps: `results/full_msdwild_test_thrsweep/`, `results/full_msdwild_test_framethrsweep/`, `results/full_msdwild_test_mediansweep/`.
- Paper-checkpoint cross-check: config `models/10attractors/paper_original_checkpoints/infer_msdwild.yaml`
  (RAMC counterpart also written: `infer_ramc.yaml`, but that run hit a
  hardware ceiling — see note below), results
  `results/paper_original_checkpoints/msdwild_test/.../metrics_per_file.csv`
  + `metrics_summary.txt`, dscore output cross-checked manually (not saved
  to a file — pooled DER 15.47%).
- Collar-feature validation run:
  `results/paper_original_checkpoints/msdwild_test_collar025/.../metrics_per_file.csv`
  + `metrics_summary.txt` (`--collar 0.25` rerun of the same paper checkpoint
  — macro DER 20.59%, matches dscore's own macro-of-per-file 20.58%).
- Note for future RAMC runs: the paper's plain self-attention architecture
  (not our conformer reproduction) at RAMC's prescribed `subsampling: 5`,
  whole-recording inference (`num_frames: -1`, no chunking — required to keep
  attractor/speaker identity consistent across a recording) needs a
  self-attention matrix over ~30k frame tokens for a 30-minute recording —
  OOM'd on this machine (~25GB RAM, tried to allocate 22GB for one file).
  Our conformer-based reproduction doesn't hit this ceiling. Unresolved as of
  this writing.
- Persistent memory of this investigation (for future sessions): `diaper-old-vs-new-der-metric` memory record.
