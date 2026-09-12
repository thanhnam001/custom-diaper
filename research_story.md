# What actually improves end-to-end neural diarization with Perceiver attractors

Research narrative for this codebase, in the structure of
`private/research_plan.txt`: survey → what is wrong with current approaches →
which gap → hypothesis → experiment → proof → ablation.

**Status of each claim is marked explicitly.** `PROVEN` = measured, with the
confound named and controlled. `PENDING` = the experiment is designed and
queued but has not run. `FAILED` = tested and rejected; reported because a
failed hypothesis is a result.

---

## 0. Where we stand

| benchmark | DiaPer published | ours | |
|---|---|---|---|
| **MSDWild** (collar 0.25, 490 files) | 15.47 | **15.49** | parity |
| **RAMC** (collar 0, 43 files) | 21.1 | **16.15** | **−4.95** |

MSDWild 15.49 is the conformer-k31 arm finetuned at lr 1e-5, epochs 535–545,
and it was **still improving when it was stopped** (dev miss 25.89 → 22.26 and
dev `avg_pred_spk_qty` 0.876 → 0.913 were both still moving monotonically), so
it is a lower bound on that configuration. RAMC 16.15 is the same architecture
finetuned at subsampling 5, epochs 490–500, which hit its cap still improving.

The asymmetry is the story: **we match the published result on one corpus and
beat it by ~5 DER on the other**, and the reasons are known and different in
each case.

---

## 1. Survey

Speaker diarization moved from clustering speaker embeddings to end-to-end
neural models: EEND (permutation-invariant training over a fixed speaker
count) → EEND-EDA (an LSTM encoder–decoder produces *attractors*, one per
speaker, so the speaker count becomes flexible) → **DiaPer**, which replaces
that LSTM with Perceiver cross-attention over a set of learned latents.

DiaPer has exactly two functional components:

1. a **frame encoder** — a stack of plain self-attention + feed-forward blocks
   turning spliced log-mel frames into frame embeddings;
2. an **attractor branch** — learned latents cross-attend to those frame
   embeddings through Perceiver blocks, and the resulting latents are mapped
   to attractors by a softmax **convex combination** (`weighted_average`),
   regularised by an entropy term `Le`.

Per-frame speaker activity is a dot product of frame embeddings against
attractors, trained with PIT-matched BCE. Attractors *are* the model's
representation of "who is in this recording".

---

## 2. What is wrong with this design

Two representational bottlenecks. Both were found the way
`private/research_plan.txt` prescribes — *by analogy to neighbouring problems,
and by observing behaviour.*

### 2a. The frame encoder has no local inductive bias

Every neighbouring speech task — ASR, self-supervised speech representation —
moved off plain transformers to convolution-augmented encoders (Conformer,
E-Branchformer), because speech structure is local as well as global. DiaPer
kept a plain self-attention stack.

*Observed behaviour.* Best pretrain-stage dev DER, same data, same budget:

| frame encoder | pretrain dev DER |
|---|---|
| E-Branchformer (mlp) | **2.60** |
| conformer k31 (mlp) | 2.80 |
| plain self-attention (LR-fixed) | 3.75 |
| plain self-attention (old LR) | 10.00 |

The conformer family fits the task 4× better than plain self-attention did
before the optimizer was fixed, and still better after.

### 2b. The attractor branch cannot represent speaker identity well

The latents→attractors map is constrained to a **convex combination**:
attractors must be weighted averages of the latents, with weights summing to
one. And the only regulariser, `Le`, acts on *those weights* — not on the
attractors themselves. **Nothing in the objective makes two attractors
different from each other.**

*Observed behaviour — this is the "the relation between feature and label is
poorly represented" symptom:*

- **Attractors collapse.** On RAMC, 2–8 of 43 files exceed 10× predicted
  speaker-duration imbalance, and up to **6 files emit fewer than two
  speakers at all** — each scoring ~40 % confusion. Six files recur across
  every condition (390, 77, 123, 151, 144, 431), and *which one is currently
  collapsed rotates between training rounds.*
- **The residual error is confusion, not detection.** Our best MSDWild system
  now beats the reference on miss (−0.84) and false alarm (−0.21). The entire
  remaining deficit is **confusion (+0.94)** — *which* speaker, not *whether*
  speech.
- **Local confusion is an invariant floor.** Re-choosing the permutation
  optimally per 30 s window leaves confusion at **5.8–6.3 %** in every
  condition measured: every encoder, both resolutions, both learning rates.
  Nothing tried has moved it. An invariant across all architectures is what a
  *representation or objective* limit looks like — not a tuning one.

### 2c. Bonus problem: the field cannot measure any of this

Worth stating because it invalidates part of our own earlier work.

- **Pooled MSDWild DER cannot resolve <1.5 DER between architectures** (paired
  bootstrap, 10–20k resamples over test files), and **seed alone moves
  0.25 DER**. Same-lineage comparisons (one run, two checkpoints) do resolve
  to ~0.3, because per-file disagreement is small — so the noise floor is a
  property of *how much two systems disagree per file*, not of the benchmark.
- **Pooled and macro DER disagree.** The 25 longest MSDWild files are 35.9 %
  of all scored speaker time and are the *easiest* (DER ~10–13 vs ~21 for the
  rest). On macro DER five of our systems sit within 0.71 of each other while
  the published system's lead stays clean. Report both; if they disagree, the
  pooled ordering is a long-file artifact.
- **MSDWild dev measures a different task.** Dev is 97.2 % five-to-ten
  speakers; test is 100 % two-to-four. **Zero overlap.** Dev DER sits near 50
  while test DER is near 17 on the same checkpoint, and dev DER is a mean over
  3 batches. No dev-based checkpoint selector is valid. Only dev `DER_miss`
  and `avg_pred_spk_qty` transfer to test.
- **Adapt-stage quality is anti-predictive of final quality.** Spearman
  **−0.667** on MSDWild across 8 lineages: the best adapt model finishes
  second-worst after finetuning, and the eventual winner was sixth of eight at
  adapt. A top-3-at-adapt screening rule would have discarded it.
- **Training loss does not rank models the way DER does** (Spearman +0.30 for
  both activation BCE and attractor-existence loss). The published system's
  dev loss got *worse* during finetuning while its DER improved more than
  ours. Optimising the objective harder does not produce better diarization.
- **RAMC's `G00000000` label** — present in all 43 test files, 3.22 % of
  reference speech, 0.87 s average turns — is a garbage/unknown bin, not a
  speaker. It inflates every RAMC DER by **~2.5**. Fine as a constant; never
  compare a RAMC number against a corpus scored without it.

---

## 3. The gap

The field scales simulated-conversation data and training length. DiaPer's own
two representational modules — the encoder's inductive bias, and the attractor
map's constrained form plus its unconstrained objective — have never been
revisited. That is what this work changes.

---

## 4. The hypothesis cycle

### H0 (not a contribution — setup). The recipe was under-specified.

Three recipe defects had to be fixed before any architecture claim could be
made, because each is *larger* than the effects being measured. They are
reported as setup, not as findings.

- **The Noam schedule was not reproducible.** `noam_warmup_fraction`
  re-derives the schedule from whatever batch size a run was last resumed
  with. One lineage configured for a 10 % warmup measurably peaked at 5.1 %,
  did 3.75× fewer optimizer steps than the published recipe, and ended at an
  LR 3.4× lower. Specifying `noam_model_size`/`noam_warmup_steps` explicitly
  fixes it. **Worth RAMC 23.85 → 20.80 with zero architecture change**
  (bootstrap +3.06, CI [+2.48, +3.67], 41/43 files better), and pretrain dev
  DER 10.00 → 3.75. *Report as a reproducibility defect; that framing is the
  contribution, "we changed the LR schedule" is not.*
- **The finetune LR was too small for our initialisation.** 1e-6 is the
  published value, but raising it to 1e-5 is worth **−1.48 DER**
  (CI [−2.34, −0.53], p = 0.0095) and reaches parity in **545 epochs instead
  of 900**. The gain is entirely detection (miss −0.87, FA −0.78).
- **Early stopping was a local deviation that fired too early.**
  `early_stopping_patience: 100` (the published recipe has no such field) cut
  one run off ~150 epochs before its own plateau; extending 305 → 750 epochs
  moved it 18.61 → 17.15. Combined with §2c's finding that dev DER cannot
  detect convergence at all, the protocol is now a **fixed 500-epoch cap,
  early stopping off, test scored once** — a prior commitment, because there
  is nothing valid to monitor.

**Everything below holds these three fixed and identical across arms.**

### H1. Plain self-attention is the wrong backbone for speech. `PENDING`

A convolution-augmented encoder represents frames better.

*Experiment:* self-attention → conformer k31, all else fixed
(arm **A3 → A4**).

*Prior evidence:* MSDWild 18.31 → 17.06/17.15, and the pretrain table in §2a.
**But confounded** — see §4.6.

### H2. The convex-combination attractor map is too constrained. `PENDING`

An MLP with genuine added capacity represents attractors better.

*Experiment:* `weighted_average` → `mlp` (arm **A2 → A3**).

*Note on the claim's shape:* `linear` is excluded, because the original work
already reported it **worse** than `weighted_average`. Since `linear` is *less*
constrained than a convex combination and loses, the hypothesis cannot be
"less constraint is better" — it must be **"the right added capacity"**. That
is a sharper and more falsifiable claim, and we owe it to the negative result.

### H3. `Le` regularises the wrong object. `PARTIAL`

The entropy term acts on combination weights; attractors need an **explicit
diversity objective** or they collapse.

*Evidence, at matched recipe.* `Le` is only computed for
`latents2attractors: weighted_average` (every other map returns a zero term),
so the arm that carries `Le` is the published architecture. The two
LR-clean isolated pairs in the 300 h sweep are the matched-recipe evidence:

| pair | what varies | MSDWild | RAMC |
|---|---|---|---|
| removing `Le` (map unchanged) | `l2a_entropy_loss_weight` 1.0 → 0.0 | 19.54 → **20.31** | 23.85 → **24.21** |
| adding unmasked diversity | on top of masked | 18.10 → **18.46** | 23.99 → 24.11 |

Both sides of each pair share the same LR, so the comparison is clean on that
axis. Limitations to state: 300 h scale rather than 2500 h, and
`early_stopping_patience` unset (framework default 30), so neither side is at
its ceiling.

*At the system level*, the swap is the published architecture (paperlr,
`weighted_average` + `Le`, at the authors' own recipe) against A2/A3/A4.
There is deliberately **no re-finetuned `Le` arm at our recipe** — building
one would mean improving the published baseline's optimizer settings, which
is the authors' work, not ours (see §5).

### H4. The resolution effect is "matched", not "finer". `PROVEN`

*Experiment.* Each corpus keeps its **own** standard inference protocol; only
*training* resolution varies. That makes the same intervention the matched arm
on one corpus and the mismatched arm on the other:

| corpus | infers at | train@10 | train@5 |
|---|---|---|---|
| RAMC | subsampling 5 | mismatched (default recipe) | **matched** |
| MSDWild | subsampling 10 | **matched** (default recipe) | mismatched |

*Result.* On RAMC, training at the evaluated resolution is worth
**−4.84 DER** (E-Branchformer) and **−4.76** (conformer) — 40/43 and 41/43
files, sign test p ≈ 1.5e-9 and 1.1e-10. On MSDWild it produces **no gain**
(18.12 vs 17.15 at matched epochs; paired bootstrap CI [−1.79, −0.22] in
favour of *not* doing it).

**The null result is the proof.** "Finer is better" predicts a gain on both
corpora; "matched is better" predicts a gain only where there is a mismatch.
The data match the second.

*The step-count confound is controlled.* The sub5 arm uses
`num_frames 1200 @ subsampling 5` against `600 @ 10` — **identical 6000 raw
frames (60 s)**, so chunks per epoch are identical. One lane wins while
running **0.97× the optimizer steps** of its baseline, which kills "it just
trained longer".

*The mechanism is measured, not asserted.* A sub10-trained model decoded at
sub5 emits **41,273 segments averaging 1.36 s, totalling 15.63 h**, against a
reference of **25,370 / 2.50 s / 17.62 h** — it shreds speech into fragments
and loses 11 % of it. The sub5-trained model emits **25,999 / 2.41 s /
17.37 h**: it *reproduces the reference's segment statistics.* The entire DER
gain is recovered missed speech (−7.9).

### H5. The three changes together beat the published method. `PENDING`

*Experiment:* A4 against the published baseline — DiaPer's own numbers
(15.47 / 21.1) and our faithful reproduction of its recipe (paperlr: MSDWild
**18.31** at ep 551-561, RAMC **20.80** at ep 321-331).

This is a **system-level** claim, and the recipe is part of the system. The
architecture-level attribution comes from the matched-recipe ablations
(A2→A3, A3→A4, and H4 within A4), which is where the encoder and the
attractor map are isolated. Keeping the two levels separate is what makes
both honest; conflating them is the error to avoid.

### H6. Long-recording attractor instability. `DIAGNOSED, UNFIXED`

Attractors are estimated per 60 s chunk and applied to recordings up to 37×
longer (RAMC median ~31 min). Decomposing RAMC confusion by re-choosing the
permutation per 30 s window separates two things:

| | value | what it is |
|---|---|---|
| windowed confusion | 5.8–6.3 % | the invariant local floor of §2b |
| global − windowed | 2.2–4.5 DER | **whole-file permutation drift** |

Only 6–9 % of windows disagree with the file's global permutation, so a few
long stretches carry it. Dev cannot see any of this: on the decisive
sub5-vs-sub10 pair, **dev DER differs by 0.0–0.2 while test DER differs by
4.7–4.8.**

This is the clearest open lever in the project (worth 2.2–4.5 DER on RAMC,
concentrated in 5–8 files) and it is **not addressed by this round**.
Candidates: overlapping inference windows with permutation alignment across
the stitch, or attractor re-estimation over the whole recording rather than
per chunk.

### 4.6. Why every architecture claim above is marked PENDING

The three existing 2500 h lineages differ in **three ways at once**:

| lineage | encoder | latents2attractors | diversity |
|---|---|---|---|
| `paperlr` | self-attention | `weighted_average` | — |
| `fixednoam_conformer_k31` | conformer k31 | **`mlp`** | **0.1** |
| `fixednoam_ebf` | E-Branchformer | **`mlp`** | **0.1** |

They also used different pretrain batch sizes (128 vs 96), hence different
Noam values. So "MSDWild 18.31 → 17.06" bundles the frame encoder, the
attractor map, the attractor objective *and* the optimizer budget into one
number — and **`mlp` and the diversity loss have never once been run apart
from each other**, so H2 and H3 are not separable from the existing data at
all.

Additionally, the older 300 h/500 h architecture sweep never set
`early_stopping_patience`, so all ~20 of those runs ran at the framework
default of 30 — a third of the patience already shown insufficient. None of
that sweep's numbers represent an architecture's ceiling.

**That is the hole this round fills.**

---

## 5. The experiment: a controlled factorial

Five arms, each a full three-stage pipeline (pretrain → adapt → finetune)
under one identical recipe, each differing from its comparison partner in
exactly one factor.

| arm | encoder | l2a map | `Le` | diversity | queued | role |
|---|---|---|---|---|---|---|
| A0 | self-attention | `weighted_average` | 1.0 | 0.0 | **no** | the published architecture |
| A1 | conformer k31 | `weighted_average` | 1.0 | 0.0 | **no** | orphaned without A0 |
| **A2** | self-attention | `weighted_average` | 0.0 | 0.1 | yes | H2 baseline |
| **A3** | self-attention | `mlp` | 0.0 | 0.1 | yes | the pivot |
| **A4** | conformer k31 | `mlp` | 0.0 | 0.1 | yes | the proposed system |

**The chain order is forced by a code constraint.** `Le` is only computed for
`latents2attractors: weighted_average`; every other map returns a zero term
(`models.py` ~1299–1307). So switching the map to `mlp` silently switches `Le`
off too, and a naive "weighted_average+Le vs mlp" arm would move two factors.
Ordering the chain so `Le` is *already off* before the map changes (A2 → A3)
makes H2 single-variable.

Single-variable comparisons, verified by pairwise config diff:

| hypothesis | comparison | what differs |
|---|---|---|
| H1 encoder | A3 → A4 | `frame_encoder_type` only |
| H2 attractor map | A2 → A3 | `latents2attractors` only |
| H4 resolution | A4, per corpus | training resolution only |

### The baseline is the published method, as published

**A0 and A1 are defined but not queued**, and this is a scope decision, not a
gap. The baseline for this work is **DiaPer as the authors specified it**, and
`paperlr` already reproduces that recipe faithfully — including the authors'
own finetune LR of 1e-6 — scoring MSDWild 18.31 and RAMC 20.80.

Re-finetuning that architecture at *our* recipe would mean improving the
published baseline's optimizer settings. That is the authors' work, not ours;
where their recipe underperforms, that is their published number to own. So
the comparison is: **each method at its own recipe** — theirs as published,
ours as ours — with the recipe counted as part of our system.

What keeps this honest is that the two claim levels stay separate:

- **system level** — A4 vs the published numbers and vs paperlr. The recipe
  is part of the system and the gain is the system's.
- **architecture level** — the matched-recipe ablations above, where the
  encoder and the attractor map are isolated with everything else identical.

Any claim of the form "the conformer is worth X DER" comes from A3→A4, never
from A4-vs-paperlr.

One knock-on constraint: the fresh arms use paperlr's **exact** pretrain and
adapt schedule, which is why the configs do not re-derive Noam. That keeps
the SC stages comparable across the whole family.

**Finetune coverage.** MSDWild for all queued arms (it is the multi-speaker
benchmark, so it carries the H1/H2 ablation); RAMC on A4 only, which is a
complete H4 test on one architecture — train@10 (mismatched) vs train@5
(matched) — plus the MSDWild train@5 null arm.

**Deferred:** E-Branchformer (queue size), and the pretrain-matched version of
H2 (the two maps differ in parameter shape, so a shared pretrain warm-starts
asymmetrically; not worth a caveated row this round).

**Excluded permanently:** ensembles and system combination — the contribution
has to be a single model; and any pipeline that depends on the original
authors' released weights — a result that needs their checkpoint is not our
result. Running their weights through our loop as a *diagnostic reference*
remains fine and has been useful.

Run with `scripts/run_4gpu_story_queue.sh` (configs from
`scripts/gen_story_queue_configs.py`). **≈ 261 GPU-h, ~4 days wall** on
4×V100: three arms of pretrain → adapt → MSDWild finetune, with GPU 3 picking
up A4's two RAMC finetunes as soon as A4's adapt checkpoint exists, which
keeps the critical path off a single lane.

---

## 6. Ablation study: metrics that show *why*

The claim is representational, so the ablations must measure representation,
not only DER.

1. **Attractor discriminability** — pairwise cosine similarity among active
   attractors, and the top-2 attractor logit margin per frame. The direct test
   of H2/H3. Tooling exists: `diaper/attractor_collapse_analysis.py`,
   `diaper/analyze_attractor_branch.py`.
2. **Collapse rate** — files emitting <2 speakers, and max predicted
   speaker-duration ratio. **Mandatory on every RAMC claim**: three successive
   rounds of RAMC gains turned out to be one collapsed file recovering each
   time (one file moved 57.99 → 29.96, supplying ~70 % of that round's macro
   gain).
3. **Local vs drift confusion** — the 30 s-window permutation rescoring of H6.
   Separates a decoding-scope problem from a discrimination floor; they respond
   to completely different fixes.
4. **DER miss / FA / confusion**, exact `md-eval-22.pl -af`. Reproduces
   dscore's pooled OVERALL exactly and yields the split for free — prefer it to
   reconstructing per-file DER, which runs ~1.9 high on MSDWild.
5. **Segment statistics vs reference** (count, mean duration, total speech) —
   what made H4's mechanism legible.
6. **Speaker-count calibration** — under/exact/over %, mean(n_sys − n_ref),
   bucketed by reference speaker count.
7. **Macro DER alongside pooled**, always (§2c).
8. **Paired bootstrap CI** on every headline delta, with the 0.25 seed floor
   stated once.
9. **Pretrain/adapt dev DER** per arm — cheap, and it is where H1's effect is
   largest (10.00 vs 2.60).

---

## 7. Failed hypotheses

Reported deliberately: the original authors devote a section to what did not
work, and it is the most reusable part of their paper.

**Ours:**

| idea | verdict |
|---|---|
| `pos_weight` on the attractor-existence BCE | **dead.** 3.0 vs 5.0 is a 67 % dose increase with a *flat* response (18.53 vs 18.62; under-prediction 36.5 % vs 37.8 %). The hardest 4-speaker bucket did not move at all. The best-calibrated model uses `pos_weight` 1.0 |
| frame-activation threshold | **exhausted in both directions.** 0.5 is near-optimal on miss-heavy *and* FA-heavy models; 0.5→0.6 cuts FA 2.11 but costs 2.81 miss |
| median filter window | already optimal, and the right window tracks **wall-clock** (~1.05 s), not frame count — 11 at sub10 ≡ 21 at sub5 |
| checkpoint-averaging window | noise-level |
| more dropout | not a difference from the reference (identical 0.1), and we do not overfit — test macro DER is *better* than train |
| longer / variable training chunks | **ruled out.** The gap *shrinks* monotonically with chunk length and vanishes at 240 s; long files are the easiest bucket |
| more SC data | 8× more changes 2-speaker files by **0.01 DER**; all benefit lands on 3–4-speaker files |
| SC data *quality* as the reproduction gap | **rejected twice** (dev-loss and test-DER). At adapt stage our architecture-matched arm trails by only +1.38, and one lineage *beats* the reference adapt checkpoint by 6 DER |
| overlap-loss weighting | motivation killed by stratification — the *lowest*-overlap bucket has the *largest* relative gap |
| speaker-counting head | ablated off, and off is better |

**Theirs, already ruled out — check before proposing:** absolute positional
encoding in the attractor decoder, SpecAugment, a speaker-recognition loss,
an LSTM over output activities, a dedicated silence attractor, a `linear`
latents→attractors layer, cosine (length-normalised) frame/attractor
comparison, cross-attention instead of dot product, and power-set encoding.

Note that three ideas our own diagnostics point at are already on that list —
the speaker-recognition loss (the obvious attack on confusion), cosine
attractor comparison (the other route to separable attractors), and
SpecAugment. `latents2attractors: mlp` is **not** on it; they tested `linear`
only. That is why H2 is ours to claim.

---

## Reproducing any number here

- DER is never computed in this repo for reporting. Score RTTMs with
  [dscore](https://github.com/nryant/dscore): **MSDWild `--collar 0.25`**,
  **RAMC collar 0** (default, no flag).
- Post-processing is tied to the collar: MSDWild median 11 at subsampling 10;
  RAMC median 1 at subsampling 5.
- Whole-recording RAMC inference is **CPU-only** (~120–130 GB RAM, ~50 min for
  43 files) and two concurrent runs take the box down.
- Every DER quoted above is re-scorable from a named RTTM directory; the
  per-experiment ledger is `private/results_all_local_experiments.csv`.
  Known defect in that ledger: the 23.71/18.12 row is the
  counting-head-**off** run, not an `overlaploss_3` run.
