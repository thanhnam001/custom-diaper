#!/usr/bin/env python3
"""Generate the five story-queue arm config sets.

Run from the repo root:

    python scripts/gen_story_queue_configs.py

Writes models/10attractors/SC_LibriSpeech_2spk_2500h_story_A{0..4}/ with six
to eight yaml files each. All five arms are generated; the queue runs A1-A4
by default (see STORY_ARMS in run_4gpu_story_queue.sh).

Re-running overwrites them, so edit THIS file rather than the generated
yaml -- the whole point of generating them is that the single-variable
property below is structural, not something to verify by eye afterwards.

WHY THIS EXISTS
===============
The architecture claims in this project cannot be made from the existing
2500h lineages, because those differ in three ways at once:

    lineage                  encoder        l2a map            diversity
    paperlr                  self-attn      weighted_average   --
    fixednoam_conformer_k31  conformer k31  mlp                0.1
    fixednoam_ebf            E-Branchformer mlp                0.1

They also used different pretrain batch sizes (128 vs 96) and therefore
different Noam values. So "18.31 -> 17.06" bundles the frame encoder, the
latents2attractors map, the attractor objective AND the optimizer budget
into one number, and `mlp` + the diversity loss have literally never been
run apart from each other.

The arms below are fresh three-stage pipelines (pretrain -> adapt ->
finetune) sharing one identical recipe, so that each comparison changes only
what it claims to change.

THE DESIGN
==========
Three factors, but note the constraint that shapes it: the entropy term Le
is only computed for `latents2attractors: weighted_average`. models.py
returns a zero term for every other map (see the `l2a_entropy_term_i =
torch.zeros(1)[0]` branches around models.py:1299-1307). So switching the
map to `mlp` silently switches Le off too, and a naive
"weighted_average+Le vs mlp" arm would move two factors at once.

The chain is therefore ordered so Le is already off before the map changes:

    arm  encoder        l2a map            Le    diversity
    A0   self_attention weighted_average   1.0   0.0     <- the published
                                                        architecture; NOT
                                                        queued, see below
    A1   conformer k31  weighted_average   1.0   0.0
    A2   self_attention weighted_average   0.0   0.1
    A3   self_attention mlp                0.0   0.1
    A4   conformer k31  mlp                0.0   0.1     <- proposed system

QUEUED BY DEFAULT: A1, A2, A3, A4. A0 is defined but out of scope (below).

Matched-recipe comparisons available from the queued arms:

    frame encoder      A3 -> A4   frame_encoder_type only (branch = ours)
    l2a map            A2 -> A3   latents2attractors only (Le off both sides)
    attractor branch   A1 -> A4   encoder fixed = conformer; the map and the
                                  objective move together, so this is the
                                  branch as a PACKAGE
    resolution         A4, per corpus (H4)

A1 -> A4 and A2 -> A3 together decompose the attractor branch: the package
effect on a fixed encoder, and the map component on its own. The objective's
share is then the difference, which is an inference rather than an
isolation -- isolating it would need a conformer + weighted_average +
diversity arm, and none is queued.

A1 is also a method in its own right (conformer encoder on the PAPER's
attractor branch), so it carries a system-level row, not just a control.

`linear` is excluded on purpose -- the paper already reported it worse than
`weighted_average`, which is also why H2's claim is "the right added
capacity", not "less constraint".

NOAM, AND WHY A0 IS OUT OF SCOPE
================================
A0 *is* the already-trained `paperlr` lineage: self-attention +
weighted_average + Le, at 2500h. The baseline for this work is DiaPer AS
THE AUTHORS SPECIFIED IT, and paperlr already reproduces that recipe
faithfully -- including the authors' own finetune LR of 1e-6 -- at MSDWild
18.31 (ep 551-561) and RAMC 20.80 (ep 321-331). Re-finetuning it at OUR
recipe would mean improving the published baseline's optimizer settings,
which is the authors' work and not ours.

So A0 is not queued, and paperlr's existing numbers serve as the published
baseline row directly.

Every fresh arm still uses paperlr's EXACT pretrain and adapt schedule --
the proven 512/50000 at batch 128 (pretrain) and 2111/48500 at batch 22
(adapt), not re-derived values -- so the SC stages stay comparable across
the whole family, including against paperlr itself.

Known wart, to report rather than fix here: that 50,000-step warmup was
derived assuming 119,160 pretrain chunks, from a *measured* 993 steps/epoch
at batch 120 -- the same measurement style that was exactly 2x wrong for
the adapt cache (asserted 19,888, actual 39,064, because each rank of a
2-rank DDP run sees half the epoch). The physical count is 150,000 (a
pretrain chunk is num_frames 600 x subsampling 10 = 6000 raw frames, and
frame_shift 160 at 16 kHz is 10 ms per raw frame, so exactly 60 s; 2500 h
is 150,000 minutes). If so the realised ramp is 42.7%, not the 53.7% the
paperlr config header claims. The queue's preflight counts the real cache
and logs the true ramp so the write-up can state it correctly -- but it
does NOT change the schedule, because uniformity across arms matters more
here than the schedule's absolute optimality, and paperlr's schedule is
the one with proven results behind it.

Adapt keeps 2111/48500 at batch 22 (peak 9.882e-5, realised ramp 27.3%) for
the same reason.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, 'models', '10attractors')

# Server-side roots. These match every other 2500h lineage in the repo.
DATA = '/data/ocr/namvt17/dataset/diarization'
EXP = '/data/ocr/namvt17/custom-diaper/experiments/10attractors'

PRETRAIN_DATA = f'{DATA}/diaper_precompute_2500h_fixed_2spks'
# A0 does not train its own SC stages -- it reuses the already-trained
# paperlr lineage, which IS this architecture (self-attention +
# weighted_average + Le) at 2500h on the corrected Noam schedule.
PAPERLR_ADAPT = (f'{EXP}/SC_LibriSpeech_2spk_2500h_paperlr'
                 '_adapted1-10_2500h_maximum10spks')
ADAPT_DATA = f'{DATA}/diaper_precompute_2500h_maximum_10spks_24000frames'
MSDWILD_DATA = f'{DATA}/msdwild_precompute_6000frames'
RAMC_DATA = f'{DATA}/ramc_precomputed_6000frames'
MSDWILD_TEST = f'{DATA}/msdwild/kaldi/test'
RAMC_TEST = f'{DATA}/ramc/kaldi/test'

# Noam at pretrain: paperlr's PROVEN values at batch 128. Deliberately not
# re-derived -- see "NOAM, AND WHY A0 IS OUT OF SCOPE" in the module
# docstring. Every fresh arm uses the identical schedule so the SC stages
# stay comparable across the family, including against paperlr itself.
PRETRAIN_NOAM_MODEL_SIZE = 512
PRETRAIN_NOAM_WARMUP_STEPS = 50000
# Adapt: the proven paperlr values at batch 22 (peak 9.882e-5).
ADAPT_NOAM_MODEL_SIZE = 2111
ADAPT_NOAM_WARMUP_STEPS = 48500

# Batch sizes. Every finetune doubles the historical value (the user's call,
# to use the 32 GB V100s): sub10/600-frame arms 32 -> 64, and the
# sub5/1200-frame arms 16 -> 32 so that tokens-per-batch stay matched
# between the two resolutions, which is what makes H4 a clean comparison.
FT_BATCH_SUB10 = 64
FT_BATCH_SUB5 = 32

# The shared architecture + data + optimization settings. Identical in every
# arm and every stage unless a stage or arm override below changes it.
BASE = {
    'activation_loss_BCE_weight': 1.0,
    'activation_loss_DER_weight': 0.0,
    'attractor_existence_loss_weight': 1.0,
    'attractor_frame_comparison': 'dotprod',
    'condition_frame_encoder': True,
    'context_size': 7,
    'd_latents': 128,
    'detach_attractor_loss': False,
    'dropout_attractors': 0.1,
    'dropout_frames': 0.1,
    'feature_dim': 40,
    'frame_encoder_heads': 4,
    'frame_encoder_layers': 4,
    'frame_encoder_units': 2048,
    'frame_shift': 160,
    'frame_size': 400,
    'gpu': 1,
    'gradclip': 5,
    'input_transform': 'logmel_meannorm',
    'intermediate_loss_frameencoder': True,
    'intermediate_loss_perceiver': True,
    'model_type': 'AttractorPerceiver',
    'n_attractors': 10,
    'n_blocks_attractors': 3,
    'n_internal_blocks_attractors': 1,
    'n_latents': 128,
    'n_sa_heads_attractors': 4,
    'n_selfattends_attractors': 2,
    'n_xa_heads_attractors': 4,
    'norm_loss_per_spk': True,
    'num_threads': 1,
    'num_workers': 4,
    'pre_xa_heads': 4,
    'sampling_rate': 16000,
    'seed': 3,
    'specaugment': False,
    'time_shuffle': False,
    'use_frame_selfattention': True,
    'use_last_samples': True,
    'use_posenc': False,
    'use_pre_crossattention': True,
}

# The five arms. These four keys are the ONLY architecture/objective
# difference between arms -- everything else comes from BASE.
ARMS = {
    'A0': {
        'desc': "baseline: the paper's own architecture "
                '(self-attention + weighted_average + entropy term Le)',
        'role': 'the anchor every other arm is compared against',
        'cfg': {
            'frame_encoder_type': 'self_attention',
            'latents2attractors': 'weighted_average',
            'l2a_entropy_loss_weight': 1.0,
            'attractor_diversity_loss_weight': 0.0,
        },
    },
    'A1': {
        'desc': 'A0 + conformer frame encoder (kernel 31)',
        'role': 'H1 alone, vs A0. Only the frame encoder changes.',
        'cfg': {
            'frame_encoder_type': 'conformer',
            'conformer_conv_kernel_size': 31,
            'conv_norm_type': 'batchnorm',
            'latents2attractors': 'weighted_average',
            'l2a_entropy_loss_weight': 1.0,
            'attractor_diversity_loss_weight': 0.0,
        },
    },
    'A2': {
        'desc': "A0 with the entropy term Le replaced by an explicit "
                'attractor diversity penalty',
        'role': 'H3 alone, vs A0. Only the attractor objective changes.',
        'cfg': {
            'frame_encoder_type': 'self_attention',
            'latents2attractors': 'weighted_average',
            'l2a_entropy_loss_weight': 0.0,
            'attractor_diversity_loss_weight': 0.1,
        },
    },
    'A3': {
        'desc': 'A2 + mlp latents2attractors instead of the convex '
                'combination',
        'role': 'H2 alone, vs A2. Only the latents2attractors map changes '
                '(Le is already off in both, which is the point of ordering '
                'the chain this way).',
        'cfg': {
            'frame_encoder_type': 'self_attention',
            'latents2attractors': 'mlp',
            'l2a_entropy_loss_weight': 0.0,
            'attractor_diversity_loss_weight': 0.1,
        },
    },
    'A4': {
        'desc': 'the proposed system: conformer encoder + mlp '
                'latents2attractors + attractor diversity penalty',
        'role': 'H5 (all three together) vs A0, and H1 again vs A3.',
        'cfg': {
            'frame_encoder_type': 'conformer',
            'conformer_conv_kernel_size': 31,
            'conv_norm_type': 'batchnorm',
            'latents2attractors': 'mlp',
            'l2a_entropy_loss_weight': 0.0,
            'attractor_diversity_loss_weight': 0.1,
        },
    },
}

# QUEUED vs DEFINED. All five arms are DEFINED (and their configs generated)
# so that re-enabling one is a single env var on the queue, but the queue runs
# only A2/A3/A4 by default -- see STORY_ARMS in run_4gpu_story_queue.sh.
#
# A0 is defined-but-not-queued by design. The baseline for this work
# is DiaPer AS THE AUTHORS SPECIFIED IT, and `paperlr` already reproduces that
# recipe faithfully -- including the authors' own finetune LR of 1e-6 --
# scoring MSDWild 18.31 (ep 551-561) and RAMC 20.80 (ep 321-331).
#
# Re-finetuning that architecture at OUR recipe would mean improving the
# published baseline's optimizer settings, which is the authors' work and not
# ours; where their recipe underperforms, that is their published number to
# own. So each method runs at its own recipe -- theirs as published, ours as
# ours -- with the recipe counted as part of our system.
#
# Two claim levels, kept separate:
#   system level        A4 vs the published numbers and vs paperlr; the recipe
#                       is part of the system, so the gain is the system's.
#   architecture level  the matched-recipe ablations A2 -> A3 (map) and
#                       A3 -> A4 (encoder), plus H4 inside A4, where
#                       everything else is identical.
# "The conformer is worth X DER" comes from A3 -> A4, never A4-vs-paperlr.
#
# A1 IS queued: it is a method in its own right (conformer + the paper's
# attractor branch) and it is A4's matched-recipe partner for the attractor
# branch as a package (same encoder, map + objective move together). With
# A2 -> A3 giving the map alone, the two together decompose the branch.
#
# The Le -> diversity swap is therefore bracketed rather than isolated at
# our recipe (A1 -> A4 = package, A2 -> A3 = map); its isolated
# matched-recipe evidence comes from the 300h sweep (two LR-clean pairs;
# see research_story.md).
#
# Which arms get which finetunes. MSDWild carries the H1/H2/H3 ablation
# because it is the multi-speaker benchmark; RAMC is 2-speaker, so only the
# baseline and the proposed system run there. The H4 resolution pair runs on
# A0 and A4 for RAMC (where sub5 is the MATCHED arm) and on A4 for MSDWild
# (where sub5 is the MISMATCHED arm and H4 predicts no gain).
RAMC_ARMS = ('A0', 'A4')  # A0 only if re-enabled; see QUEUED_ARMS note
MSDWILD_SUB5_ARMS = ('A4',)


# A0 trains no SC stages; INHERITS_SC lists the arms that do not.
INHERITS_SC = ('A0',)


def _stage_paths(arm):
    """(pretrain_dir, adapt_dir, finetune_base_dir).

    adapt_dir is where the arm's finetunes take their init weights from --
    for A0 that is paperlr's existing adapt run, not a directory this queue
    ever writes. finetune_base_dir is always story-owned so an A0 finetune
    can never collide with paperlr's own recorded finetunes."""
    pre = f'{EXP}/SC_LibriSpeech_2spk_2500h_story_{arm}_pretrain2500h'
    adapt = (PAPERLR_ADAPT if arm in INHERITS_SC
             else f'{EXP}/SC_LibriSpeech_2spk_2500h_story_{arm}_adapt2500h')
    base = f'{EXP}/SC_LibriSpeech_2spk_2500h_story_{arm}'
    return pre, adapt, base


def pretrain_cfg(arm):
    pre, _, _ = _stage_paths(arm)
    cfg = dict(BASE, **ARMS[arm]['cfg'])
    cfg.update({
        'dev_batchsize': 96,
        'train_batchsize': 128,
        'max_epochs': 100,
        'noam_model_size': PRETRAIN_NOAM_MODEL_SIZE,
        'noam_warmup_steps': PRETRAIN_NOAM_WARMUP_STEPS,
        'num_frames': 600,
        'num_speakers': 2,
        'optimizer': 'noam',
        'subsampling': 10,
        'log_report_batches_num': 256,
        'output_path': pre,
        'train_precomputed_dir': f'{PRETRAIN_DATA}/train',
        'valid_precomputed_dir': f'{PRETRAIN_DATA}/validation',
    })
    return cfg


def adapt_cfg(arm):
    pre, adapt, _ = _stage_paths(arm)
    cfg = dict(BASE, **ARMS[arm]['cfg'])
    cfg.update({
        'dev_batchsize': 32,
        'train_batchsize': 22,
        'max_epochs': 100,
        'noam_model_size': ADAPT_NOAM_MODEL_SIZE,
        'noam_warmup_steps': ADAPT_NOAM_WARMUP_STEPS,
        'num_frames': 2400,
        'num_speakers': 10,
        'optimizer': 'noam',
        'subsampling': 10,
        # 24 to match paperlr's adapt run, which A0 inherits verbatim.
        'log_report_batches_num': 24,
        'init_model_path': f'{pre}/models',
        'init_epochs': '90-100',
        'output_path': adapt,
        'train_precomputed_dir': f'{ADAPT_DATA}/train',
        'valid_precomputed_dir': f'{ADAPT_DATA}/validation',
    })
    return cfg


def finetune_cfg(arm, corpus, subsampling):
    """corpus in {'msdwild','ramc'}; subsampling in {10,5}."""
    _, adapt, base = _stage_paths(arm)
    data = MSDWILD_DATA if corpus == 'msdwild' else RAMC_DATA
    tag = 'MSDWILD' if corpus == 'msdwild' else 'RAMC'
    suffix = '' if subsampling == 10 else '_sub5'
    cfg = dict(BASE, **ARMS[arm]['cfg'])
    cfg.update({
        'dev_batchsize': 128,
        # The standing finetune protocol: lr 1e-5 (not the inherited 1e-6),
        # a fixed 500-epoch cap, early stopping OFF, test scored once at the
        # cap. MSDWild dev has zero speaker-count overlap with its test set
        # (dev 97.2% five-to-ten speakers, test 100% two-to-four), so no
        # dev-based selector is valid and the cap must be a prior commitment.
        'lr': 1e-5,
        'max_epochs': 500,
        'early_stopping': False,
        'optimizer': 'adam',
        'num_speakers': 10,
        'log_report_batches_num': 10,
        'init_model_path': f'{adapt}/models',
        'init_epochs': '90-100',
        'output_path': f'{base}/models_finetune{tag}{suffix}',
        'train_precomputed_dir': f'{data}/train',
        'valid_precomputed_dir': f'{data}/dev',
    })
    if subsampling == 10:
        cfg.update({'subsampling': 10, 'num_frames': 600,
                    'train_batchsize': FT_BATCH_SUB10})
    else:
        # 1200 frames at subsampling 5 is the same 6000 raw frames (60 s) as
        # 600 at subsampling 10, and half the batch keeps tokens-per-batch
        # matched between the two, so H4 compares resolution and nothing else.
        cfg.update({'subsampling': 5, 'num_frames': 1200,
                    'train_batchsize': FT_BATCH_SUB5})
    return cfg


def infer_cfg(arm, corpus, trained_subsampling):
    """Inference always uses the corpus's OWN standard protocol, whatever
    resolution the model was trained at -- that is what makes H4 a fair
    test of train/eval matching rather than a change of evaluation."""
    _, _, base = _stage_paths(arm)
    tag = 'MSDWILD' if corpus == 'msdwild' else 'RAMC'
    suffix = '' if trained_subsampling == 10 else '_sub5'
    ft_dir = f'{base}/models_finetune{tag}{suffix}'
    cfg = dict(BASE, **ARMS[arm]['cfg'])
    for k in ('activation_loss_BCE_weight', 'activation_loss_DER_weight',
              'attractor_existence_loss_weight', 'detach_attractor_loss',
              'intermediate_loss_frameencoder', 'intermediate_loss_perceiver',
              'norm_loss_per_spk', 'specaugment', 'use_last_samples',
              'num_workers', 'num_threads', 'gradclip',
              'l2a_entropy_loss_weight', 'attractor_diversity_loss_weight'):
        cfg.pop(k, None)
    cfg.update({
        'estimate_spk_qty': -1,
        'estimate_spk_qty_thr': 0.5,
        'threshold': 0.5,
        'num_frames': -1,      # whole recording
        'num_speakers': 10,
        'models_path': f'{ft_dir}/models',
        'rttms_dir': f'{ft_dir}/{corpus}_test_pred',
        'infer_data_dir': MSDWILD_TEST if corpus == 'msdwild' else RAMC_TEST,
    })
    if corpus == 'msdwild':
        # MSDWild: collar 0.25 scoring, so median 11 at subsampling 10.
        cfg.update({'subsampling': 10, 'median_window_length': 11})
    else:
        # RAMC: collar 0 scoring, so no median filter and finer resolution.
        cfg.update({'subsampling': 5, 'median_window_length': 1})
    return cfg


HEADER = """# STORY QUEUE ARM {arm} -- {stage}
#
# {desc}
#
# Role in the design: {role}
#
# GENERATED by scripts/gen_story_queue_configs.py -- do not hand-edit, your
# change will be lost on the next regeneration. Edit the generator instead;
# it exists so that "each arm differs from its comparison partner in exactly
# one factor" is structural rather than something to verify by eye.
#
# The four factor keys, for this arm:
{factors}
"""


def write_cfg(path, arm, stage, cfg):
    factor_keys = ('frame_encoder_type', 'conformer_conv_kernel_size',
                   'latents2attractors', 'l2a_entropy_loss_weight',
                   'attractor_diversity_loss_weight')
    factors = '\n'.join(
        f'#   {k}: {cfg[k]}' for k in factor_keys if k in cfg)
    header = HEADER.format(arm=arm, stage=stage, desc=ARMS[arm]['desc'],
                           role=ARMS[arm]['role'], factors=factors)
    lines = [f'{k}: {_fmt(v)}' for k, v in sorted(cfg.items())]
    with open(path, 'w', newline='\n') as f:
        f.write(header + '\n' + '\n'.join(lines) + '\n')


def _fmt(v):
    if isinstance(v, bool):
        return 'True' if v else 'False'
    if isinstance(v, float):
        # repr keeps 0.0 as "0.0" and 1e-05 as "1e-05", both of which match
        # how the hand-written configs in this repo spell them.
        return repr(v)
    return str(v)


def main():
    written = []
    for arm in sorted(ARMS):
        d = os.path.join(MODELS_DIR,
                         f'SC_LibriSpeech_2spk_2500h_story_{arm}')
        os.makedirs(d, exist_ok=True)

        inherited = arm in INHERITS_SC
        sc_note = ('' if not inherited else
                   ' [NOT RUN BY THE QUEUE -- this arm inherits the '
                   'already-trained paperlr checkpoints. Recorded here so '
                   'the inherited recipe is explicit and diffable against '
                   'the fresh arms.]')
        write_cfg(os.path.join(d, 'train.yaml'), arm,
                  'stage 1/3: pretrain, 2500h 2-speaker SC' + sc_note,
                  pretrain_cfg(arm))
        write_cfg(os.path.join(d, 'train_10spks.yaml'), arm,
                  'stage 2/3: adapt, 2500h 1-10 speaker SC' + sc_note,
                  adapt_cfg(arm))
        write_cfg(os.path.join(d, 'finetune_msdwild_10spks.yaml'), arm,
                  'stage 3/3: MSDWild finetune (subsampling 10, matched to '
                  'MSDWild inference)',
                  finetune_cfg(arm, 'msdwild', 10))
        write_cfg(os.path.join(d, 'infer_msdwild.yaml'), arm,
                  'inference: MSDWild test, standard protocol '
                  '(subsampling 10 / median 11, scored at collar 0.25)',
                  infer_cfg(arm, 'msdwild', 10))
        written.append((arm, 'msdwild sub10',
                        'ft+infer (SC inherited)' if inherited
                        else 'pretrain+adapt+ft+infer'))

        if arm in MSDWILD_SUB5_ARMS:
            write_cfg(os.path.join(d, 'finetune_msdwild_10spks_sub5.yaml'),
                      arm,
                      'H4 MISMATCHED arm: MSDWild finetune at subsampling 5 '
                      'while MSDWild inference stays at 10 -- H4 predicts NO '
                      'gain here, which is what makes it a mechanism',
                      finetune_cfg(arm, 'msdwild', 5))
            write_cfg(os.path.join(d, 'infer_msdwild_sub5trained.yaml'), arm,
                      'inference for the sub5-trained MSDWild model, at '
                      "MSDWild's OWN standard protocol (subsampling 10 / "
                      'median 11)',
                      infer_cfg(arm, 'msdwild', 5))
            written.append((arm, 'msdwild sub5', 'ft+infer'))

        if arm in RAMC_ARMS:
            write_cfg(os.path.join(d, 'finetune_ramc_10spks.yaml'), arm,
                      'H4 MISMATCHED arm: RAMC finetune at subsampling 10 '
                      'while RAMC inference is at 5 -- this is the default '
                      'DiaPer recipe and the mismatch H4 is about',
                      finetune_cfg(arm, 'ramc', 10))
            write_cfg(os.path.join(d, 'infer_ramc.yaml'), arm,
                      'inference: RAMC test, standard protocol '
                      '(subsampling 5 / median 1, scored at collar 0)',
                      infer_cfg(arm, 'ramc', 10))
            write_cfg(os.path.join(d, 'finetune_ramc_10spks_sub5.yaml'), arm,
                      'H4 MATCHED arm: RAMC finetune at subsampling 5, the '
                      'resolution RAMC is actually evaluated at',
                      finetune_cfg(arm, 'ramc', 5))
            write_cfg(os.path.join(d, 'infer_ramc_sub5trained.yaml'), arm,
                      'inference for the sub5-trained RAMC model, at RAMC\'s '
                      'standard protocol (subsampling 5 / median 1)',
                      infer_cfg(arm, 'ramc', 5))
            written.append((arm, 'ramc sub10 + sub5', 'ft+infer x2'))

        print(f'wrote {d}')

    print()
    print('Finetune coverage:')
    for arm, what, how in written:
        print(f'  {arm}  {what:<20} {how}')


if __name__ == '__main__':
    main()
