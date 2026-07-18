#!/usr/bin/env python3

# Compute noam_model_size / noam_warmup_steps for backend/updater.py's
# NoamOpt, given your training budget and a target peak learning rate.
#
# NoamOpt.rate(step) = model_size^-0.5 * min(step^-0.5, step * warmup^-1.5)
# The two branches meet at step == warmup, where the LR peaks at
#     peak_lr = 1 / sqrt(model_size * warmup)
# so for a chosen warmup fraction of total training and a chosen peak_lr,
# model_size = 1 / (peak_lr**2 * warmup).
#
# Defaults:
#   --peak-lr 9.882e-5: corroborated two independent ways -- it falls out of
#     BOTH original DiaPer configs (512/200000 and 1024/100000) despite very
#     different model_size/warmup values, AND it's the exact LR that cleanly
#     converged a 2-chunk overfit to DER=0 in a direct experiment against
#     this codebase. Treat this one as validated.
#   --warmup-fraction 0.10: NOT reverse-engineered from the original configs
#     -- those turned out to have warmup_steps far exceeding total_steps at
#     any reasonable batch/GPU/epoch count we could reconstruct (200000
#     steps needed 130-170+ epochs to complete depending on assumptions),
#     and we could not confirm whether that was a deliberate "never fully
#     ramp" design or just an undertuned/inherited value. 0.10 is instead
#     the conventional Transformer-training fraction (a schedule that
#     actually completes its ramp and enters a normal decay phase within
#     your run) -- pick something you can reason about and observe
#     completing, rather than a number whose provenance we can't verify.

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Compute noam_model_size/noam_warmup_steps for a "
                     "training budget and target peak LR.")
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--iters-per-epoch', type=int, required=True)
    parser.add_argument('--warmup-fraction', type=float, default=0.10,
                         help="fraction of total steps spent ramping up "
                              "(default 0.10, the conventional "
                              "Transformer-training fraction; use >1 for a "
                              "schedule that never finishes ramping)")
    parser.add_argument('--peak-lr', type=float, default=9.882e-5,
                         help="target peak learning rate (default 9.882e-5, "
                              "validated -- see module docstring)")
    args = parser.parse_args()

    total_steps = args.epochs * args.iters_per_epoch
    warmup = round(total_steps * args.warmup_fraction)
    model_size = round(1 / (args.peak_lr ** 2 * warmup))
    actual_peak = 1 / (model_size * warmup) ** 0.5

    print(f"total_steps        = {total_steps}")
    print(f"noam_warmup_steps   = {warmup}  "
          f"({100 * warmup / total_steps:.1f}% of total)")
    print(f"noam_model_size     = {model_size}")
    print(f"resulting peak LR   = {actual_peak:.3e} "
          f"(target was {args.peak_lr:.3e})")

    if warmup >= total_steps:
        final_lr = actual_peak * (total_steps / warmup)
        print(f"NOTE: warmup >= total_steps -- the schedule never finishes "
              f"ramping. LR climbs monotonically all run, ending at "
              f"{100 * total_steps / warmup:.1f}% of peak ({final_lr:.3e}). "
              f"That's unusual -- double check this is what you want.")
    else:
        final_lr = (model_size * total_steps) ** -0.5
        print(f"NOTE: LR peaks at step {warmup} "
              f"({100 * warmup / total_steps:.1f}% through training), then "
              f"decays to {final_lr:.3e} by the end (normal ramp-then-decay "
              f"shape).")


if __name__ == '__main__':
    main()
