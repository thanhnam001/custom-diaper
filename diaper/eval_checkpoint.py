#!/usr/bin/env python3

# Licensed under the MIT license.

"""Run a trained checkpoint over a validation set and report the same
metrics train.py logs at the end of each epoch (DER breakdown,
attractor_accuracy, etc.), for checkpoints saved before a given metric
existed. Reuses train.py's own parse_arguments/compute_loss_and_metrics so
the numbers are computed exactly the way train.py's dev loop computes them
-- pass the same training config plus --init-model-path/--init-epochs
pointing at the checkpoint(s) to evaluate.

Example:
    python diaper/eval_checkpoint.py -c examples/train_2speakers.yaml \\
        --init-model-path <output_path>/models --init-epochs 10 \\
        --valid-data-dir <dev Kaldi data directory> --gpu 1
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common_utils.collections_abc_compat  # noqa: E402,F401

from backend.losses import pad_labels_zeros, pad_sequence  # noqa: E402
from backend.models import average_checkpoints, get_model  # noqa: E402
from common_utils.diarization_dataset import (  # noqa: E402
    KaldiDiarizationDataset,
    PrecomputedDiarizationDataset)
from common_utils.precomputed_diarization_dataset import (  # noqa: E402
    PrecomputedKaldiDiarizationDataset)
from common_utils.metrics import new_metrics  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from types import SimpleNamespace  # noqa: E402
import functools  # noqa: E402
import numpy as np  # noqa: E402
import random  # noqa: E402
import threadpoolctl  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from train import (  # noqa: E402
    _convert,
    _format_metrics,
    _init_fn,
    compute_loss_and_metrics,
    parse_arguments,
)


def get_dev_dataloader(args: SimpleNamespace) -> DataLoader:
    dev_batchsize = (
        args.dev_batchsize * args.gpu if args.gpu >= 1
        else args.dev_batchsize)

    if args.valid_precomputed_dir is not None:
        dev_set = PrecomputedKaldiDiarizationDataset(
            precomputed_dir=args.valid_precomputed_dir,
            context_size=args.context_size,
            n_speakers=min(args.num_speakers, args.n_attractors),
            subsampling=args.subsampling,
            specaugment=args.specaugment,
        )
    elif args.valid_data_dir is not None:
        dev_set = KaldiDiarizationDataset(
            args.valid_data_dir,
            chunk_size=args.num_frames,
            context_size=args.context_size,
            feature_dim=args.feature_dim,
            frame_shift=args.frame_shift,
            frame_size=args.frame_size,
            input_transform=args.input_transform,
            n_speakers=min(args.num_speakers, args.n_attractors),
            sampling_rate=args.sampling_rate,
            shuffle=args.time_shuffle,
            subsampling=args.subsampling,
            use_last_samples=args.use_last_samples,
            min_length=args.min_length,
            specaugment=args.specaugment,
        )
    elif args.valid_features_dir is not None:
        dev_set = PrecomputedDiarizationDataset(
            features_dir=args.valid_features_dir,
            batch_size=args.dev_batchsize)
    else:
        raise ValueError(
            "One of --valid-precomputed-dir, --valid-data-dir or "
            "--valid-features-dir must be set to point at the validation "
            "set to evaluate on")

    return DataLoader(
        dev_set,
        batch_size=dev_batchsize,
        collate_fn=_convert,
        num_workers=1,
        shuffle=False,
        worker_init_fn=functools.partial(
            _init_fn, num_threads=args.num_threads),
    )


if __name__ == '__main__':
    args = parse_arguments()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        threadpoolctl.threadpool_limits(limits=args.num_threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.device = torch.device("cuda") if args.gpu >= 1 else \
        torch.device("cpu")

    assert args.init_model_path != '', \
        "--init-model-path must point at the directory containing the " \
        "checkpoint_<epoch>.tar file(s) to evaluate (e.g. " \
        "<output_path>/models)"
    assert args.init_epochs != '', \
        "--init-epochs must select which checkpoint(s) to evaluate -- a " \
        "single epoch (e.g. 10), a comma-separated list, or a - interval " \
        "to average several epochs together, same syntax as train.py's " \
        "--init-model-path/--init-epochs"

    model = get_model(args)
    model = average_checkpoints(
        args.device, model, args.init_model_path, args.init_epochs)
    model = model.to(args.device)
    model.eval()

    dev_loader = get_dev_dataloader(args)

    acum_dev_metrics = new_metrics()
    dev_batches_qty = 0
    with torch.no_grad():
        dev_pbar = tqdm(dev_loader, total=len(dev_loader))
        for i, batch in enumerate(dev_pbar):
            features = batch['xs']
            labels = batch['ts']
            spkids = batch['spk_ids']
            n_speakers = np.asarray([
                max(torch.where(t.sum(0) != 0)[0]) + 1
                if t.sum() > 0 else 0 for t in labels])
            max_n_speakers = args.n_attractors
            features, labels = pad_sequence(
                features, labels, args.num_frames)
            labels = pad_labels_zeros(labels, max_n_speakers)
            features = torch.stack(features).to(args.device)
            labels = torch.stack(labels).to(args.device)
            dev_loss, acum_dev_metrics = compute_loss_and_metrics(
                model, labels, features, n_speakers,
                spkids, acum_dev_metrics, args)
            if not torch.isfinite(dev_loss):
                dev_pbar.write(
                    f"batch {i + 1}: non-finite loss, excluded from "
                    f"metrics. names={batch['names'][:2]}")
                continue
            dev_batches_qty += 1
            dev_pbar.set_postfix(loss=f"{dev_loss.item():.4f}")

    print(
        f"\nCheckpoint(s) {args.init_epochs} from {args.init_model_path}, "
        f"{dev_batches_qty} dev batches evaluated:")
    if dev_batches_qty > 0:
        print(_format_metrics(acum_dev_metrics, dev_batches_qty))
        print("\nFull metrics:")
        for k, v in acum_dev_metrics.items():
            print(f"  {k}: {v / dev_batches_qty:.4f}")
    else:
        print("All dev batches produced non-finite loss; no metrics "
              "computed.")
