#!/usr/bin/env python3

# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

import torch.optim as optim
from torch.nn import Module
from types import SimpleNamespace
from typing import Any, Dict, Tuple


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size: int, warmup: int, optimizer: optim) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state.
            Should be an object returned from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self) -> None:
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: int = None) -> float:
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def get_rate(self) -> float:
        return self._rate

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()


def compute_noam_params(
    max_epochs: int,
    iters_per_epoch: int,
    warmup_fraction: float,
    peak_lr: float,
) -> Tuple[int, int]:
    """Derive noam_model_size/noam_warmup_steps from a training budget and a
    target peak LR, so they don't have to be hand-computed per run (mirrors
    common_utils/noam_lr_calc.py -- see that module's docstring for the
    derivation and for the provenance of a validated default peak_lr).

    NoamOpt.rate(step) = model_size^-0.5 * min(step^-0.5, step*warmup^-1.5)
    peaks at step == warmup, where peak_lr = 1/sqrt(model_size*warmup); for a
    chosen warmup (a fraction of total steps) and peak_lr, solving for
    model_size gives model_size = 1 / (peak_lr**2 * warmup).
    """
    total_steps = max_epochs * iters_per_epoch
    warmup = max(1, round(total_steps * warmup_fraction))
    model_size = max(1, round(1 / (peak_lr ** 2 * warmup)))
    return model_size, warmup


def setup_optimizer(args: SimpleNamespace, model: Module) -> optim:
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'noam':
        optimizer = NoamOpt(
            args.noam_model_size,
            args.noam_warmup_steps,
            optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        raise ValueError(args.optimizer)
    return optimizer


def get_rate(optimizer: optim) -> float:
    if isinstance(optimizer, NoamOpt):
        return optimizer.get_rate()
    else:
        for param_group in optimizer.param_groups:
            return param_group['lr']
