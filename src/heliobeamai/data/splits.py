"""Train/validation splits of calibration flux datasets.

A dataset stores per heliostat ``S`` measurements in ``n_blocks`` equal
target blocks (e.g. one block per target group). Each block starts with its
held-out test measurements, followed by the training measurements. A split
selects training measurements from some blocks and test measurements from
others:

* train sequence: ``n_train`` measurements of every training block, starting
  after ``offset`` (all remaining ones if ``n_train`` is None);
* validation sequence: the test measurements followed by the train sequence.
  During validation the first test measurements are predicted from the rest
  (see :func:`heliobeamai.data.batching.split_batch`).

Named presets reproduce the dissertation experiments. Their block indices
refer to the block order of the dissertation datasets
(``fluxTransData25All.pt``: 0 = target 3, 1 = targets 4/5/6, 2 = targets 1/7;
``fluxTransData40T7.pt``: one block, target 7). The 2026 dataset uses
(MFT = 3, Lower = 1/7/8/9, Upper = 4/5/6/10/11); its preset ``T2026`` only
uses symmetric splits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SplitSpec:
    name: str
    n_blocks: int = 1
    train_blocks: Tuple[int, ...] = (0,)
    n_train: Optional[int] = None
    offset: int = 0
    test: Tuple[Tuple[int, int], ...] = ()          # (block, count) from the block start

    def indices(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample indices ``(train, validation)`` for sequences of length ``n_samples``."""
        if n_samples % self.n_blocks:
            raise ValueError(f"{n_samples} samples cannot be split into {self.n_blocks} blocks")
        block = n_samples // self.n_blocks
        train = []
        for b in self.train_blocks:
            stop = block if self.n_train is None else self.offset + self.n_train
            if stop > block:
                raise ValueError(f"split {self.name}: block {b} has only {block} samples")
            train.append(torch.arange(self.offset, stop) + b * block)
        test = [torch.arange(count) + b * block for b, count in self.test]
        train_idx = torch.cat(train).long()
        val_idx = torch.cat(test + [train_idx]).long() if test else train_idx
        return train_idx, val_idx


def _n(name: str) -> int:
    m = re.fullmatch(r"T\d+n(\d+)", name)
    if not m:
        raise ValueError(name)
    return int(m.group(1))


def get_split(name: Optional[str]) -> SplitSpec:
    """Split preset by name.

    ``T367n<n>`` all three blocks, ``T67n<n>`` blocks 1+2 (tested on block 0),
    ``T73n<n>`` block 2 (tested on block 0), ``T6n<n>`` block 1, ``T7n<n>``
    single-block data, ``T2026`` = T367n10,
    ``None``/``"all"`` everything for training and validation.
    """
    if name in (None, "all"):
        return SplitSpec("all")
    if name == "T2026":
        return SplitSpec("T2026", 3, (0, 1, 2), 10, 5, ((0, 5), (1, 5), (2, 5)))
    prefix = name.split("n")[0]
    if prefix == "T367":
        return SplitSpec(name, 3, (0, 1, 2), _n(name), 5, ((0, 5), (1, 5), (2, 5)))
    if prefix == "T67":
        return SplitSpec(name, 3, (1, 2), _n(name), 5, ((0, 15),))
    if prefix == "T73":
        return SplitSpec(name, 3, (2,), _n(name), 5, ((0, 15),))
    if prefix == "T6":
        return SplitSpec(name, 3, (1,), _n(name), 5, ((0, 5),))
    if prefix == "T7":
        return SplitSpec(name, 1, (0,), _n(name), 15, ((0, 15),))
    raise ValueError(f"unknown split '{name}'")


SPLIT_NAMES: Sequence[str] = ("all", "T2026", "T367n<n>", "T67n<n>", "T73n<n>",
                              "T6n<n>", "T7n<n>")
