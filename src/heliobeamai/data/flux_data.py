"""Calibration flux datasets and the Lightning data module for transformer training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .splits import SplitSpec, get_split

PathLike = Union[str, Path]


def load_flux_dataset(path: PathLike) -> Dict:
    """Load a dataset file (keys ``imgs [H,S,64,64]``, ``data [H,S,7]``, ``ids`` ...)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def normalized(data: torch.Tensor, distance_normalization_m: float) -> torch.Tensor:
    """Conditions with the distance column divided by the normalization."""
    data = data.clone().float()
    data[..., 6] /= distance_normalization_m
    return data


class FluxSequenceDataset(Dataset):
    """One item = all selected measurements of one heliostat.

    With ``artificial`` (dict with ``data``/``imgs``) every item also carries a
    random artificial sequence (``data_art``/``imgs_art``).
    """

    def __init__(self, data: torch.Tensor, imgs: torch.Tensor,
                 artificial: Optional[Dict[str, torch.Tensor]] = None) -> None:
        self.data, self.imgs = data, imgs
        self.artificial = artificial

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"data": self.data[idx], "imgs": self.imgs[idx]}
        if self.artificial is not None:
            j = int(torch.randint(len(self.artificial["data"]), (1,)))
            item["data_art"] = self.artificial["data"][j]
            item["imgs_art"] = self.artificial["imgs"][j]
        return item


def load_artificial(path: PathLike, distance_normalization_m: float = 350.0) -> Dict[str, torch.Tensor]:
    """Artificial (ray-traced) sequences; long sequences are reduced as in the original code."""
    raw = load_flux_dataset(path)
    data, imgs = normalized(raw["data"], distance_normalization_m), raw["imgs"].float()
    if data.shape[1] > 40:
        idx = torch.cat([torch.arange(10), torch.arange(20, 30), torch.arange(40, 60)])
        data, imgs = data[:, idx], imgs[:, idx]
    return {"data": data, "imgs": imgs}


@dataclass
class FluxDataConfig:
    path: str
    split: Optional[str] = None
    batch_size: int = 16
    num_workers: int = 4
    distance_normalization_m: float = 350.0
    artificial_path: Optional[str] = None


def _lightning():
    import pytorch_lightning as pl
    return pl


class FluxDataModule(_lightning().LightningDataModule):
    def __init__(self, cfg: FluxDataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.split: SplitSpec = get_split(cfg.split)

    def setup(self, stage: Optional[str] = None) -> None:
        raw = load_flux_dataset(self.cfg.path)
        data = normalized(raw["data"], self.cfg.distance_normalization_m)
        imgs = raw["imgs"].float()
        train_idx, val_idx = self.split.indices(data.shape[1])
        art = (load_artificial(self.cfg.artificial_path, self.cfg.distance_normalization_m)
               if self.cfg.artificial_path else None)
        self.train_set = FluxSequenceDataset(data[:, train_idx], imgs[:, train_idx], art)
        self.val_set = FluxSequenceDataset(data[:, val_idx], imgs[:, val_idx])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.cfg.batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers)
