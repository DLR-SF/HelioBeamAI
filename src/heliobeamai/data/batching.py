"""Context/target split of measurement sequences (as in the original training)."""
from __future__ import annotations

import random
from typing import Tuple

import torch

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _gather(data: torch.Tensor, imgs: torch.Tensor, idx_src: torch.Tensor, idx_trg: torch.Tensor,
            cond_dim: int, input_noise: float) -> Batch:
    b, _, d = data.shape
    src = data.gather(1, idx_src.unsqueeze(-1).expand(-1, -1, d))
    trg = data.gather(1, idx_trg.unsqueeze(-1).expand(-1, -1, d))[..., :cond_dim]
    rows = torch.arange(b, device=data.device)[:, None]
    src_imgs = imgs[rows, idx_src]
    trg_imgs = imgs[rows, idx_trg]
    if input_noise:
        src_imgs = src_imgs + torch.randn_like(src_imgs) * input_noise
    return src, trg, src_imgs, trg_imgs


def split_batch(data: torch.Tensor, imgs: torch.Tensor, mode: str = "train", n_val: int = 2,
                input_noise: float = 0.01, cond_dim: int = 7) -> Batch:
    """Split ``[B, S, 7]`` conditions and ``[B, S, H, W]`` images into context and targets.

    * ``train``: a random number (1 .. S-1) of random targets, the rest is context;
    * ``test``: the first ``n_val`` measurements are targets, the context is the
      rest in random order;
    * ``val``: as ``test`` with the context in stored order.

    Returns ``(context_conditions, target_conditions, context_images, target_images)``;
    Gaussian noise of ``input_noise`` is added to the context images.
    """
    b, s, _ = data.shape
    dev = data.device
    if mode == "train":
        n_trg = random.randint(1, s - 1)
        perm = torch.stack([torch.randperm(s, device=dev) for _ in range(b)])
        idx_trg = perm[:, :n_trg]
        idx_src = perm[:, n_trg:].sort(dim=1).values
    elif mode == "test":
        idx_trg = torch.arange(n_val, device=dev).expand(b, -1)
        idx_src = torch.stack([torch.randperm(s - n_val, device=dev) for _ in range(b)]) + n_val
    elif mode == "val":
        idx_trg = torch.arange(n_val, device=dev).expand(b, -1)
        idx_src = torch.arange(n_val, s, device=dev).expand(b, -1)
    else:
        raise ValueError(f"unknown mode '{mode}'")
    return _gather(data, imgs, idx_src, idx_trg, cond_dim, input_noise)


def split_batch_artificial(data: torch.Tensor, imgs: torch.Tensor, n_targets: int = 20,
                           input_noise: float = 0.01, cond_dim: int = 7) -> Batch:
    """Artificial sequences: the first ``n_targets`` are targets, 1 .. n_targets-1 random others context."""
    b, s, _ = data.shape
    n_src = random.randint(1, n_targets - 1)
    idx_trg = torch.arange(n_targets, device=data.device).expand(b, -1)
    idx_src = torch.stack([torch.randperm(s - n_targets, device=data.device)[:n_src]
                           for _ in range(b)]) + n_targets
    return _gather(data, imgs, idx_src, idx_trg, cond_dim, input_noise)
