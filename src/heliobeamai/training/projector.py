"""Training of the projection CNN (:class:`FocalSpotToDirectionsNet`).

The CNN learns per-pixel ray direction deviations such that rays started on
the aim plane (plane 0) reproduce the ray-traced flux on planes further along
the aim direction. Data format: ``docs/data.md``, "Projection dataset".

Everything is in image frame C (see :mod:`heliobeamai.inference.projection`)
and uses the same ray generation as inference (:class:`RayProjector`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from ..checkpoints import save_projector
from ..inference.projection import ProjectionConfig, RayProjector
from ..models.projector import FocalSpotToDirectionsNet

PathLike = Union[str, Path]
PROJECTION_DATA_FORMAT = "heliobeamai.projection_data/1"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass
class ProjectionData:
    """Ray-traced plane fluxes of N heliostat/sun/aim cases, frame C.

    Attributes:
        fluxes: ``[N, P, H, W]``; plane 0 is the aim plane (the focal spot),
            plane k lies ``k * plane_spacing`` further along the aim direction.
        plane_centers: ``[N, P, 3]``; plane 0 centre is the aim point.
        heliostat_positions: ``[N, 3]``.
        sun_vectors: ``[N, 3]`` unit vectors towards the sun (None for old data).
        heliostat_names: N names (None for old data).
    """
    fluxes: torch.Tensor
    plane_centers: torch.Tensor
    heliostat_positions: torch.Tensor
    sun_vectors: Optional[torch.Tensor] = None
    heliostat_names: Optional[List[str]] = None
    side_lengths: Tuple[float, float] = (6.0, 6.0)

    def __len__(self) -> int:
        return self.fluxes.shape[0]

    def subset(self, idx) -> "ProjectionData":
        if isinstance(idx, slice):
            names = self.heliostat_names[idx] if self.heliostat_names else None
        else:
            idx = torch.as_tensor(idx, dtype=torch.long)
            names = [self.heliostat_names[i] for i in idx.tolist()] if self.heliostat_names else None
        return replace(self, fluxes=self.fluxes[idx], plane_centers=self.plane_centers[idx],
                       heliostat_positions=self.heliostat_positions[idx],
                       sun_vectors=None if self.sun_vectors is None else self.sun_vectors[idx],
                       heliostat_names=names)

    def to(self, device) -> "ProjectionData":
        return replace(self, fluxes=self.fluxes.to(device), plane_centers=self.plane_centers.to(device),
                       heliostat_positions=self.heliostat_positions.to(device),
                       sun_vectors=None if self.sun_vectors is None else self.sun_vectors.to(device))

    def conditions(self, distance_normalization_m: float = 350.0) -> torch.Tensor:
        """Transformer conditions ``[N, 7]``; sun columns are zero for old data."""
        aim = self.plane_centers[:, 0] - self.heliostat_positions
        length = aim.norm(dim=1, keepdim=True)
        sun = self.sun_vectors if self.sun_vectors is not None else torch.zeros_like(aim)
        return torch.cat([F.normalize(sun, dim=1) if self.sun_vectors is not None else sun,
                          aim / length, length / distance_normalization_m], dim=1)

    def heliostat_keys(self) -> List[str]:
        if self.heliostat_names is not None:
            return list(self.heliostat_names)
        return [f"{x:.1f},{y:.1f}" for x, y in self.heliostat_positions[:, :2].tolist()]

    # -- loading -------------------------------------------------------------
    @classmethod
    def load(cls, files: Sequence[PathLike]) -> "ProjectionData":
        """Load files written by the projection data generator (format v1)."""
        parts = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
        for p, f in zip(parts, files):
            if p.get("format") != PROJECTION_DATA_FORMAT or p.get("frame") != "C":
                raise ValueError(f"{f} is not a {PROJECTION_DATA_FORMAT} file in frame C")
        names = [n for p in parts for n in p["heliostat_names"]]
        return cls(torch.cat([p["fluxes"].float() for p in parts]),
                   torch.cat([p["plane_centers"].float() for p in parts]),
                   torch.cat([p["heliostat_positions"].float() for p in parts]),
                   torch.cat([p["sun_vectors"].float() for p in parts]), names,
                   tuple(parts[0].get("side_lengths_m", (6.0, 6.0))))

    @classmethod
    def from_legacy_raytracer_files(cls, flux_files: Sequence[PathLike],
                                    cond_files: Sequence[PathLike]) -> "ProjectionData":
        """Old ``targets_all_*.pt`` / ``cond_data_*.pt`` pairs (no sun vectors).

        Those images are in the ray-tracer frame (rows +z, columns +x); both
        axes are flipped to frame C. They were made before the ray-tracer fix
        of 2026-11-04 and are kept for tests only.
        """
        fluxes, centers, positions = [], [], []
        for ff, cf in zip(flux_files, cond_files, strict=True):
            cond = torch.load(cf, map_location="cpu", weights_only=False)
            fluxes.append(torch.load(ff, map_location="cpu", weights_only=False).float().flip(-1).flip(-2))
            centers.append(cond["evalPoints_all"].float())
            positions.append(cond["hp_all"].float())
        return cls(torch.cat(fluxes), torch.cat(centers), torch.cat(positions))

    # -- preprocessing -------------------------------------------------------
    def smoothed(self, sigma_px: float, kernel: int = 5) -> "ProjectionData":
        if sigma_px <= 0:
            return self
        ax = torch.arange(kernel, dtype=torch.float32) - kernel // 2
        g = torch.exp(-(ax[:, None] ** 2 + ax[None] ** 2) / (2 * sigma_px ** 2))
        g = (g / g.sum())[None, None]
        n, p, h, w = self.fluxes.shape
        f = F.conv2d(self.fluxes.reshape(n * p, 1, h, w), g.to(self.fluxes), padding=kernel // 2)
        return replace(self, fluxes=f.reshape(n, p, h, w))

    def with_vae_input(self, vae, batch_size: int = 256) -> "ProjectionData":
        """Replace the aim-plane spot by its VAE reconstruction (as the transformer outputs)."""
        spots = self.fluxes[:, 0]
        out = []
        with torch.no_grad():
            for i in range(0, len(spots), batch_size):
                s = spots[i:i + batch_size].unsqueeze(1)
                m = s.mean(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
                rec = vae.from_latent(vae.to_latent((0.1 * s / m).to(next(vae.parameters()).device)))
                out.append((rec.cpu().clamp_min(0) * m / 0.1).squeeze(1))
        fluxes = self.fluxes.clone()
        fluxes[:, 0] = torch.cat(out)
        return replace(self, fluxes=fluxes)


def split_by_heliostat(data: ProjectionData, val_fraction: float, seed: int = 0
                       ) -> Tuple[ProjectionData, ProjectionData]:
    """Train/validation split without sharing heliostats."""
    keys = data.heliostat_keys()
    unique = sorted(set(keys))
    g = torch.Generator().manual_seed(seed)
    order = [unique[i] for i in torch.randperm(len(unique), generator=g).tolist()]
    val_keys = set(order[:int(round(val_fraction * len(unique)))])
    val = [i for i, k in enumerate(keys) if k in val_keys]
    train = [i for i, k in enumerate(keys) if k not in val_keys]
    return data.subset(train), data.subset(val)


# --------------------------------------------------------------------------
# projection onto planes
# --------------------------------------------------------------------------
def bin_on_planes(origins: torch.Tensor, directions: torch.Tensor, energies: torch.Tensor,
                  plane_centers: torch.Tensor, bins: int,
                  side_lengths: Tuple[float, float] = (6.0, 6.0)) -> torch.Tensor:
    """Bilinear binning of rays ``[B, N, 3]`` on x-z planes -> ``[B, P, bins, bins]`` (frame C).

    Rays parallel to the planes or hitting them behind their origin are ignored.
    """
    b, n, _ = origins.shape
    p = plane_centers.shape[1]
    dy = directions[..., 1][:, None, :]
    ok = dy.abs() > 1e-6
    t = (plane_centers[..., 1][:, :, None] - origins[..., 1][:, None, :]) / torch.where(ok, dy, torch.ones_like(dy))
    valid = ok & (t >= 0)
    rel = origins[:, None] + t[..., None] * directions[:, None] - plane_centers[:, :, None, :]
    col = (side_lengths[0] / 2 - rel[..., 0]) / side_lengths[0] * bins - 0.5
    row = (side_lengths[1] / 2 - rel[..., 2]) / side_lengths[1] * bins - 0.5
    c0, r0 = torch.floor(col), torch.floor(row)
    wc1, wr1 = col - c0, row - r0
    out = torch.zeros(b * p, bins * bins, device=origins.device, dtype=energies.dtype)
    e = (energies[:, None, :] * valid).reshape(b * p, n)
    for dc, wc in ((0, 1 - wc1), (1, wc1)):
        for dr, wr in ((0, 1 - wr1), (1, wr1)):
            cc, rr = c0 + dc, r0 + dr
            inside = ((cc >= 0) & (cc < bins) & (rr >= 0) & (rr < bins)).reshape(b * p, n)
            idx = (rr.clamp(0, bins - 1) * bins + cc.clamp(0, bins - 1)).long().reshape(b * p, n)
            out.scatter_add_(1, idx, e * (wc * wr).reshape(b * p, n) * inside)
    return out.view(b, p, bins, bins)


def relative_flux_deviation(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean-normalized absolute deviation per image (``[..., H, W]`` -> ``[...]``)."""
    target = target / (target.mean(dim=(-1, -2), keepdim=True) + eps)
    pred = pred / (pred.mean(dim=(-1, -2), keepdim=True) + eps)
    return (target - pred).abs().mean(dim=(-1, -2))


def plane_projections(projector: RayProjector, data: ProjectionData) -> torch.Tensor:
    """Project the aim-plane spots onto planes 1.. -> ``[B, P-1, H, W]``."""
    spots = data.fluxes[:, 0]
    cond = data.conditions(projector.config.distance_normalization_m)
    s, dev = projector.prepare(spots, cond, data.side_lengths)
    rays = projector.rays_from_prepared(s, dev, data.plane_centers[:, 0], cond,
                                        side_lengths=data.side_lengths)
    rays = rays.view(len(data), -1, 7)
    up = s.shape[-1] // spots.shape[-1]
    proj = bin_on_planes(rays[..., :3], rays[..., 3:6], rays[..., 6], data.plane_centers[:, 1:],
                         spots.shape[-1] * up, data.side_lengths)
    return F.avg_pool2d(proj, up) if up > 1 else proj


def projection_loss(projector: RayProjector, data: ProjectionData, grid: Optional[int] = None,
                    projections: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-case loss: mean over planes k of ``relative_flux_deviation / k``.

    ``grid`` evaluates on a coarser square grid (both images average-pooled),
    e.g. to compare ray densities on receiver-like bin sizes.
    """
    proj = plane_projections(projector, data) if projections is None else projections
    target = data.fluxes[:, 1:]
    if grid is not None and grid < target.shape[-1]:
        k = target.shape[-1] // grid
        proj, target = F.avg_pool2d(proj, k), F.avg_pool2d(target, k)
    dev = relative_flux_deviation(target, proj)
    w = 1.0 / torch.arange(1, dev.shape[1] + 1, device=dev.device, dtype=dev.dtype)
    return (dev * w).mean(dim=1)


@torch.no_grad()
def evaluate_projector(projector: RayProjector, data: ProjectionData, batch_size: int = 16,
                       grids: Optional[Sequence[int]] = None):
    """Per-case projection loss (projector without CNN: straight rays).

    With ``grids`` returns ``{grid: per-case loss}``.
    """
    if projector.net is not None:
        projector.net.eval()
    keys = list(grids) if grids else [None]
    out = {g: [] for g in keys}
    for i in range(0, len(data), batch_size):
        batch = data.subset(slice(i, i + batch_size)).to(projector.device)
        proj = plane_projections(projector, batch)
        for g in keys:
            out[g].append(projection_loss(projector, batch, g, proj).cpu())
    out = {g: torch.cat(v) for g, v in out.items()}
    return out if grids else out[None]


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
@dataclass
class ProjectorTrainConfig:
    data_files: List[str] = field(default_factory=list)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    smooth_sigma_px: float = 1.0
    # standalone VAE checkpoint (e.g. z64CH4) whose round trip shapes the CNN input like a
    # transformer output; not the VAE inside a transformer checkpoint: that one was trained
    # jointly and its encoder -> decoder round trip is not an autoencoder any more
    input_vae_checkpoint: Optional[str] = None
    val_fraction: float = 0.1
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 8
    grad_clip: float = 1.0
    lr_patience: int = 5
    lr_factor: float = 0.5
    dropout: float = 0.01
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output: str = "runs/projector/projector.pt"


def train_projector(cfg: ProjectorTrainConfig, data: Optional[ProjectionData] = None
                    ) -> FocalSpotToDirectionsNet:
    torch.manual_seed(cfg.seed)
    data = (data or ProjectionData.load(cfg.data_files)).smoothed(cfg.smooth_sigma_px)
    if cfg.input_vae_checkpoint:
        from ..checkpoints import load_vae
        data = data.with_vae_input(load_vae(cfg.input_vae_checkpoint).to(cfg.device))
    if cfg.projection.conditions == "full" and data.sun_vectors is None:
        raise ValueError("conditions='full' needs data with sun vectors")
    train, val = split_by_heliostat(data, cfg.val_fraction, cfg.seed)
    device = torch.device(cfg.device)
    net = FocalSpotToDirectionsNet(dropout_rate=cfg.dropout, cond_dim=cfg.projection.cond_dim).to(device)
    projector = RayProjector(replace(cfg.projection, model_gradients=True, pointing_accuracy_mrad=0.0),
                             net, device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=cfg.lr_factor, patience=cfg.lr_patience)
    best = math.inf
    for epoch in range(cfg.epochs):
        net.train()
        order = torch.randperm(len(train))
        total = 0.0
        for i in range(0, len(train), cfg.batch_size):
            batch = train.subset(order[i:i + cfg.batch_size]).to(device)
            loss = projection_loss(projector, batch).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(batch)
        train_loss = total / len(train)
        val_loss = evaluate_projector(projector, val).mean().item() if len(val) else train_loss
        sched.step(val_loss)
        print(f"epoch {epoch + 1}/{cfg.epochs}  train {train_loss:.5f}  val {val_loss:.5f}", flush=True)
        if val_loss < best:
            best = val_loss
            save_projector(net, cfg.output, replace(cfg.projection, model_gradients=False),
                           epoch=epoch + 1, val_loss=val_loss, train_loss=train_loss,
                           smooth_sigma_px=cfg.smooth_sigma_px,
                           input_vae_checkpoint=cfg.input_vae_checkpoint)
    return net
