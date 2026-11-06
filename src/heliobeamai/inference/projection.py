"""Turn predicted focal spots into ray bundles for receiver ray tracing.

Image frame C (camera view, the frame of the calibration images and the
transformer): an image ``[row, col]`` covers an x-z plane facing +y;
row 0 is the top (highest z), column 0 is the east edge (highest x), i.e.
rows run along -z and columns along -x. Pixel ``(r, c)`` is centred at::

    x = aim_x + (side_x / 2 - (c + 0.5) * side_x / W)
    z = aim_z + (side_z / 2 - (r + 0.5) * side_z / H)

Every pixel becomes one ray ``(ox, oy, oz, dx, dy, dz, energy[, group])``
starting on the plane through the aim point. Its direction is the aim
direction plus the per-pixel deviation predicted by
:class:`~heliobeamai.models.FocalSpotToDirectionsNet`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from ..checkpoints import load_projector
from ..models.projector import FocalSpotToDirectionsNet

CONDITION_MODES = {"full": slice(0, 7), "aim": slice(3, 7)}
UPSAMPLING_MODES = ("rays", "cnn")
BROADENING_MODES = ("oblique", "isotropic")


@dataclass
class ProjectionConfig:
    """Settings of the focal spot -> ray projection (stored with CNN weights).

    Attributes:
        conditions: CNN conditions taken from the 7 transformer conditions
            (sun 3, aim direction 3, distance / distance_normalization_m):
            ``"full"`` all 7, ``"aim"`` the last 4.
        distance_normalization_m: distance normalization of the conditions
            (the transformer's 350 m).
        image_mean: CNN input images are scaled to this mean (as in training).
        deviation_scale: factor on the CNN output before it is added to the
            aim direction.
        upsampling: integer factor for more rays per spot.
        upsampling_mode: ``"rays"``: CNN on the native image, spot and
            deviation field interpolated afterwards; ``"cnn"``: CNN on the
            upsampled image.
        pointing_accuracy_mrad: 1-sigma tracking error (beam direction) used
            to broaden spots.
        broadening: ``"oblique"``: the tracking error displaces the beam by
            ``d * theta`` perpendicular to the beam; on the aperture plane this
            is stretched by ``1 / |a_y|`` along the in-plane incidence
            direction (same projection as :func:`oblique_gaussian`).
            ``"isotropic"``: ``d * theta`` in both image axes (the behaviour
            of aimpoint-control before HelioBeamAI).
        model_gradients: keep the CNN differentiable.
    """
    conditions: str = "full"
    distance_normalization_m: float = 350.0
    image_mean: float = 0.1
    deviation_scale: float = 1e-2
    upsampling: int = 1
    upsampling_mode: str = "rays"
    pointing_accuracy_mrad: float = 0.0
    broadening: str = "oblique"
    model_gradients: bool = False

    def __post_init__(self) -> None:
        if self.conditions not in CONDITION_MODES:
            raise ValueError(f"conditions must be one of {list(CONDITION_MODES)}")
        if self.upsampling_mode not in UPSAMPLING_MODES:
            raise ValueError(f"upsampling_mode must be one of {UPSAMPLING_MODES}")
        if self.broadening not in BROADENING_MODES:
            raise ValueError(f"broadening must be one of {BROADENING_MODES}")

    @property
    def cond_dim(self) -> int:
        s = CONDITION_MODES[self.conditions]
        return s.stop - s.start


def _check_finite(t: torch.Tensor, what: str) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"{what} contain {int((~torch.isfinite(t)).sum())} non-finite values.")


# --------------------------------------------------------------------------
# image operations (all in frame C, shape [N, H, W] or [N, C, H, W])
# --------------------------------------------------------------------------
def upsample(images: torch.Tensor, factor: int) -> torch.Tensor:
    """Bilinear upsampling of ``[N, H, W]`` or ``[N, C, H, W]`` (mean preserved)."""
    if factor <= 1:
        return images
    x = images.unsqueeze(1) if images.ndim == 3 else images
    up = F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)
    return up.squeeze(1) if images.ndim == 3 else up


def normalize_mean(images: torch.Tensor, mean: float) -> torch.Tensor:
    m = images.mean(dim=(-1, -2), keepdim=True)
    return images * (mean / torch.where(m.abs() < 1e-12, torch.ones_like(m), m))


def oblique_gaussian(dx: torch.Tensor, dz: torch.Tensor, aim_directions: torch.Tensor,
                     sigma: torch.Tensor) -> torch.Tensor:
    """Gaussian beam of perpendicular width ``sigma`` seen on the aperture plane.

    ``dx``, ``dz`` ``[H, W]``: in-plane offsets (m, or pixels with ``sigma``
    in pixels) from the beam axis on the plane facing +y; ``aim_directions``
    ``[N, 3]`` unit beam directions; ``sigma`` ``[N]``. Returns ``[N, H, W]``
    (unnormalized)::

        exp(-(|p|^2 - (p . a)^2) / (2 sigma^2)),   p = (dx, 0, dz)

    i.e. the squared distance of ``p`` from the beam axis, which stretches the
    spot by ``1 / |a_y|`` along the in-plane incidence direction. The form is
    even in ``p``, so it does not depend on the sign convention of the image
    axes (frame C flips both).
    """
    a = aim_directions.to(dx)
    along = dx[None] * a[:, 0, None, None] + dz[None] * a[:, 2, None, None]
    r2 = (dx[None] ** 2 + dz[None] ** 2 - along ** 2).clamp_min(0.0)
    return torch.exp(-0.5 * r2 / sigma.to(dx).clamp_min(1e-12)[:, None, None] ** 2)


def gaussian_kernels(sigma_px_x: torch.Tensor, sigma_px_z: torch.Tensor, size: int,
                     aim_directions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
    """Per-spot normalized Gaussian kernels ``[N, 1, k, k]`` and their radius.

    Without ``aim_directions`` axis-aligned with widths ``sigma_px_x/z``; with
    them the oblique beam kernel (:func:`oblique_gaussian`, square pixels,
    width ``sigma_px_x``).
    """
    if aim_directions is None:
        stretch = 1.0
    else:
        stretch = 1.0 / max(float(aim_directions[:, 1].abs().min()), 0.05)
    radius = min(max(1, int(math.ceil(3.0 * stretch * float(torch.maximum(sigma_px_x, sigma_px_z).max())))),
                 size)
    coords = torch.arange(-radius, radius + 1, device=sigma_px_x.device, dtype=sigma_px_x.dtype)
    zg, xg = torch.meshgrid(coords, coords, indexing="ij")
    if aim_directions is None:
        k = torch.exp(-0.5 * ((xg / sigma_px_x.clamp_min(1e-6)[:, None, None]) ** 2
                              + (zg / sigma_px_z.clamp_min(1e-6)[:, None, None]) ** 2))
    else:
        k = oblique_gaussian(xg, zg, aim_directions, sigma_px_x.clamp_min(1e-6))
    return (k / k.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)).unsqueeze(1), radius


def broaden(spots: torch.Tensor, deviations: Optional[torch.Tensor], aim_lengths: torch.Tensor,
            side_lengths: Tuple[float, float], pointing_accuracy_mrad: float,
            aim_directions: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Tracking error: shift-average spot and deviation field with one Gaussian.

    With ``aim_directions`` ``[N, 3]`` the kernel is the oblique beam kernel
    (``ProjectionConfig.broadening == "oblique"``; needs square pixels),
    otherwise isotropic in the image plane.

    The tracking error moves the whole beam, including its directions, so the
    broadened deviation field is the energy-weighted average
    ``K * (E D) / K * E``. Energy is preserved. The kernel width is detached
    (gradients through it are not needed and numerically brittle).
    """
    if pointing_accuracy_mrad <= 0.0 or spots.numel() == 0:
        return spots, deviations
    n, h, w = spots.shape
    lengths = aim_lengths.reshape(-1).to(spots).detach().abs()
    _check_finite(lengths, "Aim lengths")
    sigma_m = lengths * pointing_accuracy_mrad / 1000.0
    sx = sigma_m / (float(side_lengths[0]) / w)
    sz = sigma_m / (float(side_lengths[1]) / h)
    if float(torch.maximum(sx, sz).max()) <= 1e-6:
        return spots, deviations
    if aim_directions is not None:
        if not torch.allclose(sx, sz, rtol=1e-4):
            raise ValueError("Oblique broadening needs square pixels.")
        aim_directions = F.normalize(aim_directions.to(spots).detach(), dim=-1)
    kernels, r = gaussian_kernels(sx, sz, max(h, w), aim_directions)

    def conv(x: torch.Tensor) -> torch.Tensor:          # x [N, C, H, W], one kernel per spot
        c = x.shape[1]
        y = F.conv2d(F.pad(x.reshape(1, n * c, h, w), (r,) * 4),
                     kernels.repeat_interleave(c, dim=0), groups=n * c)
        return y.reshape(n, c, h, w)

    b = conv(spots.unsqueeze(1)).squeeze(1)
    b = b * (spots.sum(dim=(-2, -1), keepdim=True) / b.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
    if deviations is not None:
        weighted = conv(deviations * spots.unsqueeze(1))
        deviations = weighted / conv(spots.unsqueeze(1)).clamp_min(1e-12)
    return b, deviations


def scale_to_powers(spots: torch.Tensor, powers: Optional[torch.Tensor],
                    side_lengths: Tuple[float, float]) -> torch.Tensor:
    """Scale flux densities so each spot integrates to its power."""
    if powers is None:
        return spots
    h, w = spots.shape[-2:]
    current = spots.sum(dim=(-2, -1)) * float(side_lengths[0]) * float(side_lengths[1]) / (h * w)
    return spots * (powers.to(spots) / current.clamp_min(1e-12))[:, None, None]


def pixel_origins(h: int, w: int, side_lengths: Tuple[float, float], device=None,
                  dtype=torch.float32) -> torch.Tensor:
    """Ray origins ``[H*W, 3]`` relative to the aim point, frame C, row-major."""
    sx, sz = float(side_lengths[0]), float(side_lengths[1])
    x = sx / 2 - (torch.arange(w, device=device, dtype=dtype) + 0.5) * sx / w
    z = sz / 2 - (torch.arange(h, device=device, dtype=dtype) + 0.5) * sz / h
    zg, xg = torch.meshgrid(z, x, indexing="ij")
    return torch.stack([xg.flatten(), torch.zeros_like(xg.flatten()), zg.flatten()], dim=-1)


def ray_directions(aim_directions: torch.Tensor, deviations: Optional[torch.Tensor],
                   deviation_scale: float, shape: Tuple[int, int]) -> torch.Tensor:
    """Unit directions ``[N, 3, H, W]``: aim direction + scaled deviation."""
    base = aim_directions[:, :, None, None]
    if deviations is None:
        return base.expand(-1, 3, *shape)
    d = base + deviations * float(deviation_scale)
    _check_finite(d, "Ray directions")
    norm = torch.linalg.vector_norm(d, dim=1, keepdim=True)
    d = torch.where((torch.isfinite(norm) & (norm > 1e-6)).expand_as(d), d / norm.clamp_min(1e-6), base)
    return F.normalize(d, p=2, dim=1, eps=1e-6)


# --------------------------------------------------------------------------
class RayProjector(nn.Module):
    """Focal spots (frame C) -> rays, optionally with CNN direction deviations.

    ``conditions`` are the transformer conditions ``[N, 7]``
    (sun, aim direction, distance / distance_normalization_m), e.g.
    ``build_conditions(...)[0][:, 0]``. Without a CNN every ray runs along
    the aim direction.
    """

    def __init__(self, config: Optional[ProjectionConfig] = None,
                 net: Optional[FocalSpotToDirectionsNet] = None,
                 device: Union[str, torch.device] = "cpu") -> None:
        super().__init__()
        self.config = config or ProjectionConfig()
        self.device = torch.device(device)
        self.net = net.to(self.device).eval() if net is not None else None
        if self.net is not None and self.net.conv1.in_channels - 1 != self.config.cond_dim:
            raise ValueError(f"CNN expects {self.net.conv1.in_channels - 1} conditions, "
                             f"config '{self.config.conditions}' gives {self.config.cond_dim}")

    @classmethod
    def from_checkpoint(cls, path: Optional[Union[str, Path]],
                        config: Optional[ProjectionConfig] = None,
                        device: Union[str, torch.device] = "cpu", **overrides) -> "RayProjector":
        """Load the CNN with the settings stored in its checkpoint.

        ``overrides`` change single fields, e.g. ``pointing_accuracy_mrad=1.8``.
        """
        if path is None:
            return cls(replace(config or ProjectionConfig(), **overrides), None, device)
        net, stored = load_projector(path, with_config=True)
        cfg = config or ProjectionConfig(**stored)
        return cls(replace(cfg, **overrides), net, device)

    # -- pieces --------------------------------------------------------------
    def aim_geometry(self, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aim directions ``[N, 3]`` and lengths ``[N, 1]`` (m) from the conditions."""
        dirs = F.normalize(conditions[:, 3:6], dim=-1)
        return dirs, conditions[:, 6:7] * float(self.config.distance_normalization_m)

    def deviations(self, spots: torch.Tensor, conditions: torch.Tensor) -> Optional[torch.Tensor]:
        """CNN deviations ``[N, 3, H, W]`` for spots ``[N, H, W]`` (None without CNN)."""
        if self.net is None or spots.shape[0] == 0:
            return None
        _check_finite(spots, "Focal spots")
        _check_finite(conditions, "Conditions")
        x = normalize_mean(spots.clamp_min(0.0), self.config.image_mean).unsqueeze(1)
        cond = conditions[:, CONDITION_MODES[self.config.conditions]]
        if self.config.model_gradients:
            return self.net(x, cond)
        with torch.no_grad():
            return self.net(x, cond)

    def prepare(self, focal_spots: torch.Tensor, conditions: torch.Tensor,
                side_lengths: Tuple[float, float] = (6.0, 6.0), use_cnn: bool = True
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Spots and deviations ready for ray generation (cacheable).

        Order: CNN on the sharp spot (skipped with ``use_cnn=False``: straight
        rays), tracking broadening, upsampling. In ``upsampling_mode="rays"``
        the broadening runs on the native grid (both steps are linear and the
        kernel is several pixels wide, so the order does not matter, but the
        convolution is ``upsampling^4`` times cheaper); in ``"cnn"`` mode the
        CNN needs the upsampled spot and broadening follows on that grid.
        """
        spots = focal_spots.to(self.device).clamp_min(0.0)
        conditions = conditions.to(self.device, spots.dtype)
        up, mode = int(self.config.upsampling), self.config.upsampling_mode
        dirs, lengths = self.aim_geometry(conditions)
        pointing = float(self.config.pointing_accuracy_mrad)
        oblique = dirs if self.config.broadening == "oblique" else None
        if up > 1 and mode == "cnn":
            spots = upsample(spots, up)
        dev = self.deviations(spots, conditions) if use_cnn else None
        spots, dev = broaden(spots, dev, lengths, side_lengths, pointing, oblique)
        if up > 1 and mode == "rays":
            if dev is not None:      # interpolate the energy-weighted deviation field
                e = upsample(spots, up).unsqueeze(1)
                dev = upsample(dev * spots.unsqueeze(1), up) / e.clamp_min(1e-12)
            spots = upsample(spots, up)
        return spots, dev

    def rays_from_prepared(self, spots: torch.Tensor, deviations: Optional[torch.Tensor],
                           aim_points: torch.Tensor, conditions: torch.Tensor,
                           powers: Optional[torch.Tensor] = None,
                           side_lengths: Tuple[float, float] = (6.0, 6.0),
                           group_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        n, h, w = spots.shape
        dirs, _ = self.aim_geometry(conditions.to(spots))
        energy = scale_to_powers(spots, powers, side_lengths)
        directions = ray_directions(dirs, deviations, self.config.deviation_scale, (h, w))
        origins = pixel_origins(h, w, side_lengths, spots.device, spots.dtype)[None] \
            + aim_points.to(spots)[:, None, :]
        energies = energy.flatten(start_dim=1) * float(side_lengths[0]) * float(side_lengths[1]) / (h * w)
        rays = torch.cat([origins.reshape(-1, 3),
                          directions.permute(0, 2, 3, 1).reshape(-1, 3),
                          energies.reshape(-1, 1)], dim=-1)
        if group_indices is not None:
            groups = group_indices.to(spots.device).unsqueeze(1).expand(-1, h * w)
            rays = torch.cat([rays, groups.reshape(-1, 1).to(rays.dtype)], dim=-1)
        return rays

    def forward(self, focal_spots: torch.Tensor, aim_points: torch.Tensor, conditions: torch.Tensor,
                powers: Optional[torch.Tensor] = None,
                side_lengths: Tuple[float, float] = (6.0, 6.0),
                group_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rays ``[N*H*W, 7]`` (``8`` with ``group_indices``) for spots ``[N, H, W]``."""
        if focal_spots.numel() == 0:
            return torch.empty((0, 8 if group_indices is not None else 7), device=self.device)
        spots, dev = self.prepare(focal_spots, conditions, side_lengths)
        return self.rays_from_prepared(spots, dev, aim_points, conditions, powers, side_lengths,
                                       group_indices)
