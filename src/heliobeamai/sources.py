"""Focal spot sources for :class:`~heliobeamai.HelioBeamField`.

A source turns heliostat geometry into focal spot images; everything after
that (CNN directions, upsampling, tracking broadening, power scaling, rays)
is done once, by the field's :class:`~heliobeamai.RayProjector`, for every
source alike. A field holds an ordered list of sources; each heliostat is
handled by the first source that covers it (e.g. transformer, then Gaussian
fallback).

Contract of a source (:class:`SpotSource`):

* ``name``: unique label (``field.assignment`` reports it).
* ``use_cnn``: run the projection CNN on its spots (otherwise straight rays).
* ``covers(ids) -> list[bool]``: known before any prediction, cheap.
* ``spots(geometry) -> [N, H, W]``: non-negative images in frame C on the
  grid ``geometry.resolution`` x ``geometry.side_lengths`` around the aim
  point, **without** tracking error (the projector adds it). Absolute scale
  is free: rays are scaled to ``powers``; without ``powers`` sources should
  use the transformer's scale (image mean 0.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch

from .inference.conditions import normalize_heliostat_id
from .inference.predictor import FluxPredictor
from .inference.projection import normalize_mean, oblique_gaussian, pixel_origins

SigmaFn = Callable[[List[object], torch.Tensor], torch.Tensor]


@dataclass
class SpotGeometry:
    """What a source gets for one batch of heliostats (all on the field's device)."""
    heliostat_ids: List[object]            # as passed by the caller
    conditions: torch.Tensor               # [N, 7] sun, aim direction, distance / 350 m
    aim_directions: torch.Tensor           # [N, 3] unit vectors heliostat -> aim point
    aim_lengths: torch.Tensor              # [N] heliostat -> aim point (m)
    resolution: Tuple[int, int] = (64, 64)                 # (H, W)
    side_lengths: Tuple[float, float] = (6.0, 6.0)         # (x, z) extent (m)


@runtime_checkable
class SpotSource(Protocol):
    name: str
    use_cnn: bool

    def covers(self, heliostat_ids: Sequence[object]) -> List[bool]: ...

    def spots(self, geometry: SpotGeometry) -> torch.Tensor: ...


class TransformerSource:
    """Transformer focal spots for heliostats with calibration context (encoder cache)."""

    def __init__(self, predictor: FluxPredictor, name: str = "transformer", use_cnn: bool = True) -> None:
        self.predictor = predictor
        self.name = name
        self.use_cnn = use_cnn

    def covers(self, heliostat_ids: Sequence[object]) -> List[bool]:
        return [self.predictor.has_context(h) for h in heliostat_ids]

    def spots(self, geometry: SpotGeometry) -> torch.Tensor:
        spots = self.predictor.predict(list(geometry.heliostat_ids), geometry.conditions[:, None])[:, 0]
        if tuple(spots.shape[-2:]) != tuple(geometry.resolution) or tuple(geometry.side_lengths) != (6.0, 6.0):
            raise ValueError(f"The transformer predicts 64 x 64 px over 6 m x 6 m, the field uses "
                             f"{geometry.resolution} px over {geometry.side_lengths} m.")
        return spots


class GaussianSource:
    """Gaussian beam with a caller-defined width law, projected onto the aperture plane.

    ``sigma_fn(heliostat_ids, distances)`` returns the 1-sigma beam width
    perpendicular to the beam, in metres, ``[N]``, for the heliostat ->
    aim point distances ``[N]`` (m), **without** the tracking error (the
    projector adds it, as for the transformer). The ids are passed as the
    caller gave them. On the aperture plane the spot is::

        exp(-(|p|^2 - (p . a)^2) / (2 sigma^2))

    (p: offset from the aim point in the plane, a: unit aim direction),
    i.e. stretched by ``1 / |a_y|`` along the in-plane incidence direction.
    With tracking broadening ``theta`` (mrad) the projector turns it into the
    same form with ``sqrt(sigma^2 + (d theta)^2)``.

    Args:
        sigma_fn: width law, see above. Constructors for common laws:
            :meth:`field_law`, :meth:`linear`.
        heliostat_ids: heliostats this source covers (``None``: all), e.g. a
            Gaussian fallback restricted to the field configuration.
        name: label in ``field.assignment``.
        use_cnn: send these spots through the projection CNN (default: no,
            straight rays; the CNN was trained on deflectometry spots).
        image_mean: scale of the returned images (transformer scale 0.1).
    """

    def __init__(self, sigma_fn: SigmaFn, heliostat_ids: Optional[Sequence[object]] = None,
                 name: str = "gaussian", use_cnn: bool = False, image_mean: float = 0.1) -> None:
        self.sigma_fn = sigma_fn
        self._ids = None if heliostat_ids is None else {normalize_heliostat_id(h) for h in heliostat_ids}
        self.name = name
        self.use_cnn = use_cnn
        self.image_mean = float(image_mean)

    # -- width laws -----------------------------------------------------------
    @classmethod
    def field_law(cls, sigma_mrad: float, sigma0_m: float = 0.0,
                  per_heliostat_scale: Optional[Mapping[str, float]] = None, **kwargs) -> "GaussianSource":
        """``sigma = s_i * sqrt(sigma0_m^2 + (sigma_mrad/1000 * d)^2)`` (s_i = 1 if not given)."""
        if sigma_mrad < 0 or sigma0_m < 0 or sigma_mrad + sigma0_m <= 0:
            raise ValueError("sigma_mrad and sigma0_m must be >= 0, not both 0")
        scale = {normalize_heliostat_id(k): float(v) for k, v in (per_heliostat_scale or {}).items()}

        def sigma_fn(ids: List[object], d: torch.Tensor) -> torch.Tensor:
            s = torch.tensor([scale.get(normalize_heliostat_id(h), 1.0) for h in ids], dtype=d.dtype,
                             device=d.device)
            return s * torch.sqrt(sigma0_m ** 2 + (sigma_mrad / 1000.0 * d) ** 2)

        return cls(sigma_fn, **kwargs)

    @classmethod
    def linear(cls, parameters: Callable[[List[object]], Tuple[torch.Tensor, torch.Tensor]],
               distance_scale: float = 1.0, **kwargs) -> "GaussianSource":
        """``sigma = sigma_0,i + k_i * d / distance_scale`` with ``parameters(ids) -> (sigma_0, k)``.

        The width law of aimpoint-control's ``HeliostatField`` (without its
        pointing term), e.g. ``GaussianSource.linear(field.get_heliostat_parameters)``.
        """
        if distance_scale <= 0:
            raise ValueError("distance_scale must be positive")

        def sigma_fn(ids: List[object], d: torch.Tensor) -> torch.Tensor:
            sigma_0, k = parameters(list(ids))
            return sigma_0.to(d) + k.to(d) * d / float(distance_scale)

        return cls(sigma_fn, **kwargs)

    # -- source contract --------------------------------------------------------
    def covers(self, heliostat_ids: Sequence[object]) -> List[bool]:
        if self._ids is None:
            return [True] * len(heliostat_ids)
        return [normalize_heliostat_id(h) in self._ids for h in heliostat_ids]

    def sigma(self, heliostat_ids: Sequence[object], distances: torch.Tensor) -> torch.Tensor:
        """Validated beam widths ``[N]`` (m)."""
        d = distances.reshape(-1)
        sigma = torch.as_tensor(self.sigma_fn(list(heliostat_ids), d)).to(d).reshape(-1)
        if sigma.shape[0] != d.shape[0]:
            raise ValueError(f"sigma_fn returned {sigma.shape[0]} widths for {d.shape[0]} heliostats")
        if not (torch.isfinite(sigma).all() and (sigma > 0).all()):
            raise ValueError("sigma_fn must return finite, positive widths")
        return sigma

    def spots(self, geometry: SpotGeometry) -> torch.Tensor:
        h, w = geometry.resolution
        d = geometry.aim_lengths.reshape(-1)
        p = pixel_origins(h, w, geometry.side_lengths, d.device, d.dtype)       # frame C, row-major
        dx, dz = p[:, 0].reshape(h, w), p[:, 2].reshape(h, w)
        spots = oblique_gaussian(dx, dz, geometry.aim_directions, self.sigma(geometry.heliostat_ids, d))
        return normalize_mean(spots, self.image_mean)
