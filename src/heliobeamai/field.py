"""One-object interface for receiver flux simulation (e.g. aimpoint optimization).

    field = HelioBeamField.from_artifacts(device="cuda", pointing_accuracy_mrad=1.8,
                                          fallback=GaussianSource.linear(params))
    field.assignment(all_ids)                                   # who is modelled how (at setup)
    prepared = field.prepare(ids, sun, positions, aim_points)   # once per geometry refresh
    rays = field.rays(prepared, aim_points, powers)             # every iteration, differentiable in aim_points

Without ``fallback`` only the transformer is used and heliostats without
calibration context raise an error (e.g. for transformer-only experiments).

Coordinates (see docs/data.md, "Coordinates and image frame"):
tower-base ENU in metres (x east, y north, z up, z = 0 at the tower base,
i.e. calibration-database U - 87 m); ``sun`` is the unit vector towards the
sun in the same axes; aim points lie on the plane through the aim point
facing +y (receiver aperture plane y = 0 in aimpoint-control).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from . import artifacts
from .inference.conditions import build_conditions, normalize_heliostat_id
from .inference.predictor import FluxPredictor
from .inference.projection import RayProjector
from .sources import GaussianSource, SpotGeometry, SpotSource, TransformerSource

PathLike = Union[str, Path]
UNCOVERED = "uncovered"


@dataclass
class PreparedSpots:
    """Focal spots ready for ray generation (frozen between geometry refreshes)."""
    heliostat_ids: List[str]
    spots: torch.Tensor                    # [N, H, W] frame C, after CNN input, upsampling, broadening
    deviations: Optional[torch.Tensor]     # [N, 3, H, W] CNN direction deviations (None: straight rays)
    sun: torch.Tensor                      # [3] or [N, 3]
    positions: torch.Tensor                # [N, 3]
    sources: Optional[List[str]] = None    # source name per heliostat


class HelioBeamField:
    """Focal spot sources + one projection chain -> rays.

    Args:
        sources: one source or an ordered list; every heliostat is handled
            by the first source that covers it. Heliostats no source covers
            raise a ``KeyError`` in :meth:`prepare` (check with
            :meth:`assignment` at setup).
        projector: shared by all sources (frame C, tracking broadening, CNN
            for sources with ``use_cnn``).
        side_lengths, resolution: focal spot grid (transformer: 6 m, 64 px).
    """

    def __init__(self, sources: Union[SpotSource, Sequence[SpotSource]], projector: RayProjector,
                 side_lengths: Tuple[float, float] = (6.0, 6.0),
                 resolution: Tuple[int, int] = (64, 64)) -> None:
        self.sources: List[SpotSource] = [sources] if isinstance(sources, SpotSource) else list(sources)
        if not self.sources:
            raise ValueError("At least one source is needed.")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names) or UNCOVERED in names:
            raise ValueError(f"Source names must be unique and not '{UNCOVERED}': {names}")
        self.projector = projector
        self.side_lengths = tuple(side_lengths)
        self.resolution = tuple(resolution)

    @classmethod
    def from_artifacts(cls, root: Optional[PathLike] = None, device: Union[str, torch.device] = "cpu",
                       transformer: str = "transformer/T2026.pt",
                       encoder_cache: str = "encoder_cache/T2026.pt",
                       projection_cnn: Optional[str] = "projector/cnn_frameC_v1.pt",
                       fallback: Optional[Union[SpotSource, Sequence[SpotSource]]] = None,
                       **projector_overrides) -> "HelioBeamField":
        """Transformer (+ optional fallback sources) with the standard artifacts.

        ``root`` defaults to :func:`heliobeamai.artifacts.root`
        (``$HELIOBEAMAI_ARTIFACTS``, ``<repo>/artifacts`` or
        ``~/.cache/heliobeamai/artifacts``); missing files: ``heliobeamai download``.
        ``projection_cnn=None`` gives straight rays (parallel projection).
        ``fallback``: source(s) for heliostats without transformer context,
        e.g. ``GaussianSource.linear(...)``.
        ``projector_overrides``: ``pointing_accuracy_mrad``, ``upsampling``, ...
        """
        root = Path(root) if root else artifacts.root()
        predictor = FluxPredictor.from_checkpoint(artifacts.require(transformer, root), device=device,
                                                  encoder_cache=artifacts.require(encoder_cache, root)
                                                  if encoder_cache else None)
        projector = RayProjector.from_checkpoint(artifacts.require(projection_cnn, root) if projection_cnn
                                                 else None, device=device, **projector_overrides)
        sources: List[SpotSource] = [TransformerSource(predictor)]
        if fallback is not None:
            sources += [fallback] if isinstance(fallback, SpotSource) else list(fallback)
        return cls(sources, projector)

    @property
    def device(self) -> torch.device:
        return self.projector.device

    @property
    def predictor(self) -> Optional[FluxPredictor]:
        """The transformer predictor (first :class:`TransformerSource`), if any."""
        for s in self.sources:
            if isinstance(s, TransformerSource):
                return s.predictor
        return None

    # -- coverage ----------------------------------------------------------------
    def has_context(self, heliostat_id: object) -> bool:
        """True if the transformer has calibration context for this heliostat."""
        p = self.predictor
        return p is not None and p.has_context(heliostat_id)

    def source_names(self, heliostat_ids: Sequence[object]) -> List[str]:
        """Name of the source handling each heliostat (``"uncovered"`` if none)."""
        names = [UNCOVERED] * len(heliostat_ids)
        open_idx = list(range(len(heliostat_ids)))
        for source in self.sources:
            if not open_idx:
                break
            flags = source.covers([heliostat_ids[i] for i in open_idx])
            for i, ok in zip(open_idx, flags):
                if ok:
                    names[i] = source.name
            open_idx = [i for i, ok in zip(open_idx, flags) if not ok]
        return names

    def assignment(self, heliostat_ids: Sequence[object]) -> Dict[str, List[object]]:
        """``{source name: ids, ..., "uncovered": ids}`` -- call once at setup and log it."""
        out: Dict[str, List[object]] = {s.name: [] for s in self.sources}
        out[UNCOVERED] = []
        for h, n in zip(heliostat_ids, self.source_names(heliostat_ids)):
            out[n].append(h)
        return out

    def require_coverage(self, heliostat_ids: Sequence[object]) -> None:
        """Raise ``KeyError`` if a heliostat is covered by no source."""
        missing = [h for h, n in zip(heliostat_ids, self.source_names(heliostat_ids)) if n == UNCOVERED]
        if missing:
            raise KeyError(f"{len(missing)} heliostats are covered by no source "
                           f"({[s.name for s in self.sources]}), e.g. {missing[:10]}; "
                           f"add a fallback source or remove them.")

    # -- spots and rays ----------------------------------------------------------
    def focal_spots(self, heliostat_ids: Sequence[object], sun: torch.Tensor, positions: torch.Tensor,
                    aim_points: torch.Tensor) -> torch.Tensor:
        """Raw focal spots ``[N, H, W]`` (frame C, around each aim point, no tracking error)."""
        ids = list(heliostat_ids)
        cond, dirs, lengths = self._conditions(sun, positions, aim_points)
        out = torch.zeros(len(ids), *self.resolution, device=self.device)
        for source, idx in self._groups(ids):
            out[idx] = self._source_spots(source, ids, idx, cond, dirs, lengths).to(out)
        return out

    def prepare(self, heliostat_ids: Sequence[object], sun: torch.Tensor, positions: torch.Tensor,
                aim_points: torch.Tensor) -> PreparedSpots:
        """Spots and CNN deviations for the current geometry (no gradient), input order kept."""
        ids = list(heliostat_ids)
        with torch.no_grad():
            cond, dirs, lengths = self._conditions(sun, positions, aim_points)
            parts = []
            for source, idx in self._groups(ids):
                raw = self._source_spots(source, ids, idx, cond, dirs, lengths)
                s, dev = self.projector.prepare(raw, cond[idx, 0], self.side_lengths, use_cnn=source.use_cnn)
                parts.append((idx, s, dev))
            h, w = parts[0][1].shape[-2:] if parts else self.resolution
            spots = torch.zeros(len(ids), h, w, device=self.device)
            deviations = None
            if any(dev is not None for _, _, dev in parts):
                deviations = torch.zeros(len(ids), 3, h, w, device=self.device)
            for idx, s, dev in parts:
                spots[idx] = s.to(spots)
                if dev is not None:
                    deviations[idx] = dev.to(deviations)
        return PreparedSpots([normalize_heliostat_id(h) for h in ids], spots, deviations,
                             sun.detach().to(self.device), positions.detach().to(self.device),
                             self.source_names(ids))

    def rays(self, prepared: PreparedSpots, aim_points: torch.Tensor, powers: Optional[torch.Tensor] = None,
             group_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rays ``[N*H*W, 7]`` (``8`` with ``group_indices``): origin, direction, energy[, group].

        Heliostat ``i`` owns rows ``i*H*W ... (i+1)*H*W - 1`` (input order).
        Origins and directions follow ``aim_points`` (gradients flow into
        them); spot shapes and deviations stay as prepared.
        """
        cond, _, _ = self._conditions(prepared.sun, prepared.positions, aim_points)
        return self.projector.rays_from_prepared(prepared.spots, prepared.deviations, aim_points.to(self.device),
                                                 cond[:, 0], powers, self.side_lengths, group_indices)

    def __call__(self, heliostat_ids: Sequence[object], sun: torch.Tensor, positions: torch.Tensor,
                 aim_points: torch.Tensor, powers: Optional[torch.Tensor] = None,
                 group_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One-shot: ``rays(prepare(...), aim_points, ...)``."""
        return self.rays(self.prepare(heliostat_ids, sun, positions, aim_points), aim_points, powers,
                         group_indices)

    # -- internals ---------------------------------------------------------------
    def _groups(self, ids: List[object]) -> List[Tuple[SpotSource, torch.Tensor]]:
        self.require_coverage(ids)
        names = self.source_names(ids)
        groups = []
        for source in self.sources:
            idx = [i for i, n in enumerate(names) if n == source.name]
            if idx:
                groups.append((source, torch.tensor(idx, dtype=torch.long, device=self.device)))
        return groups

    def _source_spots(self, source: SpotSource, ids: List[object], idx: torch.Tensor, cond: torch.Tensor,
                      dirs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        geometry = SpotGeometry([ids[i] for i in idx.tolist()], cond[idx, 0], dirs[idx], lengths[idx].reshape(-1),
                                self.resolution, self.side_lengths)
        spots = source.spots(geometry).to(self.device, torch.float32)
        if spots.shape != (len(idx), *self.resolution):
            raise ValueError(f"Source '{source.name}' returned {tuple(spots.shape)}, "
                             f"expected {(len(idx), *self.resolution)}")
        if not torch.isfinite(spots).all():
            raise RuntimeError(f"Source '{source.name}' returned non-finite spots.")
        return spots

    def _conditions(self, sun, positions, aim_points):
        return build_conditions(sun.to(self.device, torch.float32), positions.to(self.device, torch.float32),
                                aim_points.to(self.device, torch.float32), self._distance_normalization())

    def _distance_normalization(self) -> float:
        p = self.predictor
        return float(p.model.cfg.distance_normalization_m) if p is not None \
            else float(self.projector.config.distance_normalization_m)


__all__ = ["HelioBeamField", "PreparedSpots", "GaussianSource", "TransformerSource", "SpotSource",
           "SpotGeometry", "UNCOVERED"]
