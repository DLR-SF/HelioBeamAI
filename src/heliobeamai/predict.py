"""Receiver flux of a heliostat field: the steps behind ``heliobeamai predict``.

    field = HelioBeamField.from_artifacts(pointing_accuracy_mrad=1.8)
    receiver = OVRReceiver(center=(0, 0, 55), width=8, height=8, resolution=(100, 100))
    result = receiver_flux(field, receiver, ids, positions, aim_points, sun, dni=850,
                           mirror_area=8.2, reflectivity=0.9)
    result["flux"]          # [rows, cols] W/m^2, frame C (as seen from the field)
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch

from .field import HelioBeamField
from .inference.power import heliostat_powers
from .receiver import Receiver

Number = Union[float, torch.Tensor]


def sun_vector(azimuth_deg: float, elevation_deg: float) -> torch.Tensor:
    """Unit vector towards the sun (x east, y north, z up); azimuth from north, clockwise."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return torch.tensor([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])


def read_heliostats(path: Union[str, Path]) -> Dict[str, object]:
    """Heliostat CSV: ``heliostat_id, x, y, z`` (tower-base frame, m) and optional
    ``aim_x, aim_y, aim_z``, ``mirror_area`` (m^2), ``reflectivity``."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} contains no heliostats")
    missing = {"heliostat_id", "x", "y", "z"} - set(rows[0])
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")

    def column(*names) -> Optional[torch.Tensor]:
        if not all(n in rows[0] and rows[0][n] != "" for n in names):
            return None
        return torch.tensor([[float(r[n]) for n in names] for r in rows]).squeeze(-1)

    return {"heliostat_ids": [r["heliostat_id"] for r in rows], "positions": column("x", "y", "z"),
            "aim_points": column("aim_x", "aim_y", "aim_z"), "mirror_area": column("mirror_area"),
            "reflectivity": column("reflectivity")}


def receiver_flux(field: HelioBeamField, receiver: Receiver, heliostat_ids: Sequence[object],
                  positions: torch.Tensor, aim_points: torch.Tensor, sun: torch.Tensor, dni: Number,
                  mirror_area: Number, reflectivity: Number = 1.0, soft: bool = True) -> Dict[str, object]:
    """Total flux density on the receiver and the per-heliostat powers.

    Powers: ``DNI * mirror_area * reflectivity * cos(incidence)`` (no shading,
    blocking or atmospheric attenuation). Returns ``flux [rows, cols]`` (W/m^2),
    ``powers [N]`` (W), ``sources`` (spot model per heliostat), ``intercepted``
    (fraction of the reflected power that hits the receiver).
    """
    ids: List[object] = list(heliostat_ids)
    positions = positions.to(torch.float32)
    aim_points = aim_points.to(torch.float32)
    powers = heliostat_powers(sun, positions, aim_points, dni, mirror_area, reflectivity)
    prepared = field.prepare(ids, sun, positions, aim_points)
    with torch.no_grad():
        rays = field.rays(prepared, aim_points, powers).cpu()
    flux = receiver.project_flux(rays, soft=soft)
    total = float(powers.sum())
    return {"flux": flux, "powers": powers, "sources": prepared.sources,
            "intercepted": float(flux.sum()) * receiver.bin_area() / total if total > 0 else 0.0}
