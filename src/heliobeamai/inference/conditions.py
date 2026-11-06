"""Helpers to build model inputs from plant geometry."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def normalize_heliostat_id(value: object) -> str:
    """Canonical heliostat name, e.g. ``10123`` -> ``"AA23"``, ``21005`` -> ``"BJ05"``.

    Numeric ids encode row (first digit) and column (next two digits) as
    letters; the remainder is the heliostat number in that row/column block.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, bytes):
        value = value.decode()
    text = str(value).strip().upper()
    try:
        numeric = int(float(text))
    except ValueError:
        return text
    digits = str(numeric)
    if len(digits) >= 4:
        try:
            return (chr(ord("A") + int(digits[0]) - 1)
                    + chr(ord("A") + int(digits[1:3]) - 1) + digits[3:])
        except (ValueError, IndexError):
            pass
    return digits


def normalize_images_to_mean(images: torch.Tensor, target_mean: Optional[float]) -> torch.Tensor:
    """Scale every image so that its mean equals ``target_mean`` (no-op for None)."""
    if target_mean is None:
        return images
    mean = images.mean(dim=(-1, -2), keepdim=True)
    mean = torch.where(mean.abs() < 1e-12, torch.ones_like(mean), mean)
    return images * (float(target_mean) / mean)


def build_conditions(sun_vector: torch.Tensor, heliostat_positions: torch.Tensor,
                     aim_points: torch.Tensor, distance_normalization_m: float = 350.0
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Model conditions for heliostats aiming at ``aim_points``.

    Args:
        sun_vector: ``[3]`` (shared) or ``[N, 3]`` vector towards the sun.
        heliostat_positions, aim_points: ``[N, 3]`` in the same frame.

    Returns:
        ``conditions`` ``[N, 1, 7]``, unit ``aim_directions`` ``[N, 3]`` and
        ``aim_lengths`` ``[N, 1]`` in metres.
    """
    aim_vectors = aim_points - heliostat_positions
    aim_lengths = torch.linalg.vector_norm(aim_vectors, dim=1, keepdim=True).clamp_min(1e-6)
    aim_directions = F.normalize(aim_vectors, p=2, dim=1, eps=1e-6)
    sun = sun_vector.to(aim_points)
    if sun.ndim == 1:
        sun = sun.view(1, 3).expand(aim_points.shape[0], 3)
    sun = F.normalize(sun, p=2, dim=-1, eps=1e-6)
    conditions = torch.cat([sun, aim_directions, aim_lengths / float(distance_normalization_m)],
                           dim=-1)
    return conditions.unsqueeze(1), aim_directions, aim_lengths
