"""Reflected power per heliostat from DNI, mirror area, reflectivity and the cosine effect."""
from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F

Number = Union[float, torch.Tensor]


def cosine_factor(sun: torch.Tensor, positions: torch.Tensor, aim_points: torch.Tensor) -> torch.Tensor:
    """``cos`` of the incidence angle on the mirror ``[N]``.

    The mirror normal bisects the sun vector and the aim direction, so
    ``cos(theta) = sqrt((1 + sun . a) / 2)`` with ``a`` the unit vector from
    the heliostat to its aim point.
    """
    a = F.normalize(aim_points - positions, dim=-1)
    s = F.normalize(sun.to(a).expand_as(a), dim=-1)
    return torch.sqrt(((1.0 + (s * a).sum(-1)) / 2.0).clamp_min(0.0))


def heliostat_powers(sun: torch.Tensor, positions: torch.Tensor, aim_points: torch.Tensor, dni: Number,
                     mirror_area: Number, reflectivity: Number = 1.0) -> torch.Tensor:
    """Power reflected towards the aim point ``[N]`` in W.

    ``P = DNI * A * rho * cos(theta)``. Shading, blocking, atmospheric
    attenuation and spillage are not included (spillage follows from the
    receiver intersection).
    """
    cos = cosine_factor(sun, positions, aim_points)
    return torch.as_tensor(dni, dtype=cos.dtype) * torch.as_tensor(mirror_area, dtype=cos.dtype) \
        * torch.as_tensor(reflectivity, dtype=cos.dtype) * cos
