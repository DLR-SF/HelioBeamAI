"""Receiver models: intersect rays with a receiver surface and bin their energy.

    from heliobeamai.receiver import OVRReceiver
    receiver = OVRReceiver(center=(0, 0, 55), width=6, height=6, resolution=(60, 60))
    flux = receiver.project_flux(rays)              # [rows, cols] W/m^2 (rays: HelioBeamField.rays)
    receiver.visualize_flux_distribution(flux)

Flux maps use the image convention of the package (frame C, see
``docs/data.md``): row 0 is the top, columns run from left to right *as seen
from the field* (for a surface facing north: column 0 = east). ``plt.imshow``
therefore shows the receiver as seen by the heliostats.

Rays are ``[M, 7+]`` tensors ``(origin 3, direction 3, energy, ...)`` in the
tower-base frame (x east, y north, z up, metres), as returned by
:meth:`heliobeamai.HelioBeamField.rays`.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

Vector = Union[Sequence[float], torch.Tensor]


def _vec(v: Vector) -> torch.Tensor:
    return torch.as_tensor(v, dtype=torch.float32).reshape(3)


class Receiver(ABC):
    """A receiver surface with a ``rows x cols`` grid of bins.

    Subclasses define the surface by :meth:`intersect`, which returns for every
    ray the surface coordinates ``(s, v)`` of its hit point in metres: ``s``
    runs along the columns (left to right as seen from the field), ``v`` along
    the rows (top to bottom). The grid covers ``[0, width) x [0, height)``.
    """

    def __init__(self, width: float, height: float, resolution: Tuple[int, int]) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width, self.height = float(width), float(height)
        self.rows, self.cols = int(resolution[0]), int(resolution[1])

    # -- geometry -------------------------------------------------------------------
    @abstractmethod
    def intersect(self, ray_directions: torch.Tensor, ray_origins: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hit points of rays ``[M, 3]`` (directions, origins).

        Returns ``points [M, 3]`` (world), ``sv [M, 2]`` surface coordinates
        (column direction, row direction; metres) and ``hit [M]`` (bool: the ray
        hits the receiver surface in front of its origin).
        """

    def bin_area(self) -> float:
        """Area of one bin (m^2)."""
        return self.width / self.cols * self.height / self.rows

    def extent(self) -> Tuple[float, float, float, float]:
        """``imshow`` extent in surface coordinates: (left, right, bottom, top)."""
        return 0.0, self.width, self.height, 0.0

    # -- flux -------------------------------------------------------------------------
    def project_flux(self, rays: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """Flux density ``[rows, cols]`` in W/m^2 (energies in W).

        ``soft=True`` distributes every ray bilinearly over the four nearest
        bin centres, which makes the map differentiable with respect to the ray
        origins and directions (e.g. aim points); ``soft=False`` counts each ray
        in the bin it hits.
        """
        directions, origins, energy = rays[:, 3:6], rays[:, :3], rays[:, 6]
        _, sv, hit = self.intersect(directions, origins)
        col = sv[:, 0] / self.width * self.cols            # continuous bin coordinates
        row = sv[:, 1] / self.height * self.rows
        e = energy * hit.to(energy.dtype)
        flat = torch.zeros(self.rows * self.cols, dtype=energy.dtype, device=energy.device)
        if soft:
            c, r = col - 0.5, row - 0.5                     # relative to bin centres
            c0, r0 = torch.floor(c), torch.floor(r)
            fc, fr = c - c0, r - r0
            for dr, wr in ((0, 1 - fr), (1, fr)):
                for dc, wc in ((0, 1 - fc), (1, fc)):
                    rr, cc = (r0 + dr).long(), (c0 + dc).long()
                    ok = (rr >= 0) & (rr < self.rows) & (cc >= 0) & (cc < self.cols) & hit
                    idx = (rr.clamp(0, self.rows - 1) * self.cols + cc.clamp(0, self.cols - 1))
                    flat = flat.index_add(0, idx, e * wr * wc * ok.to(e.dtype))
        else:
            rr, cc = torch.floor(row).long(), torch.floor(col).long()
            ok = (rr >= 0) & (rr < self.rows) & (cc >= 0) & (cc < self.cols) & hit
            idx = rr.clamp(0, self.rows - 1) * self.cols + cc.clamp(0, self.cols - 1)
            flat = flat.index_add(0, idx, e * ok.to(e.dtype))
        return flat.reshape(self.rows, self.cols) / self.bin_area()

    def visualize_flux_distribution(self, flux: torch.Tensor, ax=None, unit: str = "kW/m²",
                                    scale: float = 1e-3, **imshow_kwargs):
        """Plot a flux map as seen from the field; returns the matplotlib image."""
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        im = ax.imshow((flux.detach().cpu() * scale).numpy(), extent=self.extent(), **imshow_kwargs)
        ax.set_xlabel("m (left to right as seen from the field)")
        ax.set_ylabel("m (from the top)")
        plt.colorbar(im, ax=ax, label=unit)
        return im


class PlanarReceiver(Receiver):
    """Flat rectangular receiver (e.g. the aperture of an open volumetric receiver).

    Args:
        center: centre of the rectangle (tower-base frame, m).
        normal: unit normal pointing towards the field (default north, +y).
        up: direction of the upper edge, projected into the plane (default +z).
        width, height: size in m; resolution: ``(rows, cols)``.
    """

    def __init__(self, center: Vector, width: float, height: float, resolution: Tuple[int, int] = (64, 64),
                 normal: Vector = (0.0, 1.0, 0.0), up: Vector = (0.0, 0.0, 1.0)) -> None:
        super().__init__(width, height, resolution)
        self.center = _vec(center)
        self.normal = F.normalize(_vec(normal), dim=0)
        up = _vec(up)
        up = up - (up @ self.normal) * self.normal
        if float(up.norm()) < 1e-6:
            raise ValueError("up must not be parallel to the normal")
        self.up = F.normalize(up, dim=0)
        # right as seen by a viewer in front of the surface (looking along -normal)
        self.right = torch.linalg.cross(self.up, self.normal)

    def intersect(self, ray_directions, ray_origins):
        n, c = self.normal.to(ray_origins), self.center.to(ray_origins)
        denom = ray_directions @ n
        safe = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
        t = ((c - ray_origins) @ n) / safe
        points = ray_origins + t[:, None] * ray_directions
        rel = points - c
        s = rel @ self.right.to(rel) + self.width / 2
        v = self.height / 2 - rel @ self.up.to(rel)
        hit = (denom < 0) & (t > -1e-4)                               # front side, not behind the origin
        hit = hit & (s >= 0) & (s < self.width) & (v >= 0) & (v < self.height)
        return points, torch.stack([s, v], dim=-1), hit

    def bin_centers(self) -> torch.Tensor:
        """World coordinates of the bin centres ``[rows, cols, 3]``."""
        s = (torch.arange(self.cols) + 0.5) * self.width / self.cols - self.width / 2
        v = self.height / 2 - (torch.arange(self.rows) + 0.5) * self.height / self.rows
        vg, sg = torch.meshgrid(v, s, indexing="ij")
        return self.center + sg[..., None] * self.right + vg[..., None] * self.up


class OVRReceiver(PlanarReceiver):
    """Aperture of an open volumetric receiver: a rectangle facing the field.

    The aperture faces ``azimuth_deg`` (from north, clockwise; 0 = facing a
    northern field) and is tilted down towards the field by ``tilt_deg``
    (0 = vertical).
    """

    def __init__(self, center: Vector, width: float, height: float, resolution: Tuple[int, int] = (64, 64),
                 tilt_deg: float = 0.0, azimuth_deg: float = 0.0) -> None:
        az, tilt = math.radians(azimuth_deg), math.radians(tilt_deg)
        horizontal = torch.tensor([math.sin(az), math.cos(az), 0.0])
        normal = math.cos(tilt) * horizontal + torch.tensor([0.0, 0.0, -math.sin(tilt)])
        up = math.sin(tilt) * horizontal + torch.tensor([0.0, 0.0, math.cos(tilt)])
        super().__init__(center, width, height, resolution, normal=normal, up=up)
        self.tilt_deg, self.azimuth_deg = float(tilt_deg), float(azimuth_deg)


class CylindricalReceiver(Receiver):
    """Section of a cylinder (e.g. an external receiver), hit from outside.

    Args:
        base: point on the cylinder axis at the lower edge (m).
        axis: axis direction (towards the upper edge).
        radius, height: cylinder radius and section height along the axis (m).
        angles_deg: angular range ``(start, stop)`` of the section, measured
            around the axis from ``reference`` (a direction perpendicular to
            the axis; default: horizontal east) towards ``axis x reference``.
        resolution: ``(rows, cols)``; columns are ordered left to right as
            seen from outside at the centre of the section.
    """

    def __init__(self, base: Vector, axis: Vector, radius: float, height: float,
                 angles_deg: Tuple[float, float] = (0.0, 360.0), resolution: Tuple[int, int] = (64, 64),
                 reference: Optional[Vector] = None) -> None:
        a0, a1 = (math.radians(x) for x in angles_deg)
        if not a1 > a0:
            raise ValueError("angles_deg must be increasing")
        super().__init__(radius * (a1 - a0), height, resolution)
        self.base, self.axis = _vec(base), F.normalize(_vec(axis), dim=0)
        ref = _vec(reference) if reference is not None else torch.tensor([1.0, 0.0, 0.0])
        ref = ref - (ref @ self.axis) * self.axis
        if float(ref.norm()) < 1e-6:
            raise ValueError("reference must not be parallel to the axis")
        self.e1 = F.normalize(ref, dim=0)
        self.e2 = torch.linalg.cross(self.axis, self.e1)
        self.radius, self.a0, self.a1 = float(radius), a0, a1
        # column order: left to right as seen from outside at the middle angle
        mid = (a0 + a1) / 2
        outward = math.cos(mid) * self.e1 + math.sin(mid) * self.e2
        tangent = -math.sin(mid) * self.e1 + math.cos(mid) * self.e2         # d point / d angle
        right = torch.linalg.cross(self.axis, outward)
        self.flip = bool(tangent @ right < 0)

    def intersect(self, ray_directions, ray_origins):
        ax, e1, e2 = (x.to(ray_origins) for x in (self.axis, self.e1, self.e2))
        rel = ray_origins - self.base.to(ray_origins)
        d_perp = ray_directions - (ray_directions @ ax)[:, None] * ax
        o_perp = rel - (rel @ ax)[:, None] * ax
        a = (d_perp * d_perp).sum(-1)
        b = 2 * (d_perp * o_perp).sum(-1)
        c = (o_perp * o_perp).sum(-1) - self.radius ** 2
        disc = b * b - 4 * a * c
        root = torch.sqrt(disc.clamp_min(0.0))
        a_safe = a.clamp_min(1e-12)
        candidates = []
        for t in ((-b - root) / (2 * a_safe), (-b + root) / (2 * a_safe)):     # outer side first
            p = rel + t[:, None] * ray_directions
            h = p @ ax
            ang = torch.atan2(p @ e2, p @ e1)
            ang = torch.where(ang < self.a0, ang + 2 * math.pi, ang)
            ok = (disc >= 0) & (a > 1e-12) & (t > -1e-4) & (h >= 0) & (h < self.height) & (ang < self.a1)
            candidates.append((t, h, ang, ok))
        (t0, h0, g0, ok0), (t1, h1, g1, ok1) = candidates
        t = torch.where(ok0, t0, t1)
        h = torch.where(ok0, h0, h1)
        ang = torch.where(ok0, g0, g1)
        hit = ok0 | ok1
        s = (ang - self.a0) * self.radius
        if self.flip:
            s = self.width - s
        points = ray_origins + t[:, None] * ray_directions
        return points, torch.stack([s, self.height - h], dim=-1), hit


__all__ = ["Receiver", "PlanarReceiver", "OVRReceiver", "CylindricalReceiver"]
