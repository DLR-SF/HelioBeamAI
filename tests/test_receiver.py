"""Receiver models, heliostat powers, `heliobeamai predict`."""
import math
from pathlib import Path

import pytest
import torch

from heliobeamai import CylindricalReceiver, OVRReceiver, PlanarReceiver, RayProjector, build_conditions
from heliobeamai.inference.power import cosine_factor, heliostat_powers
from heliobeamai.predict import sun_vector

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "data"


def _rays(origins, directions, energy=1.0):
    o = torch.as_tensor(origins, dtype=torch.float32).reshape(-1, 3)
    d = torch.nn.functional.normalize(torch.as_tensor(directions, dtype=torch.float32).reshape(-1, 3), dim=-1)
    return torch.cat([o, d, torch.full((len(o), 1), float(energy))], dim=-1)


def test_planar_receiver_frame_c_and_energy():
    rec = PlanarReceiver(center=(0, 0, 55), width=4, height=2, resolution=(2, 4))
    # rays from the north, travelling south (-y), hitting top-east and bottom-west
    rays = _rays([[1.5, 10, 55.5], [-1.5, 10, 54.5]], [[0, -1, 0], [0, -1, 0]], energy=2.0)
    points, sv, hit = rec.intersect(rays[:, 3:6], rays[:, :3])
    assert hit.all()
    torch.testing.assert_close(points[:, 1], torch.zeros(2))
    flux = rec.project_flux(rays)
    assert flux[0, 0] > 0 and flux[1, 3] > 0                       # east = left column, up = top row
    torch.testing.assert_close(flux.sum() * rec.bin_area(), torch.tensor(4.0))
    # rays from behind and rays pointing away do not count
    back = _rays([[0, -10, 55], [0, 10, 55]], [[0, 1, 0], [0, 1, 0]])
    assert not rec.intersect(back[:, 3:6], back[:, :3])[2].any()


def test_bin_centers_match_ray_projector_layout():
    """A RayProjector spot binned on a receiver of the same grid reproduces the spot."""
    rec = PlanarReceiver(center=(0, 0, 55), width=6, height=6, resolution=(64, 64))
    torch.manual_seed(0)
    spot = torch.rand(1, 64, 64)
    cond, _, _ = build_conditions(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([[0.0, 80.0, 1.6]]),
                                  torch.tensor([[0.0, 0.0, 55.0]]))
    rays = RayProjector()(spot, torch.tensor([[0.0, 0.0, 55.0]]), cond[:, 0], powers=torch.tensor([36.0]))
    flux = rec.project_flux(rays)
    torch.testing.assert_close(flux / flux.sum(), spot[0] / spot[0].sum(), atol=1e-6, rtol=1e-4)
    c = rec.bin_centers()
    assert c[0, 0, 0] > 2.9 and c[0, 0, 2] > 57.9                  # bin (0, 0): east, top


def test_soft_binning_is_differentiable_and_conserves_energy():
    rec = PlanarReceiver(center=(0, 0, 55), width=6, height=6, resolution=(30, 30))
    origins = torch.tensor([[0.13, 5.0, 55.07], [-1.02, 5.0, 54.41]], requires_grad=True)
    d = torch.tensor([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    rays = torch.cat([origins, d, torch.ones(2, 1)], dim=-1)
    flux = rec.project_flux(rays, soft=True)
    torch.testing.assert_close(flux.sum() * rec.bin_area(), torch.tensor(2.0))
    (flux * torch.arange(900.0).reshape(30, 30)).sum().backward()
    assert origins.grad is not None and origins.grad.abs().sum() > 0


def test_ovr_receiver_orientation():
    rec = OVRReceiver(center=(0, 0, 55), width=2, height=2, resolution=(2, 2), tilt_deg=30)
    torch.testing.assert_close(rec.normal, torch.tensor([0.0, math.cos(math.radians(30)), -0.5]))
    torch.testing.assert_close(rec.right, torch.tensor([-1.0, 0.0, 0.0]))
    facing_east = OVRReceiver(center=(0, 0, 55), width=2, height=2, azimuth_deg=90)
    torch.testing.assert_close(facing_east.normal, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=0)


def test_cylindrical_receiver_hits_outer_surface():
    rec = CylindricalReceiver(base=(0, 0, 50), axis=(0, 0, 1), radius=4, height=10, angles_deg=(45, 135),
                              resolution=(10, 18))
    rays = _rays([[0, 20, 55], [2, 20, 52]], [[0, -1, 0], [0, -1, 0]])        # angles 90 and 60 deg
    points, sv, hit = rec.intersect(rays[:, 3:6], rays[:, :3])
    assert hit.all()
    torch.testing.assert_close(points[0], torch.tensor([0.0, 4.0, 55.0]))
    torch.testing.assert_close(points[:, :2].norm(dim=-1), torch.tensor([4.0, 4.0]))
    # columns left to right as seen from the north: the east ray (x = 2) lies left of the centre ray
    assert sv[1, 0] < sv[0, 0]
    flux = rec.project_flux(rays)
    torch.testing.assert_close(flux.sum() * rec.bin_area(), torch.tensor(2.0))
    miss = _rays([[0, 20, 45]], [[0, -1, 0]])                       # below the section
    assert not rec.intersect(miss[:, 3:6], miss[:, :3])[2].any()


def test_heliostat_powers_cosine():
    sun = torch.tensor([0.0, 1.0, 0.0])                           # sun in the north, horizontal
    pos = torch.tensor([[0.0, 100.0, 0.0], [0.0, 100.0, 0.0]])
    aim = torch.tensor([[0.0, 200.0, 0.0], [0.0, 0.0, 0.0]])       # back to the sun / away from it
    torch.testing.assert_close(cosine_factor(sun, pos, aim), torch.tensor([1.0, 0.0]), atol=1e-4, rtol=0)
    p = heliostat_powers(sun_vector(180, 90), pos[:1], torch.tensor([[0.0, 100.0, 100.0]]), 800.0, 8.0, 0.9)
    torch.testing.assert_close(p, torch.tensor([800 * 8 * 0.9]))
    torch.testing.assert_close(sun_vector(90, 0), torch.tensor([1.0, 0.0, 0.0]), atol=1e-7, rtol=0)


def test_predict_cli(tmp_path, artifact):
    import numpy as np
    from heliobeamai.cli import main
    artifact("transformer/T2026.pt")
    csv = tmp_path / "field.csv"
    csv.write_text((EXAMPLES / "heliostat_positions.csv").read_text()
                   .replace("x_east_m,y_north_m,z_up_m", "x,y,z") + "XX01,0.0,120.0,1.6\n")
    out = tmp_path / "flux"
    assert main(["predict", "--heliostats", str(csv), "--aim", "0", "0", "55", "--sun", "180", "45",
                 "--dni", "850", "--mirror-area", "8.2", "--reflectivity", "0.9", "--fallback", "3.1", "0.58",
                 "--resolution", "40", "40", "--device", "cpu", "--output", str(out)]) == 0
    r = np.load(out.with_suffix(".npz"))
    assert r["flux_w_m2"].shape == (40, 40) and list(r["sources"]).count("gaussian") == 1
    assert 0.5 < float(r["intercepted"]) <= 1.0 + 1e-5
    assert r["powers_w"].max() <= 850 * 8.2 * 0.9 + 1e-3
