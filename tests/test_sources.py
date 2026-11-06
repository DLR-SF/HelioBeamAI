"""Spot sources, fallback assignment and the shared tracking broadening."""
import math

import pytest
import torch

from heliobeamai import (FocalSpotToDirectionsNet, GaussianSource, HelioBeamField, ProjectionConfig,
                         RayProjector, SpotGeometry, build_conditions)

SUN = torch.tensor([0.3, -0.5, 0.8])


class SetSource:
    """Test source: covers a fixed id set, returns a constant-shaped spot."""

    def __init__(self, ids, name, value=1.0, use_cnn=True):
        self.ids, self.name, self.value, self.use_cnn = set(ids), name, value, use_cnn
        self.calls = []

    def covers(self, ids):
        return [h in self.ids for h in ids]

    def spots(self, g: SpotGeometry):
        self.calls.append(list(g.heliostat_ids))
        s = torch.zeros(len(g.heliostat_ids), *g.resolution)
        s[:, 30:34, 30:34] = self.value
        return s


def _geometry(n=3):
    pos = torch.tensor([[-57.3, 24.8, 1.6], [120.0, 180.0, 1.7], [-200.0, 260.0, 1.8]])[:n]
    aim = torch.tensor([[0.0, 0.0, 55.0], [1.0, 0.0, 54.0], [-1.0, 0.0, 56.0]])[:n]
    return pos, aim


def _projector(pointing=0.0, cnn=False, broadening="oblique"):
    torch.manual_seed(0)
    net = FocalSpotToDirectionsNet(0.0, cond_dim=7) if cnn else None
    return RayProjector(ProjectionConfig(pointing_accuracy_mrad=pointing, broadening=broadening), net)


def _moments(img, side=6.0):
    h, w = img.shape
    x = side / 2 - (torch.arange(w) + 0.5) * side / w
    z = side / 2 - (torch.arange(h) + 0.5) * side / h
    zg, xg = torch.meshgrid(z, x, indexing="ij")
    p = img / img.sum()
    mx, mz = (p * xg).sum(), (p * zg).sum()
    return mx, mz, (p * (xg - mx) ** 2).sum(), (p * (zg - mz) ** 2).sum(), (p * (xg - mx) * (zg - mz)).sum()


# -- GaussianSource ----------------------------------------------------------------
def test_gaussian_spot_is_stretched_by_incidence():
    pos = torch.tensor([[0.0, 100.0, 0.0]])                       # due north, beam rises to the aim point
    aim = torch.tensor([[0.0, 0.0, 55.0]])
    cond, dirs, lengths = build_conditions(SUN, pos, aim)
    g = SpotGeometry(["X"], cond[:, 0], dirs, lengths.reshape(-1))
    spot = GaussianSource(lambda ids, d: torch.full_like(d, 0.4)).spots(g)[0]
    mx, mz, vx, vz, _ = _moments(spot)
    assert abs(mx) < 1e-4 and abs(mz) < 1e-4
    assert math.isclose(math.sqrt(vx), 0.4, rel_tol=0.02)
    assert math.isclose(math.sqrt(vz), 0.4 / abs(float(dirs[0, 1])), rel_tol=0.02)
    assert math.isclose(float(spot.mean()), 0.1, rel_tol=1e-5)


def test_gaussian_plus_tracking_equals_combined_width():
    """sigma (source) (*) d*theta (projector) == sqrt(sigma^2 + (d theta)^2), also oblique."""
    pos, aim = _geometry()
    theta = 1.8
    _, _, lengths = build_conditions(SUN, pos, aim)
    sigma = 0.35
    ids = ["A", "B", "C"]
    broad = HelioBeamField(GaussianSource(lambda i, d: torch.full_like(d, sigma)), _projector(theta))
    combined = HelioBeamField(GaussianSource(lambda i, d: torch.sqrt(sigma ** 2 + (d * theta / 1000) ** 2)),
                              _projector(0.0))
    a = broad.prepare(ids, SUN, pos, aim).spots
    b = combined.prepare(ids, SUN, pos, aim).spots
    for i in range(3):
        rel = float((a[i] / a[i].sum() - b[i] / b[i].sum()).abs().sum())
        assert rel < 0.01, (i, rel)
    # the isotropic (old) kernel does not reproduce the oblique beam for the far heliostats
    iso = HelioBeamField(GaussianSource(lambda i, d: torch.full_like(d, sigma)),
                         _projector(theta, broadening="isotropic")).prepare(ids, SUN, pos, aim).spots
    assert float((iso[2] / iso[2].sum() - b[2] / b[2].sum()).abs().sum()) > 0.03


def test_gaussian_source_coverage_and_validation():
    src = GaussianSource(lambda i, d: torch.ones_like(d), heliostat_ids=["AA23", 10124])
    assert src.covers(["AA23", "AA24", "BB01", 10123]) == [True, True, False, True]
    bad = GaussianSource(lambda i, d: -torch.ones_like(d))
    with pytest.raises(ValueError):
        bad.sigma(["A"], torch.tensor([100.0]))
    law = GaussianSource.field_law(1.5, 0.2, {"AA23": 2.0})
    s = law.sigma(["AA23", "AA24"], torch.tensor([100.0, 100.0]))
    torch.testing.assert_close(s, torch.tensor([2.0, 1.0]) * math.sqrt(0.2 ** 2 + 0.15 ** 2))
    lin = GaussianSource.linear(lambda ids: (torch.tensor([0.1, 0.2]), torch.tensor([0.5, 1.0])), 100.0)
    torch.testing.assert_close(lin.sigma(["A", "B"], torch.tensor([200.0, 100.0])), torch.tensor([1.1, 1.2]))


# -- assignment and mixing ---------------------------------------------------------
def test_assignment_and_uncovered_error():
    field = HelioBeamField([SetSource({"A", "B"}, "transformer")], _projector())
    assert field.assignment(["A", "C", "B"]) == {"transformer": ["A", "B"], "uncovered": ["C"]}
    pos, aim = _geometry()
    with pytest.raises(KeyError, match="covered by no source"):
        field.prepare(["A", "C", "B"], SUN, pos, aim)
    with pytest.raises(ValueError):
        HelioBeamField([SetSource({"A"}, "x"), SetSource({"B"}, "x")], _projector())


def test_first_covering_source_wins_and_order_is_kept():
    first = SetSource({"A", "C"}, "transformer", value=1.0)
    fallback = SetSource({"A", "B", "C"}, "gaussian", value=2.0, use_cnn=False)
    field = HelioBeamField([first, fallback], _projector(cnn=True))
    pos, aim = _geometry()
    ids = ["B", "A", "C"]
    assert field.assignment(ids) == {"transformer": ["A", "C"], "gaussian": ["B"], "uncovered": []}
    prepared = field.prepare(ids, SUN, pos, aim)
    assert first.calls == [["A", "C"]] and fallback.calls == [["B"]]
    assert prepared.sources == ["gaussian", "transformer", "transformer"]
    assert float(prepared.spots[0].max()) == 2.0 and float(prepared.spots[1].max()) == 1.0
    # CNN only for the transformer source: fallback heliostat gets straight rays
    assert prepared.deviations is not None
    assert float(prepared.deviations[0].abs().max()) == 0.0 and float(prepared.deviations[1].abs().max()) > 0
    powers = torch.tensor([10.0, 20.0, 30.0])
    rays = field.rays(prepared, aim, powers)
    torch.testing.assert_close(rays[:, 6].view(3, -1).sum(1), powers)
    _, dirs, _ = build_conditions(SUN, pos, aim)
    torch.testing.assert_close(rays.view(3, -1, 7)[0, :, 3:6], dirs[0].expand(64 * 64, 3))


# -- against aimpoint-control (read-only import) -----------------------------------
