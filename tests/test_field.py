"""HelioBeamField: the one-object interface used by aimpoint-control."""
import torch

from heliobeamai import FluxPredictor, HelioBeamField, RayProjector, build_conditions


def test_field_equals_manual_chain_and_is_differentiable(artifact):
    root = artifact("transformer/T2026.pt").parents[1]
    field = HelioBeamField.from_artifacts(root, pointing_accuracy_mrad=1.8)
    ids = ["AA23", "AC29", "AA44"]
    sun = torch.tensor([0.3, -0.5, 0.8])
    pos = torch.tensor([[-57.26, 24.80, 1.59], [-30.8, 34.0, 1.7], [35.0, 25.0, 1.6]])
    aim = torch.tensor([[0.0, 0.0, 55.0], [0.5, 0.0, 54.0], [-0.5, 0.0, 56.0]])
    powers = torch.tensor([30.0, 25.0, 20.0])

    rays = field(ids, sun, pos, aim, powers)
    pred = FluxPredictor.from_checkpoint(root / "transformer/T2026.pt", encoder_cache=root / "encoder_cache/T2026.pt")
    proj = RayProjector.from_checkpoint(root / "projector/cnn_frameC_v1.pt", pointing_accuracy_mrad=1.8)
    cond, _, _ = build_conditions(sun, pos, aim)
    torch.testing.assert_close(rays, proj(pred.predict(ids, cond)[:, 0], aim, cond[:, 0], powers))
    torch.testing.assert_close(rays[:, 6].view(3, -1).sum(1), powers)

    prepared = field.prepare(ids, sun, pos, aim)
    aim_var = aim.clone().requires_grad_(True)
    out = field.rays(prepared, aim_var, powers, group_indices=torch.tensor([0, 1, 1]))
    assert out.shape == (3 * 64 * 64, 8)
    (out[:, 0] * out[:, 6]).sum().backward()
    assert aim_var.grad is not None and torch.isfinite(aim_var.grad).all() and aim_var.grad.abs().sum() > 0


def test_field_places_spots_in_frame_c(artifact):
    root = artifact("transformer/T2026.pt").parents[1]
    field = HelioBeamField.from_artifacts(root, projection_cnn=None)
    spot = torch.zeros(1, 64, 64)
    spot[0, 0, 0] = 1.0                                          # top row, east column
    cond, _, _ = build_conditions(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([[0.0, 80.0, 1.6]]),
                                  torch.tensor([[0.0, 0.0, 55.0]]))
    from heliobeamai.field import PreparedSpots
    prepared = PreparedSpots(["X"], spot, None, torch.tensor([0.0, 0.0, 1.0]), torch.tensor([[0.0, 80.0, 1.6]]))
    rays = field.rays(prepared, torch.tensor([[0.0, 0.0, 55.0]]))
    o = rays[rays[:, 6].argmax(), :3]
    assert o[0] > 2.9 and o[2] > 57.9                           # east (+x) and up (+z) of the aim point
