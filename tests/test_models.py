"""Unit tests that need no artifacts."""
import torch

from heliobeamai import (FluxModel, FluxModelConfig, FluxPredictor, ProjectionConfig,
                         RayProjector, build_conditions, normalize_heliostat_id)


def test_shapes_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = FluxModel().eval()
    imgs, cond = torch.rand(2, 5, 64, 64), torch.rand(2, 5, 7)
    trg = torch.rand(2, 3, 7)
    out = model(imgs, cond, trg)
    assert out.shape == (2, 3, 64, 64)

    model.save(tmp_path / "m.ckpt", note="test")
    loaded = FluxModel.from_checkpoint(tmp_path / "m.ckpt")
    torch.testing.assert_close(loaded(imgs, cond, trg), out)


def test_decoder_queries_are_independent():
    torch.manual_seed(0)
    model = FluxModel().eval()
    imgs, cond, trg = torch.rand(1, 4, 64, 64), torch.rand(1, 4, 7), torch.rand(1, 3, 7)
    full = model(imgs, cond, trg)
    single = model(imgs, cond, trg[:, 1:2])
    torch.testing.assert_close(full[:, 1:2], single)


def test_config_dict_roundtrip():
    cfg = FluxModelConfig(n_layers_dec=2)
    assert FluxModelConfig.from_dict(cfg.to_dict()) == cfg


def test_heliostat_ids():
    assert normalize_heliostat_id(10123) == "AA23"
    assert normalize_heliostat_id(21005) == "BJ05"
    assert normalize_heliostat_id(torch.tensor(10123)) == "AA23"
    assert normalize_heliostat_id(" aa23 ") == "AA23"


def test_build_conditions():
    pos = torch.tensor([[0.0, -100.0, 0.0]])
    aim = torch.tensor([[0.0, 0.0, 100.0]])
    cond, dirs, lengths = build_conditions(torch.tensor([0.0, 0.0, 2.0]), pos, aim)
    assert cond.shape == (1, 1, 7)
    torch.testing.assert_close(cond[0, 0, :3], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(lengths, torch.tensor([[141.42136]]))
    torch.testing.assert_close(cond[0, 0, 6], lengths[0, 0] / 350.0)


def test_predictor_and_projection_without_net():
    torch.manual_seed(0)
    predictor = FluxPredictor(FluxModel())
    predictor.encode_context("AB01", torch.rand(6, 64, 64), torch.rand(6, 7) * 100)
    cond, dirs, lengths = build_conditions(torch.tensor([0.0, 0.0, 1.0]),
                                           torch.zeros(1, 3), torch.tensor([[0.0, 100.0, 50.0]]))
    spots = predictor.predict(["ab01"], cond)
    assert spots.shape == (1, 1, 64, 64)
    rays = RayProjector(ProjectionConfig())(spots[:, 0], torch.tensor([[0.0, 100.0, 50.0]]),
                                            cond[:, 0], powers=torch.tensor([10.0]))
    assert rays.shape == (64 * 64, 7)
    torch.testing.assert_close(rays[:, 6].sum(), torch.tensor(10.0))
    torch.testing.assert_close(rays[:, 3:6], dirs.expand(64 * 64, 3))
