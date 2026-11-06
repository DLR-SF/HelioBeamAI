"""Batching, transformer training and the shipped configs."""
import pytest
import torch

from heliobeamai.data import get_split, split_batch


@pytest.fixture
def synthetic(tmp_path):
    torch.manual_seed(0)
    s = 75
    ds = {"data": torch.rand(3, s, 7), "imgs": torch.rand(3, s, 8, 8),
          "ids": torch.arange(3 * s).view(3, s)}
    path = tmp_path / "ds.pt"
    torch.save(ds, path)
    return ds, str(path)


def test_train_batching_shapes(synthetic):
    ds, _ = synthetic
    src, trg, src_imgs, trg_imgs = split_batch(ds["data"][:, :20], ds["imgs"][:, :20], "train")
    assert src.shape[1] + trg.shape[1] == 20 and trg.shape[-1] == 7
    assert src_imgs.shape[:2] == src.shape[:2] and trg_imgs.shape[:2] == trg.shape[:2]


def test_training_smoke(tmp_path):
    pytest.importorskip("pytorch_lightning")
    from heliobeamai import FluxModel
    from heliobeamai.training.train_transformer import train
    torch.manual_seed(0)
    data = torch.rand(4, 20, 7)
    data[..., 6] = 100 + 200 * data[..., 6]
    torch.save({"data": data, "imgs": torch.rand(4, 20, 64, 64)}, tmp_path / "ds.pt")
    cfg = {"name": "smoke", "output": str(tmp_path / "runs"),
           "data": {"path": str(tmp_path / "ds.pt"), "split": "T7n5", "batch_size": 2, "num_workers": 0},
           "trainer": {"max_steps": 2, "val_check_interval": 1, "accelerator": "cpu", "devices": 1,
                       "enable_progress_bar": False}}
    out = train(cfg)
    model = FluxModel.from_checkpoint(out)
    assert model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 7), torch.rand(1, 2, 7)).shape == (1, 2, 64, 64)


def test_configs_are_valid():
    import yaml
    from heliobeamai.training.train_transformer import REPO_ROOT
    configs = sorted((REPO_ROOT / "configs" / "transformer").glob("*.yaml"))
    assert {p.stem for p in configs} >= {"T2026", "example"}
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        assert cfg["name"] == path.stem
        get_split(cfg["data"]["split"])
        assert isinstance(cfg["module"]["lr"], float) and isinstance(cfg["module"]["weight_decay"], float)
        init = cfg.get("init") or {}
        assert init.get("checkpoint") or init.get("vae"), path.name


def test_vae_training_smoke(tmp_path):
    pytest.importorskip("pytorch_lightning")
    from heliobeamai import load_vae
    from heliobeamai.training.vae import VAETrainConfig, load_images, train_vae
    torch.manual_seed(0)
    torch.save({"imgs": torch.rand(2, 6, 64, 64)}, tmp_path / "ds.pt")
    images = load_images([tmp_path / "ds.pt"])
    assert images.shape == (12, 1, 64, 64)
    torch.testing.assert_close(images.mean(dim=(-1, -2)), torch.full((12, 1), 0.1))
    out = train_vae(images, tmp_path / "vae", VAETrainConfig(epochs=1, batch_size=4, num_workers=0,
                                                             accelerator="cpu"),
                    enable_progress_bar=False, enable_model_summary=False)
    vae = load_vae(out)
    assert vae.to_latent(images[:2]).shape == (2, 1, 8, 8)
