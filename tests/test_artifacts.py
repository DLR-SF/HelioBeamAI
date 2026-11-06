"""Release manifest, download/verification and the exported (clean) model files."""
from pathlib import Path

import pytest
import torch

from heliobeamai import artifacts
from heliobeamai.checkpoints import load_vae, save_vae

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_manifest_covers_prediction_files():
    paths = {a.path for a in artifacts.select("prediction")}
    assert paths == {"transformer/T2026.pt", "encoder_cache/T2026.pt", "projector/cnn_frameC_v1.pt"}
    assert all(len(a.sha256) == 64 and a.size > 0 for a in artifacts.MANIFEST.values())
    assert len({a.asset for a in artifacts.MANIFEST.values()}) == len(artifacts.MANIFEST)
    with pytest.raises(ValueError):
        artifacts.select("nonsense")


def test_download_from_a_mirror_checks_checksums(tmp_path, artifact):
    src = artifact("transformer/T2026.pt").parents[1]
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    for a in artifacts.select("prediction"):
        (mirror / a.asset).symlink_to(src / a.path)
    out = tmp_path / "artifacts"
    artifacts.download("prediction", out, base_url=mirror.as_uri(), quiet=True)
    assert set(artifacts.verify("prediction", out).values()) == {"ok"}
    bad = out / "projector/cnn_frameC_v1.pt"
    bad.write_bytes(b"corrupt")
    assert artifacts.verify("prediction", out)["projector/cnn_frameC_v1.pt"] == "checksum mismatch"
    with pytest.raises(RuntimeError, match="checksum"):
        artifacts.download("prediction", out, base_url=mirror.as_uri(), quiet=True)


def test_require_names_the_download_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="heliobeamai download prediction"):
        artifacts.require("transformer/T2026.pt", tmp_path)


def test_vae_format_round_trip(tmp_path):
    from heliobeamai import AutoencoderKL
    torch.manual_seed(0)
    vae = AutoencoderKL().eval()
    save_vae(vae, tmp_path / "vae.pt")
    x = torch.rand(2, 1, 64, 64)
    torch.testing.assert_close(load_vae(tmp_path / "vae.pt").encode(x).mode(), vae.encode(x).mode())


def test_example_data_loads_with_the_package_loaders():
    from heliobeamai import normalize_heliostat_id
    from heliobeamai.training.projector import ProjectionData
    cal = torch.load(EXAMPLES / "calibration_2026.pt", weights_only=False)
    assert [normalize_heliostat_id(int(h)) for h in cal["heliostat_ids"]] == ["AC26", "AA44", "AO50", "AY27", "BC71"]
    assert cal["imgs"].shape == (5, 45, 64, 64) and cal["data"].shape == (5, 45, 7)
    p = ProjectionData.load([EXAMPLES / "projection_receiver.pt", EXAMPLES / "projection_calibration.pt"])
    assert p.fluxes.shape[1:] == (6, 64, 64) and p.fluxes.dtype == torch.float32
