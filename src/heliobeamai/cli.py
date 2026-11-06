"""Command line interface: ``heliobeamai <command> ...`` (or ``python -m heliobeamai``).

Commands
  build-dataset      calibration flux dataset for transformer training
  train-transformer  train the transformer from a YAML config
  encode             encoder cache (context of every heliostat) for fast prediction
  train-projector    train the projection CNN
  eval-projector     evaluate projection CNNs
  predict            flux density of heliostats on a flat receiver (DNI, mirror area, reflectivity)
  train-vae          train the autoencoder that compresses the flux images
  download           download the released model files (prediction | training | all)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_path(p):
    if p is None:
        return None
    q = Path(p).expanduser()
    return str(q if q.is_absolute() or q.exists() else REPO_ROOT / q)


def build_dataset(argv):
    from .data.build_calibration import (TARGET_GROUPS_2026, TARGET_GROUPS_DISSERTATION,
                                         CalibrationDatasetConfig, build_calibration_dataset)
    import torch
    d = CalibrationDatasetConfig()
    p = argparse.ArgumentParser(prog="heliobeamai build-dataset", description=build_calibration_dataset.__module__)
    p.add_argument("--output", required=True)
    p.add_argument("--groups", choices=("2026", "dissertation"), default="2026",
                   help="target groups and block order: 2026 = (MFT, Lower, Upper); "
                        "dissertation = (3, 4/5/6, 1/7)")
    p.add_argument("--samples-per-target", type=int, default=d.samples_per_target)
    p.add_argument("--min-total-samples", type=int, default=d.min_total_samples)
    p.add_argument("--calibration-csv", default=d.calibration_csv)
    p.add_argument("--heliostat-positions", default=d.heliostat_positions)
    p.add_argument("--flux-dir", default=d.flux_dir)
    p.add_argument("--vae", default=d.vae_checkpoint, help="VAE checkpoint, 'none' to skip")
    p.add_argument("--outlier-scores", default=None, help="score CSV (index id) for the outlier filter")
    p.add_argument("--outlier-column", default=d.outlier_score_column)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit-heliostats", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    groups = TARGET_GROUPS_2026 if a.groups == "2026" else TARGET_GROUPS_DISSERTATION
    cfg = CalibrationDatasetConfig(
        calibration_csv=a.calibration_csv, heliostat_positions=a.heliostat_positions, flux_dir=a.flux_dir,
        vae_checkpoint=None if a.vae in (None, "none") else _repo_path(a.vae),
        target_groups=dict(groups), group_order=tuple(groups),
        samples_per_target=a.samples_per_target, min_total_samples=a.min_total_samples,
        outlier_scores_csv=a.outlier_scores, outlier_score_column=a.outlier_column, seed=a.seed,
        limit_heliostats=a.limit_heliostats, device=a.device)
    ds = build_calibration_dataset(cfg)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ds, a.output)
    print(f"{ds['imgs'].shape[0]} heliostats x {ds['imgs'].shape[1]} samples -> {a.output}  {ds['build_stats']}")


def train_transformer(argv):
    import yaml
    from .training.train_transformer import load_config, train
    p = argparse.ArgumentParser(prog="heliobeamai train-transformer")
    p.add_argument("config")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config entries, e.g. trainer.max_epochs=10 data.batch_size=8")
    a = p.parse_args(argv)
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    train(load_config(a.config, **overrides))


def encode(argv):
    import torch
    from .inference.predictor import FluxPredictor
    p = argparse.ArgumentParser(prog="heliobeamai encode",
                                description="Encode every heliostat of a dataset into an encoder cache.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    pred = FluxPredictor.from_checkpoint(_repo_path(a.checkpoint), device=a.device)
    pred.encode_dataset(a.dataset)
    pred.save_encoder_cache(a.output, checkpoint=str(a.checkpoint), dataset=str(a.dataset))
    print(f"{len(pred.contexts)} heliostats -> {a.output}")


def predict(argv):
    import numpy as np
    import torch
    from .field import HelioBeamField
    from .predict import read_heliostats, receiver_flux, sun_vector
    from .receiver import OVRReceiver
    from .sources import GaussianSource
    p = argparse.ArgumentParser(
        prog="heliobeamai predict",
        description="Flux density of a heliostat field on a flat receiver (transformer spots, projection "
                    "CNN, DNI x mirror area x reflectivity x cosine). Writes <output>.npz and <output>.png.")
    p.add_argument("--heliostats", required=True,
                   help="CSV: heliostat_id,x,y,z [,aim_x,aim_y,aim_z] [,mirror_area] [,reflectivity] "
                        "(tower-base frame: x east, y north, z up, m)")
    p.add_argument("--aim", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="aim point for all heliostats (if the CSV has no aim columns)")
    sun = p.add_mutually_exclusive_group(required=True)
    sun.add_argument("--sun", nargs=2, type=float, metavar=("AZIMUTH", "ELEVATION"),
                     help="sun position in degrees, azimuth from north clockwise")
    sun.add_argument("--sun-vector", nargs=3, type=float, metavar=("X", "Y", "Z"), help="vector towards the sun")
    p.add_argument("--dni", type=float, required=True, help="direct normal irradiance, W/m^2")
    p.add_argument("--mirror-area", type=float, help="m^2 per heliostat (if not in the CSV)")
    p.add_argument("--reflectivity", type=float, default=1.0, help="if not in the CSV (default 1)")
    p.add_argument("--context", help="flux dataset (.pt, keys imgs/data/heliostat_ids) to encode heliostats "
                                     "from their calibration measurements, in addition to the encoder cache")
    p.add_argument("--fallback", nargs=2, type=float, metavar=("SIGMA_MRAD", "SIGMA0_M"),
                   help="Gaussian beam for heliostats without context: sqrt(sigma0^2 + (sigma_mrad d)^2)")
    p.add_argument("--receiver-center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="aperture centre (default: mean aim point)")
    p.add_argument("--receiver-size", nargs=2, type=float, default=(8.0, 8.0), metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--receiver-tilt", type=float, default=0.0, help="degrees, tilted down towards the field")
    p.add_argument("--receiver-azimuth", type=float, default=0.0, help="facing direction, degrees from north")
    p.add_argument("--resolution", nargs=2, type=int, default=(80, 80), metavar=("ROWS", "COLS"))
    p.add_argument("--pointing-accuracy", type=float, default=0.0, help="tracking error, mrad (default 0)")
    p.add_argument("--projection-cnn", default="projector/cnn_frameC_v1.pt", help="'none' for straight rays")
    p.add_argument("--upsampling", type=int, default=1, help="rays per spot pixel = upsampling^2")
    p.add_argument("--output", required=True, help="output prefix")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    h = read_heliostats(a.heliostats)
    aims = h["aim_points"]
    if aims is None:
        if a.aim is None:
            p.error("give --aim or aim_x/aim_y/aim_z columns")
        aims = torch.tensor(a.aim).expand(len(h["heliostat_ids"]), 3).clone()
    area = h["mirror_area"] if h["mirror_area"] is not None else a.mirror_area
    if area is None:
        p.error("give --mirror-area or a mirror_area column")
    rho = h["reflectivity"] if h["reflectivity"] is not None else a.reflectivity
    s = sun_vector(*a.sun) if a.sun else torch.nn.functional.normalize(torch.tensor(a.sun_vector), dim=0)
    if float(s[2]) <= 0:
        p.error("the sun is below the horizon")

    fallback = GaussianSource.field_law(a.fallback[0], a.fallback[1]) if a.fallback else None
    cnn = None if a.projection_cnn.lower() == "none" else a.projection_cnn
    field = HelioBeamField.from_artifacts(device=a.device, projection_cnn=cnn, fallback=fallback,
                                          pointing_accuracy_mrad=a.pointing_accuracy, upsampling=a.upsampling)
    if a.context:
        field.predictor.encode_dataset(_repo_path(a.context))
    field.require_coverage(h["heliostat_ids"])
    center = a.receiver_center or aims.mean(0).tolist()
    receiver = OVRReceiver(center, *a.receiver_size, resolution=tuple(a.resolution),
                           tilt_deg=a.receiver_tilt, azimuth_deg=a.receiver_azimuth)
    out = receiver_flux(field, receiver, h["heliostat_ids"], h["positions"], aims, s, a.dni, area, rho)

    prefix = Path(a.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    flux = out["flux"].numpy()
    np.savez(prefix.with_suffix(".npz"), flux_w_m2=flux, extent_m=np.array(receiver.extent()),
             receiver_center=np.array(center), receiver_size=np.array(a.receiver_size),
             heliostat_ids=np.array(h["heliostat_ids"]), powers_w=out["powers"].numpy(),
             sources=np.array(out["sources"]), sun=s.numpy(), intercepted=out["intercepted"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        receiver.visualize_flux_distribution(out["flux"], ax=ax)
        ax.set_title(f"{len(h['heliostat_ids'])} heliostats, {float(out['powers'].sum()) / 1e3:.0f} kW reflected, "
                     f"{100 * out['intercepted']:.1f} % intercepted", fontsize=9)
        fig.tight_layout()
        fig.savefig(prefix.with_suffix(".png"), dpi=120)
    except ImportError:
        print("matplotlib not installed: no plot")
    counts = {k: out["sources"].count(k) for k in dict.fromkeys(out["sources"])}
    print(f"{counts}; peak {flux.max() / 1e3:.1f} kW/m^2, intercepted {100 * out['intercepted']:.1f} % "
          f"-> {prefix.with_suffix('.npz')}")
    return 0


def train_vae(argv):
    from .training.vae import VAETrainConfig, load_images, train_vae as train
    d = VAETrainConfig()
    p = argparse.ArgumentParser(prog="heliobeamai train-vae",
                                description="Train the flux-image autoencoder (port of the original script).")
    p.add_argument("--images", nargs="+", required=True,
                   help="PNG folders, flux datasets (.pt with imgs) or image tensors (.pt)")
    p.add_argument("--output", required=True, help="output folder (vae.pt, checkpoints, logs)")
    p.add_argument("--ch", type=int, default=d.ch, help="4: z64CH4 (transformer VAE), 16: z64 (builder VAE)")
    p.add_argument("--embed-dim", type=int, default=d.embed_dim)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--kl-weight", type=float, default=d.kl_weight)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--limit", type=int, help="use at most N images")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--accelerator", default=d.accelerator)
    a = p.parse_args(argv)
    images = load_images([_repo_path(x) for x in a.images], limit=a.limit)
    print(f"{len(images)} images, mean {float(images.mean()):.3f}")
    cfg = VAETrainConfig(ch=a.ch, embed_dim=a.embed_dim, lr=a.lr, batch_size=a.batch_size, epochs=a.epochs,
                         val_fraction=a.val_fraction, kl_weight=a.kl_weight, seed=a.seed,
                         num_workers=a.num_workers, accelerator=a.accelerator)
    train(images, a.output, cfg)
    return 0


def download(argv):
    from . import artifacts
    ap = argparse.ArgumentParser(prog="heliobeamai download", description="Download released model files.")
    ap.add_argument("groups", nargs="*", default=["prediction"], help="prediction (default), training or all")
    ap.add_argument("--root", help="target folder (default: heliobeamai.artifacts.root())")
    ap.add_argument("--url", help="base URL (default: GitHub release or $HELIOBEAMAI_ARTIFACTS_URL)")
    ap.add_argument("--verify", action="store_true", help="only check local files")
    a = ap.parse_args(argv)
    if a.verify:
        status = artifacts.verify(a.groups, a.root)
        for path, s in status.items():
            print(f"{s:20s} {path}")
        return 0 if all(s == "ok" for s in status.values()) else 1
    print(f"model files in {artifacts.download(a.groups, a.root, a.url)}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"build-dataset": build_dataset, "train-transformer": train_transformer, "encode": encode,
                "download": download, "predict": predict, "train-vae": train_vae}
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in (*commands, "train-projector", "eval-projector"):
        print(__doc__)
        return 0 if argv[:1] in ([], ["-h"], ["--help"]) else 2
    cmd, rest = argv[0], argv[1:]
    if cmd in ("train-projector", "eval-projector"):
        from .training.projector_cli import main as projector_main
        projector_main(rest, command=cmd.split("-")[0])
    else:
        return commands[cmd](rest) or 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
