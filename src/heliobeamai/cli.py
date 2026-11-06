"""Command line interface: ``heliobeamai <command> ...`` (or ``python -m heliobeamai``).

Commands
  build-dataset      calibration flux dataset for transformer training
  train-transformer  train the transformer from a YAML config
  encode             encoder cache (context of every heliostat) for fast prediction
  train-projector    train the projection CNN
  eval-projector     evaluate projection CNNs
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
                "download": download}
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
