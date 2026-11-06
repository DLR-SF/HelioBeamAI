"""Train or evaluate the projection CNN (``heliobeamai train-projector`` / ``eval-projector``).

    heliobeamai train-projector --data 'data/projection_v3/receiver_*.pt' \\
        --conditions full --output runs/projector/A_full.pt
    heliobeamai eval-projector --data '...receiver_*.pt' --models runs/projector/*.pt \\
        --upsampling 1 2 4 --grids 64 32 16
"""
import argparse
import glob
import json
from dataclasses import replace
from pathlib import Path

import torch

from heliobeamai import ProjectionConfig, RayProjector
from heliobeamai.training.projector import (ProjectionData, ProjectorTrainConfig, evaluate_projector,
                                            split_by_heliostat, train_projector)


def main(argv=None, command=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if command is None:
        p.add_argument("command", choices=("train", "eval"))
    p.add_argument("--data", required=True, help="glob of data chunk files")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smooth", type=float, default=1.0, help="Gaussian sigma [px] applied to all planes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # train
    p.add_argument("--conditions", choices=("full", "aim"), default="full")
    p.add_argument("--input-vae", default=None,
                   help="standalone VAE checkpoint (artifacts/vae/z64CH4.ckpt) whose round trip shapes the input")
    p.add_argument("--cnn-upsampling", type=int, default=1, help="train the CNN on upsampled spots")
    p.add_argument("--ray-upsampling", type=int, default=1,
                   help="CNN on the native spot, more rays (interpolated deviations) in the loss")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", default="runs/projector/projector.pt")
    # eval
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--upsampling", type=int, nargs="*", default=[1])
    p.add_argument("--grids", type=int, nargs="*", default=[64, 32, 16])
    p.add_argument("--eval-input-vae", default=None,
                   help="evaluate all models on VAE round-trip inputs (transformer-like spots)")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    if command is not None:
        a.command = command

    files = sorted(glob.glob(a.data))
    data = ProjectionData.load(files)
    print(f"{len(data)} cases from {len(files)} files", flush=True)

    if a.command == "train":
        if a.cnn_upsampling > 1 and a.ray_upsampling > 1:
            raise SystemExit("use either --cnn-upsampling or --ray-upsampling")
        proj = ProjectionConfig(conditions=a.conditions, upsampling=max(a.cnn_upsampling, a.ray_upsampling),
                                upsampling_mode="cnn" if a.cnn_upsampling > 1 else "rays")
        cfg = ProjectorTrainConfig(projection=proj, smooth_sigma_px=a.smooth, input_vae_checkpoint=a.input_vae,
                                   val_fraction=a.val_fraction, epochs=a.epochs, lr=a.lr,
                                   batch_size=a.batch_size, seed=a.seed, device=a.device, output=a.output)
        train_projector(cfg, data)
        return

    _, val = split_by_heliostat(data.smoothed(a.smooth), a.val_fraction, a.seed)
    if a.eval_input_vae:
        from heliobeamai import load_vae
        val = val.with_vae_input(load_vae(a.eval_input_vae).to(a.device))
        print(f"all inputs through VAE {a.eval_input_vae}", flush=True)
    print(f"validation: {len(val)} cases, {len(set(val.heliostat_names))} heliostats", flush=True)
    results = {}
    candidates = [("straight rays", None)] + [(m, m) for m in a.models]
    for name, path in candidates:
        for up in a.upsampling:
            projector = RayProjector.from_checkpoint(path, device=a.device)
            if path is not None and projector.config.upsampling_mode == "cnn":
                if up != a.upsampling[0]:
                    continue                     # trained for one CNN resolution
            else:
                projector.config = replace(projector.config, upsampling=up, upsampling_mode="rays")
            vdata = val
            vae = torch.load(path, map_location="cpu", weights_only=False).get("metadata", {}).get("input_vae_checkpoint") if path else None
            if vae and not a.eval_input_vae:
                from heliobeamai import load_vae
                vdata = val.with_vae_input(load_vae(vae).to(a.device))
            losses = evaluate_projector(projector, vdata, grids=a.grids)
            key = f"{Path(name).stem if path else name} | rays x{projector.config.upsampling}"
            results[key] = {g: float(v.mean()) for g, v in losses.items()}
            print(f"{key:45s} " + "  ".join(f"grid {g}: {v:.5f}" for g, v in results[key].items()), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
