"""Train the flux transformer from a YAML config (``configs/transformer/*.yaml``).

Config keys::

    name: T367n20                         # run name (output folder, checkpoint prefix)
    data:  {path, split, batch_size, num_workers, distance_normalization_m, artificial_path}
    model: {...}                          # FluxModelConfig fields (defaults: dissertation model)
    init:  {checkpoint: ..., vae: ...}    # warm start / initial VAE
    module: {lr, weight_decay, freeze_vae, n_val, input_noise}
    trainer: {max_epochs, val_check_interval, accelerator, devices, precision}
    output: runs/transformer

Relative input paths are resolved against the working directory, else against
the HelioBeamAI repository (source checkout, e.g. ``artifacts/...``); ``output``
is relative to the working directory. The best checkpoint
(``loss_val_ep``) is exported as ``<output>/<name>/model.pt`` (HelioBeamAI format).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: Optional[str]) -> Optional[str]:
    """Input path: absolute, or relative to the working directory, else to the repository."""
    if not path:
        return None
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists() or not (REPO_ROOT / p).exists():
        return str(p.absolute())
    return str(REPO_ROOT / p)


def load_config(path: Union[str, Path], **overrides) -> Dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for key, value in overrides.items():           # dotted keys, e.g. trainer.max_epochs=10
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return cfg


def train(cfg: Dict) -> Path:
    import importlib.util

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    from ..checkpoints import load_flux_model, save_flux_model
    from ..data.flux_data import FluxDataConfig, FluxDataModule
    from .lit_transformer import FluxTransformerModule

    torch.set_float32_matmul_precision("medium")
    name = cfg["name"]
    out = Path(cfg.get("output", "runs/transformer")).expanduser().absolute() / name
    data_cfg = dict(cfg["data"])
    data_cfg["path"] = _resolve(data_cfg["path"])
    data_cfg["artificial_path"] = _resolve(data_cfg.get("artificial_path"))
    datamodule = FluxDataModule(FluxDataConfig(**data_cfg))
    init = cfg.get("init", {}) or {}
    module = FluxTransformerModule(model_config=cfg.get("model"), init_checkpoint=_resolve(init.get("checkpoint")),
                                   vae_checkpoint=_resolve(init.get("vae")),
                                   with_artificial=bool(data_cfg["artificial_path"]),
                                   **(cfg.get("module", {}) or {}))
    ckpt = ModelCheckpoint(dirpath=out / "checkpoints", filename=name + "-{epoch:05d}-{loss_val_ep:.4f}",
                           monitor="loss_val_ep", mode="min", save_top_k=1, save_last=True)
    logger = (TensorBoardLogger(str(out), name="tb") if importlib.util.find_spec("tensorboard")
              else CSVLogger(str(out), name="csv"))
    trainer = pl.Trainer(logger=logger, callbacks=[ckpt],
                         **(cfg.get("trainer", {}) or {}))
    trainer.fit(module, datamodule)
    best = ckpt.best_model_path or ckpt.last_model_path
    target = out / "model.pt"
    save_flux_model(load_flux_model(best), target, run=name, source_checkpoint=str(best),
                    best_loss_val_ep=float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
                    split=data_cfg.get("split"), data=data_cfg["path"])
    print(f"best checkpoint {best} -> {target}")
    return target
