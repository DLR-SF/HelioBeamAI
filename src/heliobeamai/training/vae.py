"""Train the autoencoder (VAE) that compresses 64 x 64 flux images for the transformer.

Port of the original training script (``VAE_local/train.py`` of the
dissertation code); the released VAEs were trained this way:

* images: flux images (frame C, 6 m x 6 m around the spot), each scaled to
  mean 0.1, resized to 64 x 64; random 80 / 20 train / validation split;
* loss: mean squared reconstruction error with a posterior sample (the
  original configured a KL term but never used it: ``kl_weight=0``);
  validation: mean absolute error;
* Adam, lr 1e-3, betas (0.5, 0.9), batch 256, 1000 epochs, best validation
  checkpoint.

``z64CH4`` (the transformers' VAE): ``ch=4``; ``z64`` (dataset builder):
``ch=16`` (both ``z_channels=1``, ``embed_dim=4``, 1 x 8 x 8 latent).

    heliobeamai train-vae --images data/fluxTransData2026.pt --output runs/vae/z64CH4 --ch 4
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from ..checkpoints import save_vae
from ..models.vae import AutoencoderKL, VAEConfig

PathLike = Union[str, Path]


def load_images(paths: Sequence[PathLike], size: int = 64, image_mean: float = 0.1,
                limit: Optional[int] = None) -> torch.Tensor:
    """Flux images ``[N, 1, size, size]``, each scaled to ``image_mean``.

    Accepts folders of PNG files (grayscale flux images), flux datasets
    (``.pt`` with ``imgs [..., H, W]``, e.g. from ``heliobeamai build-dataset``)
    and tensor files (``[N, H, W]`` or ``[N, 1, H, W]``).
    """
    parts: List[torch.Tensor] = []
    for path in map(Path, paths):
        if path.is_dir():
            import numpy as np
            from PIL import Image
            files = sorted(path.glob("*.png"))
            imgs = [torch.from_numpy(np.asarray(Image.open(f).convert("L"), dtype=np.float32)) for f in files]
            if not imgs:
                raise ValueError(f"no PNG files in {path}")
            x = torch.stack([F.interpolate(i[None, None], size=(size, size), mode="bilinear",
                                           align_corners=False)[0, 0] for i in imgs])
        else:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            x = obj["imgs"] if isinstance(obj, dict) else obj
            x = x.float().reshape(-1, *x.shape[-2:])
            if tuple(x.shape[-2:]) != (size, size):
                x = F.interpolate(x[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
        parts.append(x)
    x = torch.cat(parts)[:, None]
    m = x.mean(dim=(-1, -2), keepdim=True)
    x = x[(m > 0).flatten()]
    x = x * (image_mean / x.mean(dim=(-1, -2), keepdim=True))
    return x[:limit] if limit else x


@dataclass
class VAETrainConfig:
    ch: int = 4                      # 4: z64CH4 (transformer VAE), 16: z64 (dataset builder VAE)
    embed_dim: int = 4
    z_channels: int = 1
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 1000
    val_fraction: float = 0.2
    kl_weight: float = 0.0           # original: 0 (KL configured but not used)
    seed: int = 0
    num_workers: int = 4
    accelerator: str = "auto"
    devices: int = 1


def _module(vae: AutoencoderKL, cfg: VAETrainConfig):
    import pytorch_lightning as pl

    class VAEModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.vae = vae

        def training_step(self, x, _):
            rec, posterior = self.vae(x)
            loss = F.mse_loss(rec, x)
            if cfg.kl_weight > 0:
                loss = loss + cfg.kl_weight * posterior.kl().mean()
            self.log("mse_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, x, _):
            rec, _ = self.vae(x)
            loss = F.l1_loss(rec, x)
            self.log("val_l1_loss", loss, prog_bar=True, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.vae.parameters(), lr=cfg.lr, betas=(0.5, 0.9))

    return VAEModule()


def train_vae(images: torch.Tensor, output: PathLike, cfg: Optional[VAETrainConfig] = None,
              **trainer_kwargs) -> Path:
    """Train on ``images [N, 1, 64, 64]``; returns ``<output>/vae.pt`` (best validation epoch)."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, random_split

    cfg = cfg or VAETrainConfig()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    vae_cfg = replace(VAEConfig(), ch=cfg.ch, embed_dim=cfg.embed_dim, z_channels=cfg.z_channels,
                      resolution=int(images.shape[-1]))
    module = _module(AutoencoderKL(vae_cfg), cfg)
    n_val = max(1, int(round(cfg.val_fraction * len(images))))
    train_set, val_set = random_split(images, [len(images) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(cfg.seed))
    ckpt = pl.callbacks.ModelCheckpoint(dirpath=out / "checkpoints", monitor="val_l1_loss", mode="min",
                                        save_top_k=1, filename="vae-{epoch}-{val_l1_loss:.5f}")
    try:
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(str(out), name="logs")
    except Exception:                                   # tensorboard not installed
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(str(out), name="logs")
    kwargs = dict(max_epochs=cfg.epochs, accelerator=cfg.accelerator, devices=cfg.devices,
                  callbacks=[ckpt], logger=logger)
    kwargs.update(trainer_kwargs)
    trainer = pl.Trainer(**kwargs)
    trainer.fit(module, DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                                   num_workers=cfg.num_workers),
                DataLoader(val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers))
    if ckpt.best_model_path:
        state = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
        module.load_state_dict(state)
    target = out / "vae.pt"
    save_vae(module.vae.cpu().eval(), target, images=len(images), best=str(ckpt.best_model_path),
             best_val_l1=float(ckpt.best_model_score) if ckpt.best_model_score is not None else None)
    print(f"VAE (latent {module.vae.cfg.latent_shape}) -> {target}")
    return target
