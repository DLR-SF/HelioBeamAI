"""Lightning module for training the flux transformer."""
from __future__ import annotations

from typing import Dict, Optional

import pytorch_lightning as pl
import torch

from ..checkpoints import load_flux_model, load_vae
from ..data.batching import split_batch, split_batch_artificial
from ..models.flux_model import FluxModel, FluxModelConfig
from .losses import relative_flux_deviation


class FluxTransformerModule(pl.LightningModule):
    """Trains :class:`FluxModel` (transformer + VAE) on measurement sequences.

    Args:
        model_config: :class:`FluxModelConfig` as dict.
        lr, weight_decay: AdamP settings.
        init_checkpoint: warm start from a transformer checkpoint (any format).
        vae_checkpoint: initial VAE weights (standalone VAE checkpoint) when
            not warm starting.
        freeze_vae: keep the VAE fixed. The original training (all
            dissertation and T2026 models) optimized it jointly (False).
        n_val: targets per sequence in validation (``loss_val_ep``, ``loss_test_ep``).
        input_noise: Gaussian noise on context images.
        with_artificial: add the loss on artificial sequences (data module
            with ``artificial_path``).
    """

    def __init__(self, model_config: Optional[Dict] = None, lr: float = 1e-5, weight_decay: float = 1e-8,
                 init_checkpoint: Optional[str] = None, vae_checkpoint: Optional[str] = None,
                 freeze_vae: bool = False, n_val: int = 2, input_noise: float = 0.01,
                 with_artificial: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters()
        if init_checkpoint:
            self.model = load_flux_model(init_checkpoint)
        else:
            self.model = FluxModel(FluxModelConfig.from_dict(model_config or {}))
            if vae_checkpoint:
                self.model.vae.load_state_dict(load_vae(vae_checkpoint).state_dict())
        # stored in the Lightning checkpoint, read by heliobeamai.load_flux_model
        self.hparams.model_config = self.model.cfg.to_dict()
        if freeze_vae:
            self.model.vae.requires_grad_(False)

    def _loss(self, src, trg, src_imgs, trg_imgs) -> torch.Tensor:
        return relative_flux_deviation(trg_imgs, self.model(src_imgs, src, trg)).mean()

    def training_step(self, batch, batch_idx):
        loss = self._loss(*split_batch(batch["data"], batch["imgs"], "train",
                                       input_noise=self.hparams.input_noise))
        self.log("loss_flux", loss, on_epoch=True)
        if self.hparams.with_artificial and "data_art" in batch:
            loss_art = self._loss(*split_batch_artificial(batch["data_art"], batch["imgs_art"],
                                                          input_noise=self.hparams.input_noise))
            self.log("loss_flux_art", loss_art, on_epoch=True)
            loss = loss + loss_art
        return loss

    def validation_step(self, batch, batch_idx):
        for mode, key in (("test", "loss_test_ep"), ("val", "loss_val_ep")):
            loss = self._loss(*split_batch(batch["data"], batch["imgs"], mode, n_val=self.hparams.n_val,
                                           input_noise=self.hparams.input_noise))
            self.log(key, loss, on_epoch=True, batch_size=batch["data"].shape[0])

    def configure_optimizers(self):
        from .adamp import AdamP
        params = [p for p in self.parameters() if p.requires_grad]
        return AdamP(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
