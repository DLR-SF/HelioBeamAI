"""Transformer + VAE: predict focal spot images from calibration context."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from .transformer import FluxTransformer
from .vae import AutoencoderKL, VAEConfig


@dataclass
class FluxModelConfig:
    """Architecture and input normalization of a flux model.

    ``distance_normalization_m`` and ``image_mean`` describe how inputs were
    normalized for training; inference applies the same values.
    """
    cond_dim: int = 7
    d_model: int = 64
    ffn_hidden: int = 64
    n_head: int = 4
    n_layers_enc: int = 2
    n_layers_dec: int = 4
    dropout: float = 0.1
    decoder_dropout: float = 0.1
    distance_normalization_m: float = 350.0
    image_mean: Optional[float] = 0.1
    vae: VAEConfig = field(default_factory=VAEConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "FluxModelConfig":
        d = dict(d)
        vae = d.pop("vae", None)
        cfg = cls(**d)
        if vae is not None:
            cfg.vae = vae if isinstance(vae, VAEConfig) else VAEConfig(**{
                **vae, "ch_mult": tuple(vae.get("ch_mult", VAEConfig.ch_mult))})
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vae"] = self.vae.to_dict()
        return d


class FluxModel(nn.Module):
    """Predicts focal spots for target conditions from measured context.

    Shapes: images ``[B, S, H, W]``, conditions ``[B, S, cond_dim]`` with
    columns (sun_x, sun_y, sun_z, aim_dir_x, aim_dir_y, aim_dir_z,
    distance / distance_normalization_m).
    """

    def __init__(self, cfg: Optional[FluxModelConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or FluxModelConfig()
        self.vae = AutoencoderKL(self.cfg.vae)
        self.transformer = FluxTransformer(
            latent_dim=self.cfg.vae.latent_dim, cond_dim=self.cfg.cond_dim,
            d_model=self.cfg.d_model, ffn_hidden=self.cfg.ffn_hidden,
            n_head=self.cfg.n_head, n_layers_enc=self.cfg.n_layers_enc,
            n_layers_dec=self.cfg.n_layers_dec, dropout=self.cfg.dropout,
            decoder_dropout=self.cfg.decoder_dropout)

    # -- latent space --------------------------------------------------------
    def images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, S, H, W]`` -> ``[B, S, latent_dim]``."""
        b, s, h, w = images.shape
        latents = self.vae.to_latent(images.reshape(b * s, 1, h, w))
        return latents.reshape(b, s, -1)

    def latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """``[B, S, latent_dim]`` -> ``[B, S, H, W]``."""
        b, s, _ = latents.shape
        images = self.vae.from_latent(latents.reshape(b * s, *self.cfg.vae.latent_shape))
        return images.reshape(b, s, *images.shape[-2:])

    # -- transformer ---------------------------------------------------------
    def encode_context(self, images: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Encode measured context -> memory ``[B, S, d_model]``.

        The memory only depends on the context, so it can be cached per
        heliostat and reused for any number of predictions.
        """
        src = torch.cat([self.images_to_latents(images), conditions], dim=-1)
        return self.transformer.encoder(src)

    def decode(self, conditions: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Target conditions ``[B, T, cond_dim]`` + memory -> images ``[B, T, H, W]``."""
        return self.latents_to_images(self.transformer.decoder(conditions, memory))

    def forward(self, context_images: torch.Tensor, context_conditions: torch.Tensor,
                target_conditions: torch.Tensor) -> torch.Tensor:
        return self.decode(target_conditions,
                           self.encode_context(context_images, context_conditions))

    # -- persistence ---------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], map_location="cpu",
                        cfg: Optional[FluxModelConfig] = None) -> "FluxModel":
        """Load from a HelioBeamAI checkpoint or an original Lightning checkpoint."""
        from ..checkpoints import load_flux_model
        return load_flux_model(path, map_location=map_location, cfg=cfg)

    def save(self, path: Union[str, Path], **metadata) -> None:
        from ..checkpoints import save_flux_model
        save_flux_model(self, path, **metadata)
