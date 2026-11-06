"""Loading and saving of model checkpoints.

Supported transformer checkpoint formats:

* HelioBeamAI format: ``{"format": "heliobeamai.flux_model/1", "config": ...,
  "state_dict": ...}`` written by :func:`save_flux_model`.
* Lightning checkpoints written by HelioBeamAI training
  (``hyper_parameters["model_config"]``, keys ``model.transformer.*`` /
  ``model.vae.*``).
* Original Lightning checkpoints from the dissertation code
  (keys ``model.*`` for the transformer and ``vae.*`` for the autoencoder,
  no stored config). These contain decoder parameters that were never used in
  the forward pass (``self_attention``, ``norm1``, ``linear``); they are
  dropped on load.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import torch

from .models.flux_model import FluxModel, FluxModelConfig
from .models.projector import FocalSpotToDirectionsNet
from .models.vae import AutoencoderKL, VAEConfig

FLUX_FORMAT = "heliobeamai.flux_model/1"

# decoder parameters of the original implementation that never took part in
# the forward pass
_UNUSED_LEGACY_KEYS = re.compile(
    r"^transformer\.decoder\.(layers\.\d+\.(self_attention|norm1)\.|linear\.)")


def torch_load(path: Union[str, Path], map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def _legacy_to_flux_state_dict(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = "transformer." + key[len("model."):]
        elif not key.startswith("vae."):
            continue
        if not _UNUSED_LEGACY_KEYS.match(key):
            out[key] = value
    return out


def load_flux_model(path: Union[str, Path], map_location="cpu",
                    cfg: Optional[FluxModelConfig] = None) -> FluxModel:
    ckpt = torch_load(path, map_location)
    if isinstance(ckpt, dict) and ckpt.get("format") == FLUX_FORMAT:
        cfg = cfg or FluxModelConfig.from_dict(ckpt["config"])
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt.get("state_dict", ckpt)
        hparams = ckpt.get("hyper_parameters") or {}
        if "model_config" in hparams:
            cfg = cfg or FluxModelConfig.from_dict(hparams["model_config"])
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()
                          if k.startswith("model.")}
        else:  # original dissertation checkpoint: default architecture
            cfg = cfg or FluxModelConfig()
            state_dict = _legacy_to_flux_state_dict(state_dict)
    model = FluxModel(cfg)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def save_flux_model(model: FluxModel, path: Union[str, Path], **metadata) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FLUX_FORMAT, "config": model.cfg.to_dict(),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata": metadata}, path)


VAE_FORMAT = "heliobeamai.vae/1"


def load_vae(path: Union[str, Path], map_location="cpu") -> AutoencoderKL:
    """Load an autoencoder (:func:`save_vae` format or an original VAE Lightning checkpoint)."""
    ckpt = torch_load(path, map_location)
    if isinstance(ckpt, dict) and ckpt.get("format") == VAE_FORMAT:
        cfg = VAEConfig(**{k: tuple(v) if k == "ch_mult" else v for k, v in ckpt["config"].items()})
        vae = AutoencoderKL(cfg)
        vae.load_state_dict(ckpt["state_dict"], strict=True)
        return vae.eval()
    cfg = VAEConfig.from_hparams(ckpt["hyper_parameters"])
    state_dict = {k.replace("auto_encoder.", ""): v for k, v in ckpt["state_dict"].items()
                  if not k.startswith("loss.")}
    vae = AutoencoderKL(cfg)
    vae.load_state_dict(state_dict, strict=True)
    return vae.eval()


def save_vae(vae: AutoencoderKL, path: Union[str, Path], **metadata) -> None:
    """Weights and architecture only (no optimizer state, CPU tensors)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": VAE_FORMAT, "config": vae.cfg.to_dict(),
                "state_dict": {k: v.detach().cpu() for k, v in vae.state_dict().items()},
                "metadata": metadata}, path)


PROJECTOR_FORMAT = "heliobeamai.projector/1"


def load_projector(path: Union[str, Path], map_location="cpu", with_config: bool = False):
    """Load the projection CNN.

    Accepts HelioBeamAI projector checkpoints (with the projection settings
    the CNN was trained with), plain state dicts and ``{"state_dict": ...}``.
    With ``with_config`` returns ``(net, settings dict or None)``.
    """
    ckpt = torch_load(path, map_location)
    settings = None
    if isinstance(ckpt, dict) and ckpt.get("format") == PROJECTOR_FORMAT:
        settings = ckpt.get("projection_config")
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        ckpt = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    net = FocalSpotToDirectionsNet(cond_dim=ckpt["conv1.weight"].shape[1] - 1)
    net.load_state_dict(ckpt, strict=True)
    net.eval()
    return (net, settings) if with_config else net


def save_projector(net: FocalSpotToDirectionsNet, path: Union[str, Path], projection_config,
                   **metadata) -> None:
    """Save the CNN together with the settings it must be applied with."""
    from dataclasses import asdict
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": PROJECTOR_FORMAT,
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "projection_config": asdict(projection_config), "metadata": metadata}, path)
