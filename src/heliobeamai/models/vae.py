"""KL autoencoder for 64x64 focal spot images.

Derived from the latent-diffusion ``AutoencoderKL`` (encoder/decoder with
residual blocks, GroupNorm with a single group, one mid attention block).
Only the parts used by the flux models are kept; parameter names are unchanged
so the existing ``z64``/``z64CH4`` checkpoints load directly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-6, affine=True)


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Upsample(nn.Module):

    def __init__(self, channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.with_conv else x


class Downsample(nn.Module):

    def __init__(self, channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            return self.conv(F.pad(x, (0, 1, 0, 1)))
        return F.avg_pool2d(x, 2, 2)


class ResnetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = _normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.nin_shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                             if in_channels != out_channels else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(_swish(self.norm1(x)))
        h = self.conv2(self.dropout(_swish(self.norm2(h))))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Single-head spatial self-attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, height, width = q.shape
        q = q.reshape(b, c, height * width).permute(0, 2, 1)
        k = k.reshape(b, c, height * width)
        w = torch.softmax(torch.bmm(q, k) * c ** -0.5, dim=2)
        h = torch.bmm(v.reshape(b, c, height * width), w.permute(0, 2, 1))
        return x + self.proj_out(h.reshape(b, c, height, width))


def _make_attn(channels: int, attn_type: str) -> nn.Module:
    if attn_type == "vanilla":
        return AttnBlock(channels)
    if attn_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported attn_type '{attn_type}'")


@dataclass
class VAEConfig:
    """Architecture of the autoencoder (``ddconfig`` + ``embed_dim``).

    Defaults are the ``z64CH4`` model used by all transformer checkpoints:
    64x64 images -> 1x8x8 latent (64 values).
    """
    ch: int = 4
    out_ch: int = 1
    ch_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 1
    attn_resolutions: Sequence[int] = field(default_factory=list)
    dropout: float = 0.2
    resamp_with_conv: bool = True
    in_channels: int = 1
    resolution: int = 64
    z_channels: int = 1
    double_z: bool = True
    attn_type: str = "vanilla"
    tanh_out: bool = False
    embed_dim: int = 4

    @classmethod
    def from_hparams(cls, hparams: dict) -> "VAEConfig":
        """Build from the ``hyper_parameters`` of an original VAE checkpoint."""
        dd = dict(hparams.get("ddconfig", hparams))
        if dd.pop("use_linear_attn", False):
            raise ValueError("Linear attention VAEs are not supported.")
        dd.pop("give_pre_end", None)
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in dd.items() if k in known}
        if "ch_mult" in kwargs:
            kwargs["ch_mult"] = tuple(kwargs["ch_mult"])
        if "embed_dim" in hparams:
            kwargs["embed_dim"] = hparams["embed_dim"]
        return cls(**kwargs)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ch_mult"] = list(d["ch_mult"])
        d["attn_resolutions"] = list(d["attn_resolutions"])
        return d

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        res = self.resolution // 2 ** (len(self.ch_mult) - 1)
        return (self.z_channels, res, res)

    @property
    def latent_dim(self) -> int:
        c, h, w = self.latent_shape
        return c * h * w


class VAEEncoder(nn.Module):

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.num_resolutions = len(cfg.ch_mult)
        self.num_res_blocks = cfg.num_res_blocks
        self.conv_in = nn.Conv2d(cfg.in_channels, cfg.ch, 3, 1, 1)
        in_ch_mult = (1,) + tuple(cfg.ch_mult)
        curr_res = cfg.resolution
        self.down = nn.ModuleList()
        for level in range(self.num_resolutions):
            block_in, block_out = cfg.ch * in_ch_mult[level], cfg.ch * cfg.ch_mult[level]
            down = nn.Module()
            down.block = nn.ModuleList()
            down.attn = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                down.block.append(ResnetBlock(block_in, block_out, cfg.dropout))
                block_in = block_out
                if curr_res in cfg.attn_resolutions:
                    down.attn.append(_make_attn(block_in, cfg.attn_type))
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, cfg.resamp_with_conv)
                curr_res //= 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, cfg.dropout)
        self.mid.attn_1 = _make_attn(block_in, cfg.attn_type)
        self.mid.block_2 = ResnetBlock(block_in, block_in, cfg.dropout)
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * cfg.z_channels if cfg.double_z else cfg.z_channels,
                                  3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for level in range(self.num_resolutions):
            down = self.down[level]
            for i in range(self.num_res_blocks):
                h = down.block[i](h)
                if len(down.attn) > 0:
                    h = down.attn[i](h)
            if level != self.num_resolutions - 1:
                h = down.downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.conv_out(_swish(self.norm_out(h)))


class VAEDecoder(nn.Module):

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.num_resolutions = len(cfg.ch_mult)
        self.num_res_blocks = cfg.num_res_blocks
        self.tanh_out = cfg.tanh_out
        self.z_shape = (1,) + cfg.latent_shape
        block_in = cfg.ch * cfg.ch_mult[-1]
        curr_res = cfg.latent_shape[-1]
        self.conv_in = nn.Conv2d(cfg.z_channels, block_in, 3, 1, 1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, cfg.dropout)
        self.mid.attn_1 = _make_attn(block_in, cfg.attn_type)
        self.mid.block_2 = ResnetBlock(block_in, block_in, cfg.dropout)
        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            block_out = cfg.ch * cfg.ch_mult[level]
            up = nn.Module()
            up.block = nn.ModuleList()
            up.attn = nn.ModuleList()
            for _ in range(cfg.num_res_blocks + 1):
                up.block.append(ResnetBlock(block_in, block_out, cfg.dropout))
                block_in = block_out
                if curr_res in cfg.attn_resolutions:
                    up.attn.append(_make_attn(block_in, cfg.attn_type))
            if level != 0:
                up.upsample = Upsample(block_in, cfg.resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, cfg.out_ch, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        for level in reversed(range(self.num_resolutions)):
            up = self.up[level]
            for i in range(self.num_res_blocks + 1):
                h = up.block[i](h)
                if len(up.attn) > 0:
                    h = up.attn[i](h)
            if level != 0:
                h = up.upsample(h)
        h = self.conv_out(_swish(self.norm_out(h)))
        return torch.tanh(h) if self.tanh_out else h


class DiagonalGaussian:
    """Posterior of the autoencoder: mean/logvar split along channels."""

    def __init__(self, moments: torch.Tensor) -> None:
        self.mean, logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(self.mean ** 2 + self.std ** 2 - 1.0 - self.logvar, dim=[1, 2, 3])


class AutoencoderKL(nn.Module):

    def __init__(self, cfg: Optional[VAEConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or VAEConfig()
        if not self.cfg.double_z:
            raise ValueError("AutoencoderKL needs double_z=True (mean and logvar).")
        self.encoder = VAEEncoder(self.cfg)
        self.decoder = VAEDecoder(self.cfg)
        self.quant_conv = nn.Conv2d(2 * self.cfg.z_channels, 2 * self.cfg.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(self.cfg.embed_dim, self.cfg.z_channels, 1)

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(self.quant_conv(self.encoder(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x: torch.Tensor, sample_posterior: bool = True
                ) -> Tuple[torch.Tensor, DiagonalGaussian]:
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior

    def to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """Images [N, 1, H, W] -> decoder-space latents [N, z_channels, h, w].

        This is the representation the transformer works in: the posterior
        mode passed through ``post_quant_conv``, i.e. exactly what
        ``decoder`` consumes.
        """
        return self.post_quant_conv(self.encode(images).mode())

    def from_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Decoder-space latents [N, z_channels, h, w] -> images [N, 1, H, W]."""
        return self.decoder(latents)
