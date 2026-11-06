"""CNN that turns a focal spot into per-pixel ray direction deviations."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FocalSpotToDirectionsNet(nn.Module):
    """Predicts a 3D direction deviation for every focal spot pixel.

    Input is the focal spot image plus a condition vector broadcast to every
    pixel: aim direction (3) and aim distance normalized by
    ``ProjectionConfig.distance_normalization_m``. The output is in [-1, 1]
    and is scaled by ``ProjectionConfig.deviation_scale`` before it is added
    to the aim direction.
    """

    def __init__(self, dropout_rate: float = 0.01, n_ch: int = 128,
                 cond_dim: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1 + cond_dim, n_ch, kernel_size=5, padding=8, dilation=4)
        self.conv2 = nn.Conv2d(n_ch, n_ch, kernel_size=5, padding=8, dilation=4)
        self.conv3 = nn.Conv2d(n_ch, n_ch, kernel_size=7, padding=12, dilation=4)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.fusion = nn.Conv2d(n_ch, 3, kernel_size=1)

    def forward(self, focal_spot: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """``focal_spot`` [N, 1, H, W], ``condition`` [N, C] -> [N, 3, H, W]."""
        cond = condition[:, :, None, None].expand(-1, -1, *focal_spot.shape[-2:])
        x = torch.cat([focal_spot, cond], dim=1)
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        return torch.tanh(self.fusion(x))
