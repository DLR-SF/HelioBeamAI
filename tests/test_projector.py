"""Ray projection in frame C and the projection CNN training loss."""
import pytest
import torch
from torch import nn

from heliobeamai import ProjectionConfig, RayProjector
from heliobeamai.inference.projection import broaden, pixel_origins
from heliobeamai.training.projector import bin_on_planes


def test_pixel_layout_is_frame_c():
    o = pixel_origins(64, 64, (6.0, 6.0))
    px = 6.0 / 64
    torch.testing.assert_close(o[0], torch.tensor([3 - px / 2, 0.0, 3 - px / 2]))      # top east
    torch.testing.assert_close(o[63], torch.tensor([-3 + px / 2, 0.0, 3 - px / 2]))    # top west
    torch.testing.assert_close(o[-1], torch.tensor([-3 + px / 2, 0.0, -3 + px / 2]))   # bottom west


def test_binning_inverts_ray_layout():
    torch.manual_seed(0)
    spots = torch.rand(2, 16, 16)
    aim = torch.tensor([[0.0, 0.0, 50.0], [1.0, 0.0, 52.0]])
    cond = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.3]] * 2)
    rays = RayProjector()(spots, aim, cond).view(2, -1, 7)
    planes = (aim + torch.tensor([0.0, 3.0, 0.0]))[:, None]
    pixel_area = 6.0 * 6.0 / (16 * 16)               # ray energy = flux density x pixel area
    torch.testing.assert_close(bin_on_planes(rays[..., :3], rays[..., 3:6], rays[..., 6], planes, 16)[:, 0],
                               spots * pixel_area)


def test_broadening_keeps_energy_and_constant_deviations():
    torch.manual_seed(0)
    spots = torch.rand(3, 32, 32)
    dev = torch.ones(3, 3, 32, 32) * torch.tensor([0.2, -0.1, 0.4])[None, :, None, None]
    b, d = broaden(spots, dev, torch.tensor([100.0, 200.0, 300.0]), (6.0, 6.0), 1.8)
    torch.testing.assert_close(b.sum((-1, -2)), spots.sum((-1, -2)))
    torch.testing.assert_close(d, dev)


def test_ray_upsampling_keeps_deviations():
    class Const(nn.Module):
        conv1 = nn.Conv2d(8, 1, 1)

        def forward(self, x, c):
            return torch.full((x.shape[0], 3, *x.shape[-2:]), 0.5)

    cond = torch.tensor([[0.0, -0.5, 0.8, 0.6, -0.6, 0.5, 0.4]])
    p = RayProjector(ProjectionConfig(upsampling=4, upsampling_mode="rays"), Const())
    s, d = p.prepare(torch.rand(1, 16, 16), cond)
    assert s.shape == (1, 64, 64)
    torch.testing.assert_close(d, torch.full_like(d, 0.5))
