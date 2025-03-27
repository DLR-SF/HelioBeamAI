# focalspot_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalSpotToDirectionsNet(nn.Module):
    def __init__(self, dropout_rate=0.01):
        super(FocalSpotToDirectionsNet, self).__init__()
        self.dropout_rate = dropout_rate
        n_ch = 128
        self.conv1 = nn.Conv2d(1 + 4, n_ch, kernel_size=5, padding=8, dilation=4)
        self.conv2 = nn.Conv2d(n_ch, n_ch, kernel_size=5, padding=8, dilation=4)
        self.conv3 = nn.Conv2d(n_ch, n_ch, kernel_size=7, padding=12, dilation=4)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.fusion = nn.Conv2d(n_ch, 3, kernel_size=1)

    def forward(self, focal_spot, heliostat_pos):
        condition = heliostat_pos.unsqueeze(-1).unsqueeze(-1)
        condition = condition.expand(-1, -1, focal_spot.size(2), focal_spot.size(3))
        x = torch.cat([focal_spot, condition], dim=1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        directions = torch.tanh(self.fusion(x))
        return directions


class RayTracingUtils:

    @staticmethod
    def get_origins(device, center=[0, 1, 54], side_lengths=(6, 6), bins=64):
        if isinstance(bins, int):
            height = width = bins
        else:
            height, width = bins
        x = torch.linspace((side_lengths[0] / width - side_lengths[0]) / 2,
                           (side_lengths[0] - side_lengths[0] / width) / 2, width, device=device)
        z = torch.linspace((side_lengths[1] / height - side_lengths[1]) / 2,
                           (side_lengths[1] - side_lengths[1] / height) / 2, height, device=device)
        x_grid, z_grid = torch.meshgrid(x, z, indexing='ij')
        x_grid = x_grid.flatten()
        z_grid = z_grid.flatten()
        y_grid = torch.zeros_like(x_grid, device=device)
        origins = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        origins += torch.tensor(center, dtype=torch.float32, device=device)
        return origins

    @staticmethod
    def ray_plane_intersection(directions, origins, energies, plane_center, plane_normal=[0, 1, 0], 
                                side_length1=6, side_length2=6, num_bins1=64, direction1=[-1, 0, 0]):
        device = directions.device
        plane_center = torch.tensor(plane_center, device=device, dtype=torch.float32)
        plane_normal = F.normalize(torch.tensor(plane_normal, device=device, dtype=torch.float32), dim=0)
        direction1 = torch.tensor(direction1, device=device, dtype=torch.float32)
        direction1 -= torch.dot(direction1, plane_normal) * plane_normal
        direction1 = F.normalize(direction1, dim=0)
        direction2 = F.normalize(torch.cross(plane_normal, direction1), dim=0)
        direction1 *= -1

        denom = torch.matmul(directions, plane_normal)
        valid_rays = torch.abs(denom) > 1e-6
        denom = denom[valid_rays]
        origins = origins[valid_rays]
        directions = directions[valid_rays]
        energies = energies[valid_rays]

        t = torch.matmul(plane_center - origins, plane_normal) / denom
        valid_t = t >= 0
        t = t[valid_t]
        origins = origins[valid_t]
        directions = directions[valid_t]
        energies = energies[valid_t]

        intersection_points = origins + t[:, None] * directions
        relative_intersections = intersection_points - plane_center
        proj_length1 = torch.matmul(relative_intersections, direction1)
        proj_length2 = torch.matmul(relative_intersections, direction2)

        proj_length1_norm = (proj_length1 + side_length1 / 2) / side_length1 * num_bins1 - 0.5
        proj_length2_norm = (proj_length2 + side_length2 / 2) / side_length2 * num_bins1 - 0.5

        bin1_low = torch.clamp(torch.floor(proj_length1_norm).long(), 0, num_bins1 - 1)
        bin1_high = torch.clamp(bin1_low + 1, 0, num_bins1 - 1)
        bin2_low = torch.clamp(torch.floor(proj_length2_norm).long(), 0, num_bins1 - 1)
        bin2_high = torch.clamp(bin2_low + 1, 0, num_bins1 - 1)

        w1_low = bin1_high.float() - proj_length1_norm
        w1_high = proj_length1_norm - bin1_low.float()
        w2_low = bin2_high.float() - proj_length2_norm
        w2_high = proj_length2_norm - bin2_low.float()

        bin_energies = torch.zeros((num_bins1, num_bins1), device=device, dtype=torch.float32)
        bin_energies.index_put_((bin1_low, bin2_low), energies * w1_low * w2_low, accumulate=True)
        bin_energies.index_put_((bin1_low, bin2_high), energies * w1_low * w2_high, accumulate=True)
        bin_energies.index_put_((bin1_high, bin2_low), energies * w1_high * w2_low, accumulate=True)
        bin_energies.index_put_((bin1_high, bin2_high), energies * w1_high * w2_high, accumulate=True)

        return bin_energies

    @staticmethod
    def calc_flux_deviation(f1, f2, down_sample=1, epsilon = 1e-8):
        if down_sample > 1:
            f1 = F.avg_pool2d(f1, kernel_size=down_sample, stride=down_sample)
            f2 = F.avg_pool2d(f2, kernel_size=down_sample, stride=down_sample)
        
        f1 = f1 / (torch.mean(f1) + epsilon)
        f2 = f2 / (torch.mean(f2) + epsilon)
        return torch.sum(torch.abs(f1 - f2)) / ((torch.mean(f1) + epsilon) * f1.numel())
