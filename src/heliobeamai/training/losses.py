"""Loss functions."""
import torch


def relative_flux_deviation(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Sum of absolute deviations relative to the target energy, per image (``[..., H, W]`` -> ``[...]``).

    The error measure of the dissertation (``calcFluxDev``): 0 is perfect,
    0.1 means 10 % of the target energy is misplaced (counted on both sides).
    """
    return (target - pred).abs().sum(dim=(-1, -2)) / (target.mean(dim=(-1, -2)) * target.shape[-1] * target.shape[-2])
