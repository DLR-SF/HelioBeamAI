import os
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from HelioBeamAI.model.vae.autoencoder import AutoencoderKL
from HelioBeamAI.config import VAE_CONFIG, DEVICE

def calcFluxDev(f1, f2, normalize_by="f1"):
    """
    Calculates the flux deviation between two tensors.

    Args:
        f1 (torch.Tensor): First flux tensor (reference).
        f2 (torch.Tensor): Second flux tensor (comparison).
        normalize_by (str, optional): Specifies the normalization method.
                                       "f1" (default) - Normalizes by mean(f1)
                                       "f2" - Normalizes by mean(f2)

    Returns:
        torch.Tensor: Flux deviation.
    """
    if f1.shape != f2.shape:
        raise ValueError(f"❌ ERROR: Shape mismatch: f1 {f1.shape} vs f2 {f2.shape}")

    abs_diff = torch.abs(f1 - f2)  # Element-wise absolute difference
    sum_abs_diff = torch.sum(abs_diff, dim=(-1, -2))  # Sum over last two dimensions (H, W)

    # Choose normalization factor
    norm_factor = torch.mean(f1 if normalize_by == "f1" else f2, dim=(-1, -2))
    norm_factor = torch.clamp(norm_factor, min=1e-8)  # Avoid division by zero

    # Compute flux deviation
    dev = sum_abs_diff / (norm_factor * f1.size(-1) * f1.size(-2))

    return dev


def loadVAE(file=None, model_dir="./models/"):
    """
    Load a VAE model from a checkpoint file or initialize a new one.

    Args:
        file (str, optional): Name of the checkpoint file (without .ckpt extension).
        model_dir (str, optional): Directory where model checkpoints are stored.

    Returns:
        AutoencoderKL: The loaded or newly initialized model.
    """
    
    if file is None:
        print("🔄 No checkpoint provided. Initializing a new VAE model...")

        return AutoencoderKL(
            ddconfig=VAE_CONFIG["ddconfig"],
            lossconfig=VAE_CONFIG["lossconfig"],
            embed_dim=VAE_CONFIG["embed_dim"],
        )

    # Load model from checkpoint
    # Get the absolute path to the "models" directory
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))

    # Define the checkpoint file path
    checkpoint_path = os.path.join(BASE_DIR, f"{file}.ckpt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ ERROR: Checkpoint file '{checkpoint_path}' not found!")

    print(f"🔄 Loading checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    except Exception as e:
        raise RuntimeError(f"❌ ERROR: Failed to load checkpoint '{file}.ckpt': {e}")

    # Extract hyperparameters and initialize model
    hyper_parameters = checkpoint.get("hyper_parameters")
    if hyper_parameters is None:
        raise KeyError("❌ ERROR: Missing 'hyper_parameters' in checkpoint file.")

    print("✅ Hyperparameters loaded successfully.")

    model = AutoencoderKL(**hyper_parameters)

    # Load model weights
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("❌ ERROR: Missing 'state_dict' in checkpoint file.")

    model_weights = {key.replace("auto_encoder.", ""): val for key, val in state_dict.items()}

    try:
        model.load_state_dict(model_weights)
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"❌ ERROR: Failed to load model weights: {e}")

    return model


def generate_image_grid(b1: torch.Tensor, b2: torch.Tensor, num_images: int = 10, img_size: int = 256, grid_padding: int = 2):
    """
    Generates an image grid for logging, visualizing pairs of images.

    Args:
        b1 (torch.Tensor): First batch of images with shape (batch, channels, height, width).
        b2 (torch.Tensor): Second batch of images with shape (batch, channels, height, width).
        num_images (int): Number of images to include in the grid (default: 16).
        img_size (int): Target image size for interpolation (default: 256).
        grid_padding (int): Padding between images in the grid (default: 2).

    Returns:
        numpy.ndarray: Image grid in NumPy format for visualization or logging.
    """
    assert b1.dim() == 4 and b2.dim() == 4, "❌ ERROR: Both inputs must be 4D tensors (batch, channels, height, width)"
    assert b1.shape[0] == b2.shape[0], "❌ ERROR: Batch sizes of b1 and b2 must match"

    # Select `num_images` or the maximum available
    num_images = min(num_images, b1.size(0))

    # Extract first channel and concatenate images
    data = torch.cat([b1[:num_images, 0].unsqueeze(1), b2[:num_images, 0].unsqueeze(1)], dim=0)

    # Resize images using bilinear interpolation
    data = F.interpolate(data, size=(img_size, img_size), mode='bilinear', align_corners=False)

    # Create a grid of images
    img_grid = make_grid(data, nrow=num_images, padding=grid_padding, normalize=True)

    return img_grid.cpu().detach().numpy()
