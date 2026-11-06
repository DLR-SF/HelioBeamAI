"""HelioBeamAI: transformer-based heliostat focal spot prediction.

Inference entry points::

    from heliobeamai import HelioBeamField, GaussianSource      # everything in one object
    from heliobeamai import FluxPredictor, RayProjector, build_conditions
"""
from .checkpoints import load_flux_model, load_projector, load_vae, save_flux_model
from .field import HelioBeamField, PreparedSpots
from .receiver import CylindricalReceiver, OVRReceiver, PlanarReceiver, Receiver
from .sources import GaussianSource, SpotGeometry, SpotSource, TransformerSource
from .inference import (FluxPredictor, ProjectionConfig, RayProjector, build_conditions,
                        normalize_heliostat_id)
from .models import (AutoencoderKL, FluxModel, FluxModelConfig, FocalSpotToDirectionsNet,
                     VAEConfig)

__version__ = "1.0.0.dev0"

__all__ = ["HelioBeamField", "PreparedSpots", "GaussianSource", "TransformerSource", "SpotSource",
           "SpotGeometry", "Receiver", "PlanarReceiver", "OVRReceiver", "CylindricalReceiver", "FluxModel", "FluxModelConfig", "AutoencoderKL", "VAEConfig",
           "FocalSpotToDirectionsNet", "FluxPredictor", "RayProjector", "ProjectionConfig",
           "build_conditions", "normalize_heliostat_id", "load_flux_model", "load_vae",
           "load_projector", "save_flux_model"]
