"""Per-heliostat focal spot prediction with cached encoder states."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch

from ..checkpoints import torch_load
from ..models.flux_model import FluxModel
from .conditions import normalize_heliostat_id, normalize_images_to_mean

PathLike = Union[str, Path]


class FluxPredictor:
    """Predict focal spots of known heliostats for new conditions.

    Every heliostat's calibration context is encoded once
    (:meth:`encode_context`, :meth:`encode_dataset` or :meth:`load_encoder_cache`);
    :meth:`predict` then only runs the decoder.

    Example::

        predictor = FluxPredictor.from_checkpoint("T2026.ckpt",
                                                  encoder_cache="cache_T2026.pt")
        cond, _, _ = build_conditions(sun, positions, aim_points)
        spots = predictor.predict(["AA23", "AB12"], cond)   # [2, 1, 64, 64]
    """

    def __init__(self, model: FluxModel, device: Union[str, torch.device] = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.contexts: Dict[str, torch.Tensor] = {}

    @classmethod
    def from_checkpoint(cls, checkpoint: PathLike, device: Union[str, torch.device] = "cpu",
                        encoder_cache: Optional[PathLike] = None,
                        dataset: Optional[PathLike] = None) -> "FluxPredictor":
        predictor = cls(FluxModel.from_checkpoint(checkpoint, map_location="cpu"), device)
        if encoder_cache is not None:
            predictor.load_encoder_cache(encoder_cache)
        elif dataset is not None:
            predictor.encode_dataset(dataset)
        return predictor

    # -- context -------------------------------------------------------------
    @property
    def heliostat_ids(self) -> List[str]:
        return list(self.contexts)

    def has_context(self, heliostat_id: object) -> bool:
        return normalize_heliostat_id(heliostat_id) in self.contexts

    @torch.no_grad()
    def encode_context(self, heliostat_id: object, images: torch.Tensor,
                       conditions: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode one heliostat's measurements (``[S, H, W]``, ``[S, 7]``).

        With ``normalize`` the images are scaled to the training image mean and
        the distance column is divided by the training normalization.
        """
        images = images.to(self.device, torch.float32)
        conditions = conditions.to(self.device, torch.float32).clone()
        if normalize:
            images = normalize_images_to_mean(images, self.model.cfg.image_mean)
            conditions[:, 6] /= self.model.cfg.distance_normalization_m
        memory = self.model.encode_context(images[None], conditions[None])[0]
        self.contexts[normalize_heliostat_id(heliostat_id)] = memory
        return memory

    def encode_dataset(self, dataset: Union[PathLike, dict], normalize: bool = True,
                       heliostat_ids: Optional[Iterable[object]] = None) -> None:
        """Encode all heliostats of a flux dataset (keys ``imgs``, ``data``, ``heliostat_ids``)."""
        if not isinstance(dataset, dict):
            dataset = torch_load(dataset)
        ids = dataset["heliostat_ids"]
        wanted = None if heliostat_ids is None else {normalize_heliostat_id(h) for h in heliostat_ids}
        for idx, raw_id in enumerate(ids):
            if wanted is not None and normalize_heliostat_id(raw_id) not in wanted:
                continue
            self.encode_context(raw_id, dataset["imgs"][idx], dataset["data"][idx], normalize)

    def save_encoder_cache(self, path: PathLike, **metadata) -> None:
        ids = self.heliostat_ids
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"heliostat_ids": ids,
                    "encoder_memory": {h: self.contexts[h].cpu() for h in ids},
                    "metadata": metadata}, path)

    def load_encoder_cache(self, path: PathLike) -> None:
        """Load encoder states.

        Accepts ``{"heliostat_ids": [...], "encoder_memory": [H, S, D] or {id: [S, D]}}``
        and the older ``{"contexts": {id: tensor or {"encoder_memory": tensor}}}``.
        """
        cache = torch_load(path, map_location=self.device)
        if "contexts" in cache:
            items = [(k, v.get("encoder_memory", v.get("memory")) if isinstance(v, dict) else v)
                     for k, v in cache["contexts"].items()]
        elif "heliostat_ids" in cache:
            memory = cache.get("encoder_memory", cache.get("memory"))
            if isinstance(memory, dict):
                items = list(memory.items())
            else:
                if len(cache["heliostat_ids"]) != memory.shape[0]:
                    raise ValueError("heliostat_ids and encoder_memory disagree in length.")
                items = list(zip(cache["heliostat_ids"], memory))
        else:
            raise ValueError(f"Unsupported encoder cache format in {path}.")
        for raw_id, memory in items:
            memory = torch.as_tensor(memory, dtype=torch.float32, device=self.device)
            if memory.ndim == 3 and memory.shape[0] == 1:
                memory = memory[0]
            if memory.ndim != 2:
                raise ValueError(f"Encoder memory for {raw_id} must be [S, D], got {tuple(memory.shape)}.")
            self.contexts[normalize_heliostat_id(raw_id)] = memory

    # -- prediction ----------------------------------------------------------
    def predict(self, heliostat_ids: Sequence[object], conditions: torch.Tensor,
                grad: bool = False) -> torch.Tensor:
        """Focal spots ``[N, T, H, W]`` for conditions ``[N, T, 7]`` (distance normalized).

        Use :func:`heliobeamai.build_conditions` to create ``conditions``.
        With ``grad`` the result stays differentiable w.r.t. ``conditions``.
        """
        if len(heliostat_ids) == 0:
            res = self.model.cfg.vae.resolution
            return torch.empty((0, conditions.shape[1] if conditions.ndim == 3 else 1, res, res),
                               device=self.device)
        keys = [normalize_heliostat_id(h) for h in heliostat_ids]
        missing = [k for k in keys if k not in self.contexts]
        if missing:
            raise KeyError(f"No encoded context for heliostats {missing[:10]}")
        memory = torch.stack([self.contexts[k] for k in keys])
        conditions = conditions.to(self.device, torch.float32)
        if grad:
            return self.model.decode(conditions, memory)
        with torch.no_grad():
            return self.model.decode(conditions, memory)
