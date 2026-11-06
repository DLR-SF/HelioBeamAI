"""Datasets, splits and batching (Lightning parts need the ``train`` extra)."""
from .batching import split_batch, split_batch_artificial
from .splits import SplitSpec, get_split

__all__ = ["SplitSpec", "get_split", "split_batch", "split_batch_artificial"]
