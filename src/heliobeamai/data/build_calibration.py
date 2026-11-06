"""Build a calibration flux dataset for transformer training.

Inputs: the calibration database export (one row per measurement), the
heliostat positions and the centred flux images
(``GANCSTR/CalibrationDataFluxCentered``, frame C, 6 m x 6 m, see
``docs/data.md``). For every heliostat with enough measurements, each target
group contributes ``samples_per_target`` measurements chosen by max-min
sampling of the sun position (azimuth, elevation). Missing measurements of a
group are borrowed from the other groups, then duplicated.

Images are smoothed (Gaussian), scaled to mean 0.1 and passed through a VAE
(the dissertation and T2026 datasets used ``z64``). Output keys: ``imgs``,
``data`` (sun 3, aim direction 3, aim distance m), ``ids``, ``heliostat_ids``
and build metadata.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..inference.conditions import normalize_heliostat_id

# target group definitions; the block order of the dataset follows ``group_order``
TARGET_GROUPS_2026 = {"MFT": (3,), "Lower": (1, 7, 8, 9), "Upper": (4, 5, 6, 10, 11)}
TARGET_GROUPS_DISSERTATION = {"3": (3,), "456": (4, 5, 6), "17": (1, 7)}
HEIGHT_OFFSET_M = 87.0      # database U and position z are absolute; the model frame is tower-base relative


@dataclass
class CalibrationDatasetConfig:
    calibration_csv: str = "data/calibrationdata.csv"            # calibration database export
    heliostat_positions: str = "data/heliostat_positions.xlsx"
    flux_dir: str = "data/CalibrationDataFluxCentered"           # <record id>.png flux images
    vae_checkpoint: Optional[str] = "artifacts/vae/z64.pt"
    vae_sample_posterior: bool = True          # as the original builder (VAE forward samples)
    target_groups: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: dict(TARGET_GROUPS_2026))
    group_order: Tuple[str, ...] = ("MFT", "Lower", "Upper")
    samples_per_target: int = 15
    min_total_samples: int = 10
    image_size: int = 64
    smoothing_kernel: int = 5
    smoothing_sigma: float = 1.0
    # optional outlier filter of the dissertation datasets: per heliostat and
    # group keep measurements with score < median + c * std (c from 0.5 in
    # 0.25 steps until enough remain); scores: CSV with index id
    outlier_scores_csv: Optional[str] = None
    outlier_score_column: str = "Trans367"
    seed: Optional[int] = None
    limit_heliostats: Optional[int] = None
    device: str = "cpu"


def read_calibration_csv(path: str):
    import pandas as pd
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(path, sep=sep, decimal="," if sep == ";" else ".", encoding="utf-8-sig",
                     quotechar='"', engine="python" if sep == ";" else "c")
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    cols = ["SunPosE", "SunPosN", "SunPosU", "TargetOffsetE", "TargetOffsetN", "TargetOffsetU"]
    for c in ["id", "HeliostatId", "CalibrationTargetId", *cols]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["id", "HeliostatId", "CalibrationTargetId", *cols])
    df[["id", "HeliostatId", "CalibrationTargetId"]] = df[["id", "HeliostatId", "CalibrationTargetId"]].astype(int)
    return df.set_index("id").sort_index()


def maxmin_indices(points: np.ndarray, count: int, rng: np.random.Generator) -> List[int]:
    """Greedy max-min (farthest point) sampling."""
    if len(points) == 0 or count <= 0:
        return []
    selected = [int(rng.integers(len(points)))]
    dist = np.sum((points - points[selected[0]]) ** 2, axis=1)
    dist[selected[0]] = -np.inf
    while len(selected) < min(count, len(points)):
        nxt = int(np.argmax(dist))
        if not np.isfinite(dist[nxt]):
            break
        selected.append(nxt)
        dist = np.minimum(dist, np.sum((points - points[nxt]) ** 2, axis=1))
        dist[selected] = -np.inf
    return selected


def spread_by_sun(frame, count: int, rng: np.random.Generator):
    """``count`` measurements spread over sun azimuth/elevation."""
    if frame.empty or count <= 0:
        return frame.iloc[0:0]
    e, n, u = (frame[c].to_numpy(dtype=np.float64) for c in ("SunPosE", "SunPosN", "SunPosU"))
    az = np.degrees(np.arctan2(e, -n))
    el = np.degrees(np.arctan2(u, np.sqrt(e ** 2 + n ** 2)))
    return frame.iloc[maxmin_indices(np.column_stack((az, el)), count, rng)]


def filter_outliers(frame, scores, n: int):
    s = torch.tensor(scores.reindex(frame.index).to_numpy(dtype=np.float64))
    if len(s) == 0 or torch.isnan(s).all():
        return frame
    std, median = torch.nanmean((s - torch.nanmean(s)) ** 2).sqrt(), torch.nanmedian(s)
    c = 0.5
    while int((s < median + c * std).sum()) < min(n, int((~torch.isnan(s)).sum())):
        c += 0.25
    return frame[(s < median + c * std).numpy()]


def select_group(group, others, n: int, rng) -> Tuple[Optional[object], bool]:
    import pandas as pd
    primary = spread_by_sun(group, min(n, len(group)), rng)
    if len(primary) >= n:
        return primary, False
    parts = [primary] if len(primary) else []
    pool = others.loc[~others.index.isin(primary.index)]
    borrowed = spread_by_sun(pool, min(n - len(primary), len(pool)), rng)
    if len(borrowed):
        parts.append(borrowed)
    remaining = n - sum(len(p) for p in parts)
    if remaining > 0:
        dup = pd.concat(parts) if parts else pd.concat([group, others])
        dup = dup.loc[~dup.index.duplicated()]
        if dup.empty:
            return None, True
        parts.append(dup.iloc[rng.integers(len(dup), size=remaining)])
    return pd.concat(parts), True


def load_images(ids: Sequence[int], paths: Dict[int, Path], cfg: CalibrationDatasetConfig) -> torch.Tensor:
    from PIL import Image
    imgs = []
    for i in ids:
        with Image.open(paths[i]) as im:
            g = im.convert("L")
            if g.size != (cfg.image_size, cfg.image_size):
                g = g.resize((cfg.image_size, cfg.image_size), resample=Image.BILINEAR)
            imgs.append(torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0))
    x = torch.stack(imgs)
    k = cfg.smoothing_kernel
    if k > 1:
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-(ax[:, None] ** 2 + ax[None] ** 2) / (2 * cfg.smoothing_sigma ** 2))
        x = F.conv2d(x.unsqueeze(1), (g / g.sum())[None, None], padding=k // 2).squeeze(1)
    x = x.clamp_min(0)
    return 0.1 * x / x.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)


def build_calibration_dataset(cfg: CalibrationDatasetConfig) -> Dict:
    import pandas as pd
    from ..checkpoints import load_vae

    rng = np.random.default_rng(cfg.seed)
    df = read_calibration_csv(cfg.calibration_csv)
    positions = pd.read_excel(cfg.heliostat_positions).set_index("InternalName")
    paths = {int(p.stem): p for p in Path(cfg.flux_dir).glob("*.png") if p.stem.isdigit()}
    group_of = {t: g for g in cfg.group_order for t in cfg.target_groups[g]}
    df = df[df.index.isin(paths) & df.CalibrationTargetId.isin(group_of)].copy()
    df["group"] = df.CalibrationTargetId.map(group_of)
    scores = (pd.read_csv(cfg.outlier_scores_csv, index_col=0)[cfg.outlier_score_column]
              if cfg.outlier_scores_csv else None)
    counts = df.groupby("HeliostatId").size()
    heliostats = counts.index[counts >= cfg.min_total_samples].to_numpy()
    if cfg.limit_heliostats:
        heliostats = heliostats[:cfg.limit_heliostats]
    vae = load_vae(cfg.vae_checkpoint).to(cfg.device) if cfg.vae_checkpoint else None

    out = {"imgs": [], "data": [], "ids": [], "heliostat_ids": []}
    stats = Counter()
    with torch.inference_mode():
        for hid in heliostats:
            frame = df[df.HeliostatId == hid]
            chosen = []
            copied = False
            for g in cfg.group_order:
                group, others = frame[frame.group == g], frame[frame.group != g]
                if scores is not None:
                    group = filter_outliers(group, scores, cfg.samples_per_target)
                sel, used_copy = select_group(group, others, cfg.samples_per_target, rng)
                if sel is None or len(sel) < cfg.samples_per_target:
                    chosen = []
                    stats["selection_failed"] += 1
                    break
                chosen.append(sel.iloc[:cfg.samples_per_target])
                copied |= used_copy
            if not chosen:
                continue
            name = normalize_heliostat_id(int(hid))
            if name not in positions.index:
                stats["missing_position"] += 1
                continue
            sel = pd.concat(chosen)
            pos = positions.loc[name][["x", "y", "z"]].to_numpy(dtype=np.float32).copy()
            pos[2] -= HEIGHT_OFFSET_M
            aim = sel[["TargetOffsetE", "TargetOffsetN", "TargetOffsetU"]].to_numpy(dtype=np.float32).copy()
            aim[:, 2] -= HEIGHT_OFFSET_M
            vec = torch.from_numpy(aim - pos[None])
            dist = vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
            sun = torch.from_numpy(sel[["SunPosE", "SunPosN", "SunPosU"]].to_numpy(dtype=np.float32))
            imgs = load_images(sel.index.tolist(), paths, cfg)
            if vae is not None:
                x = imgs.unsqueeze(1).to(cfg.device)
                post = vae.encode(x)
                z = post.sample() if cfg.vae_sample_posterior else post.mode()
                imgs = vae.decode(z).squeeze(1).clamp_min(0).cpu()
            out["imgs"].append(imgs)
            out["data"].append(torch.cat([sun, vec / dist, dist], dim=1))
            out["ids"].append(torch.tensor(sel.index.tolist()))
            out["heliostat_ids"].append(int(hid))
            stats["copied_heliostats"] += int(copied)
    if not out["imgs"]:
        raise RuntimeError("no heliostat produced a sequence")
    return {"imgs": torch.stack(out["imgs"]), "data": torch.stack(out["data"]), "ids": torch.stack(out["ids"]),
            "heliostat_ids": torch.tensor(out["heliostat_ids"]),
            "target_ids": torch.tensor(sorted(group_of)), "target_groups": list(cfg.group_order),
            "samples_per_target": cfg.samples_per_target, "min_total_samples": cfg.min_total_samples,
            "copied_heliostats": stats["copied_heliostats"], "frame": "C",
            "build_config": asdict(cfg), "build_stats": dict(stats)}
