"""Batch feature extraction over processed assets → tidy per-plane DataFrame,
optionally merged with visual labels. See metrics.py for the feature definitions.
"""

from __future__ import annotations

from typing import Iterable, Optional

from . import io, metrics
from .labels import LabelStore


def scan(assets: Iterable[io.AssetRef], planes: Optional[Iterable[int]] = None,
         with_images: bool = False):
    """One row per (asset, plane) with coefficient/grid features (+ pair features).

    ``with_images=True`` also computes the Tier-2 pixel-histogram-imbalance features
    (loads EMF stacks — slower).
    """
    import pandas as pd

    planes = list(range(io.N_PLANES)) if planes is None else list(planes)
    rows = []
    for asset in assets:
        for p in planes:
            try:
                f = metrics.plane_features(asset, p)
            except Exception as exc:  # missing plane / unreadable — skip, note
                rows.append({"asset_name": asset.name, "plane": p, "error": str(exc)})
                continue
            f.update(asset_name=asset.name, plane=p,
                     subject_id=asset.subject_id, pair_idx=p // 2)
            if with_images:
                try:
                    f.update(metrics.histogram_imbalance(asset, p))
                except Exception:
                    pass
            rows.append(f)
    df = pd.DataFrame(rows)
    if "alpha_mean" in df.columns:
        df = metrics.add_pair_features(df)
    return df


def scan_labeled(store: LabelStore, with_images: bool = False):
    """Scan only the assets/planes that have a visual label, and merge the label in.
    Returns a DataFrame with a `label` column (latest label per (asset, plane))."""
    import pandas as pd

    labels = store.to_df()
    if labels.empty:
        return pd.DataFrame()
    labels["plane"] = labels["plane"].astype(int)
    assets = [io.asset_from_name(n) for n in sorted(labels["asset_name"].unique())]
    df = scan(assets, with_images=with_images)
    merged = df.merge(labels[["asset_name", "plane", "label", "note"]],
                      on=["asset_name", "plane"], how="left")
    return merged
