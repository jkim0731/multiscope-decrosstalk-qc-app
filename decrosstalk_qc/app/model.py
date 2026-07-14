"""Qt-free viewer state + image provider for the decrosstalk QC app.

Navigation unit is a **plane-pair** (0-1, 2-3, 4-5, 6-7): the viewer shows both
planes of a pair at once (2 rows), each with three flip comparisons (3 cols).

Kept free of any Qt import so navigation/loading logic is unit-testable headlessly.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np

from .. import images, io

SOURCES = ("mean", "epoch")


def _lru_get(cache: "OrderedDict", key, fn, cap: int):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    val = fn()
    cache[key] = val
    while len(cache) > cap:
        cache.popitem(last=False)
    return val


class ImageProvider:
    """Lazy, memory-bounded access to images/α,β/NMI for a work-list.

    Memory hygiene (see plan §3): the 20 MB EMF stacks are NOT kept — mean images
    are computed from a transient stack and only the 2 MB mean is cached. EMF stacks
    are cached in a *tiny* separate LRU used only for epoch-scrub mode. This keeps the
    resident set to a few hundred MB instead of GBs (which previously spilled to swap
    on the 5 GB ``/`` overlay). ``mean_cap`` × 2 MB is the dominant footprint.
    """

    def __init__(self, mean_cap: int = 64, emf_cap: int = 6):
        self._means: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._emfs: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._scalars: dict = {}  # α/β, NMI, n_epochs — tiny, keep all
        self.mean_cap = mean_cap
        self.emf_cap = emf_cap

    def mean_img(self, asset: io.AssetRef, plane: int, kind: str) -> np.ndarray:
        # loads a transient EMF stack, averages, keeps only the 2 MB mean
        return _lru_get(self._means, (asset.name, plane, kind),
                        lambda: images.load_mean_img(asset, plane, kind), self.mean_cap)

    def emf(self, asset: io.AssetRef, plane: int, kind: str) -> np.ndarray:
        return _lru_get(self._emfs, (asset.name, plane, kind),
                        lambda: images.load_emf(asset, plane, kind), self.emf_cap)

    def alpha_beta(self, asset: io.AssetRef, plane: int):
        return self._scalars.setdefault(("ab", asset.name, plane),
                                        io.read_alpha_beta(asset, plane))

    def nmi_cached(self, asset: io.AssetRef, plane: int):
        """Cached (pre, post) NMI or None — never computes (nav-safe)."""
        return self._scalars.get(("nmi", asset.name, plane))

    def nmi_pre_post(self, asset: io.AssetRef, plane: int):
        """Compute (and cache) (pre, post) NMI — ~0.7 s/plane, call off the nav path."""
        key = ("nmi", asset.name, plane)
        if key not in self._scalars:
            from skimage.metrics import normalized_mutual_information as nmi

            pair = self.mean_img(asset, plane, "paired").ravel()
            pre = nmi(self.mean_img(asset, plane, "pre").ravel(), pair)
            post = nmi(self.mean_img(asset, plane, "post").ravel(), pair)
            self._scalars[key] = (float(pre), float(post))
        return self._scalars[key]

    def n_epochs(self, asset: io.AssetRef, plane: int) -> int:
        return self._scalars.setdefault(("ne", asset.name, plane),
                                        images.n_epochs(asset, plane))


class ViewerModel:
    """Navigation over (asset × plane-pair × epoch) + typed image access."""

    def __init__(self, assets: list[io.AssetRef], provider: Optional[ImageProvider] = None):
        if not assets:
            raise ValueError("ViewerModel needs at least one asset")
        self.assets = assets
        self.provider = provider or ImageProvider()
        self.asset_idx = 0
        self.pair_idx = 0
        self.epoch = 0
        self.source = "mean"  # "mean" | "epoch"

    # --- current selection ---
    @property
    def asset(self) -> io.AssetRef:
        return self.assets[self.asset_idx]

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def n_pairs(self) -> int:
        return len(io.PAIRS)

    @property
    def pair(self) -> tuple[int, int]:
        return io.PAIRS[self.pair_idx]

    @property
    def n_epochs(self) -> int:
        try:
            return self.provider.n_epochs(self.asset, self.pair[0])
        except Exception:
            return 1

    # --- image access ---
    def image(self, plane: int, role: str) -> np.ndarray:
        """2D image for a signal plane and role under the current source mode."""
        if self.source == "epoch":
            stack = self.provider.emf(self.asset, plane, role)
            e = min(self.epoch, stack.shape[0] - 1)
            return stack[e]
        return self.provider.mean_img(self.asset, plane, role)

    def alpha_beta(self, plane: int):
        return self.provider.alpha_beta(self.asset, plane)

    def nmi(self, plane: int):
        """Cached NMI or None — nav-safe (does not compute)."""
        return self.provider.nmi_cached(self.asset, plane)

    def compute_nmi(self, plane: int):
        """Force NMI compute (expensive) — call off the nav path."""
        try:
            return self.provider.nmi_pre_post(self.asset, plane)
        except Exception:
            return (float("nan"), float("nan"))

    def info(self) -> dict:
        planes = self.pair
        ab = {p: self.alpha_beta(p) for p in planes}
        return {
            "asset": self.asset.name,
            "asset_idx": self.asset_idx,
            "pair_idx": self.pair_idx,
            "planes": planes,
            "epoch": self.epoch if self.source == "epoch" else None,
            "source": self.source,
            "alpha_beta": ab,
            "nmi": {p: self.nmi(p) for p in planes},
        }

    # --- navigation (return True if state changed) ---
    def set_asset(self, idx: int) -> bool:
        idx = max(0, min(idx, self.n_assets - 1))
        if idx == self.asset_idx:
            return False
        self.asset_idx = idx
        self.epoch = 0
        return True

    def set_pair(self, idx: int) -> bool:
        idx = max(0, min(idx, self.n_pairs - 1))
        if idx == self.pair_idx:
            return False
        self.pair_idx = idx
        self.epoch = 0
        return True

    def set_epoch(self, epoch: int) -> bool:
        epoch = max(0, min(epoch, self.n_epochs - 1))
        if epoch == self.epoch:
            return False
        self.epoch = epoch
        return True

    def set_source(self, source: str) -> bool:
        if source not in SOURCES or source == self.source:
            return False
        self.source = source
        return True

    def toggle_source(self) -> bool:
        return self.set_source("epoch" if self.source == "mean" else "mean")

    def next_pair(self):
        return self.set_pair(self.pair_idx + 1)

    def prev_pair(self):
        return self.set_pair(self.pair_idx - 1)

    def next_asset(self):
        return self.set_asset(self.asset_idx + 1)

    def prev_asset(self):
        return self.set_asset(self.asset_idx - 1)

    def next_epoch(self):
        return self.set_epoch(self.epoch + 1)

    def prev_epoch(self):
        return self.set_epoch(self.epoch - 1)


def contrast_levels(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0):
    """(vmin, vmax) from percentiles; robust display contrast."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = float(np.percentile(finite, lo_pct))
    vmax = float(np.percentile(finite, hi_pct))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax
