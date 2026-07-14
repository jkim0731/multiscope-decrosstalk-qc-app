"""Decrosstalk QC (Phase 1).

Tooling to QC multiplane-ophys decrosstalk outputs: load PRE/POST/paired images and
per-epoch grid arrays from processed assets (attached ``/data`` or S3), a Qt visual-QC
app, and an image-based quality metric. See ``code/docs/phase1_qc_plan.md``.
"""

from __future__ import annotations

from . import arrays, images, io, labels, metrics, scan
from .io import (
    AssetRef,
    asset_from_name,
    available_planes,
    find_processed_assets,
    read_alpha_beta,
)
from .labels import LabelStore

# Redirect all process temp/cache to /scratch at import (before Qt/matplotlib/fsspec)
# so nothing lands on the tiny 5 GB `/` overlay (`/tmp`). See io.use_scratch_tmp.
io.use_scratch_tmp()

__all__ = [
    "io",
    "images",
    "arrays",
    "labels",
    "AssetRef",
    "asset_from_name",
    "find_processed_assets",
    "available_planes",
    "read_alpha_beta",
    "LabelStore",
]
