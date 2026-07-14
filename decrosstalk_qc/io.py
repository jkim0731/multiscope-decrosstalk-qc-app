"""IO layer for decrosstalk QC.

Resolves multiplane-ophys processed assets from either a mounted ``/data`` asset
(attached) or from S3 via ``aind_session`` (not attached), and provides low-level
HDF5 / JSON readers used by :mod:`decrosstalk_qc.images` and
:mod:`decrosstalk_qc.arrays`.

Conventions (see code/docs/phase1_qc_plan.md):
- All *code* lives under ``code/...``; all *artifacts* under ``/scratch/...``.
- Ephemeral cache: ``/scratch/tmp/decrosstalk_qc_cache`` (safe to delete/rebuild).
- Durable outputs: ``/scratch/decrosstalk_qc``.
- Never ``rglob`` over ``/data`` (segmentation zarrs stall it); address files by
  exact path.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# --- paths -----------------------------------------------------------------

DATA_DIR = Path("/root/capsule/data")
SCRATCH_TMP = Path("/scratch/tmp")
CACHE_DIR = SCRATCH_TMP / "decrosstalk_qc_cache"
ARTIFACT_DIR = Path("/scratch/decrosstalk_qc")
# per-run scratch temp dir; created + torn down by use_scratch_tmp()
RUN_TMP_DIR = SCRATCH_TMP / "decrosstalk_qc_runtmp"

N_PLANES = 8
PLANE_PREFIX = "VISp"


def use_scratch_tmp() -> Path:
    """Redirect ALL process temp/cache to /scratch so nothing lands on the tiny 5 GB
    ``/`` overlay (``/tmp`` lives there). Idempotent; safe to call at import time,
    before Qt/matplotlib/fsspec are imported. Returns the run temp dir.

    A plain ``python -m decrosstalk_qc.app`` shell has empty TMPDIR/TEMP/TMP, so
    Python's ``tempfile`` and libraries default to ``/tmp`` (→ ``/`` overlay, ~5 GB).
    Point them all at ``/scratch`` (8 EB) instead, and register cleanup on exit.
    """
    RUN_TMP_DIR.mkdir(parents=True, exist_ok=True)
    for var in ("TMPDIR", "TEMP", "TMP"):
        os.environ[var] = str(RUN_TMP_DIR)
    os.environ.setdefault("XDG_CACHE_HOME", str(SCRATCH_TMP / "xdg_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(SCRATCH_TMP / "mplconfig"))
    os.environ.setdefault("FSSPEC_CACHE_DIR", str(SCRATCH_TMP / "fsspec"))
    tempfile.tempdir = str(RUN_TMP_DIR)  # force the tempfile module too
    # NB: XDG_RUNTIME_DIR is left as Qt's default (/tmp/runtime-root, a few bytes of
    # lock/socket files) — redirecting it to NFS /scratch risks Qt permission/socket
    # errors on a real display, and it is not a disk-fill source.
    atexit.register(clear_run_tmp)
    return RUN_TMP_DIR


def clear_run_tmp() -> None:
    """Remove the per-run scratch temp dir (registered atexit)."""
    shutil.rmtree(RUN_TMP_DIR, ignore_errors=True)


def clear_cache() -> None:
    """Empty the ephemeral image cache (``CACHE_DIR``). Durable outputs
    (figures, labels under ``ARTIFACT_DIR``) are left untouched."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)

# Plane pairing (confirmed from each plane's decrosstalk data_process.json
# `paired_emf`): planes are imaged as 4 simultaneous pairs. pair_of(p) == p ^ 1.
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def pair_of(plane: int) -> int:
    """The plane paired with ``plane`` (the crosstalk source for it)."""
    return plane ^ 1

# processed asset dir: multiplane-ophys_<subject>_<acq d/t>_processed_<proc d/t>
PROC_PATTERN = re.compile(
    r"^multiplane-ophys_(?P<subject_id>\d+)_"
    r"(?P<acq_date>\d{4}-\d{2}-\d{2})_(?P<acq_time>\d{2}-\d{2}-\d{2})_"
    r"processed_"
    r"(?P<proc_date>\d{4}-\d{2}-\d{2})_(?P<proc_time>\d{2}-\d{2}-\d{2})$"
)


# --- asset reference -------------------------------------------------------


@dataclass(frozen=True)
class AssetRef:
    """A processed multiplane-ophys asset, attached (``local_dir``) or remote (``id``)."""

    name: str
    id: Optional[str] = None
    local_dir: Optional[Path] = None

    @property
    def is_local(self) -> bool:
        return self.local_dir is not None

    @property
    def subject_id(self) -> Optional[str]:
        m = PROC_PATTERN.match(self.name)
        return m["subject_id"] if m else None

    @property
    def raw_name(self) -> str:
        """The raw session name (drop the ``_processed_...`` suffix)."""
        return self.name.split("_processed_")[0]


# module-level cache of resolved S3 source dirs, keyed by asset id
_source_dir_cache: dict[str, object] = {}


def find_processed_assets(subject_id: Optional[str] = None) -> list[AssetRef]:
    """List attached processed assets under ``/data`` (optionally one subject)."""
    if not DATA_DIR.is_dir():
        return []
    refs = []
    for name in sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()):
        m = PROC_PATTERN.match(name)
        if not m:
            continue
        if subject_id is not None and m["subject_id"] != str(subject_id):
            continue
        refs.append(AssetRef(name=name, local_dir=DATA_DIR / name))
    return refs


def asset_from_name(name: str) -> AssetRef:
    """Resolve an asset by name: prefer the attached ``/data`` copy, else S3."""
    local = DATA_DIR / name
    if local.is_dir():
        return AssetRef(name=name, local_dir=local)
    return AssetRef(name=name, id=_lookup_asset_id(name))


def _lookup_asset_id(name: str) -> str:
    """Find the Code Ocean data-asset id for a processed asset name via aind_session."""
    import aind_session  # lazy: heavy import

    raw = name.split("_processed_")[0]
    matches = [d for d in aind_session.Session(raw).data_assets if d.name == name]
    if not matches:
        raise FileNotFoundError(f"No Code Ocean data asset named {name!r}")
    return matches[0].id


def _source_dir(asset: AssetRef):
    """Return the asset root as a path object (local ``Path`` or S3 ``S3Path``)."""
    if asset.is_local:
        return asset.local_dir
    if asset.id is None:
        raise ValueError(f"AssetRef {asset.name!r} has neither local_dir nor id")
    if asset.id not in _source_dir_cache:
        import aind_session  # lazy

        _source_dir_cache[asset.id] = aind_session.get_data_asset_source_dir(asset.id)
    return _source_dir_cache[asset.id]


def plane_decrosstalk_dir(asset: AssetRef, plane: int):
    """Path to ``VISp_<plane>/decrosstalk`` (local ``Path`` or remote ``S3Path``)."""
    return _source_dir(asset) / f"{PLANE_PREFIX}_{plane}" / "decrosstalk"


def plane_file(asset: AssetRef, plane: int, suffix: str):
    """Path to ``VISp_<plane>/decrosstalk/VISp_<plane>_<suffix>``."""
    return plane_decrosstalk_dir(asset, plane) / f"{PLANE_PREFIX}_{plane}_{suffix}"


def available_planes(asset: AssetRef) -> list[int]:
    """Planes whose decrosstalk data_process.json is present."""
    planes = []
    for p in range(N_PLANES):
        jf = plane_file(asset, p, "decrosstalk_data_process.json")
        if _path_exists(jf):
            planes.append(p)
    return planes


# --- low-level readers -----------------------------------------------------


def _is_remote(path) -> bool:
    return str(path).startswith("s3://")


def _path_exists(path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def read_h5_dataset(path, dataset: str, index=None) -> np.ndarray:
    """Read one dataset (optionally sliced) from an HDF5 file, local or on S3.

    ``index`` is passed straight to h5py fancy indexing (e.g. ``slice(0, 10)``);
    ``None`` reads the whole dataset. Only the named dataset is touched, so this is
    safe on the multi-GB ``*_decrosstalk.h5`` movie (never reads ``data``).
    """
    import h5py

    if _is_remote(path):
        import fsspec

        with fsspec.open(str(path), "rb") as fobj:
            with h5py.File(fobj, "r") as f:
                dset = f[dataset]
                return dset[()] if index is None else dset[index]
    with h5py.File(str(path), "r") as f:
        dset = f[dataset]
        return dset[()] if index is None else dset[index]


def read_h5_shape(path, dataset: str) -> tuple:
    """Read just a dataset's shape (no data), local or on S3."""
    import h5py

    if _is_remote(path):
        import fsspec

        with fsspec.open(str(path), "rb") as fobj:
            with h5py.File(fobj, "r") as f:
                return tuple(f[dataset].shape)
    with h5py.File(str(path), "r") as f:
        return tuple(f[dataset].shape)


def read_json(path) -> dict:
    """Read a JSON file, local or on S3."""
    if _is_remote(path):
        import fsspec

        with fsspec.open(str(path), "rt") as fobj:
            return json.load(fobj)
    return json.loads(Path(str(path)).read_text())


def read_alpha_beta(asset: AssetRef, plane: int) -> tuple[Optional[float], Optional[float]]:
    """``(alpha_mean, beta_mean)`` from the plane's decrosstalk data_process.json."""
    jf = plane_file(asset, plane, "decrosstalk_data_process.json")
    params = read_json(jf).get("parameters", {})
    return params.get("alpha_mean"), params.get("beta_mean")


def cache_path(asset: AssetRef, name: str) -> Path:
    """A per-asset cache file path under ``CACHE_DIR`` (created on demand)."""
    key = asset.id or asset.name
    d = CACHE_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    return d / name
