"""Per-epoch decrosstalk grid arrays (alpha/beta and the MI objective grid).

Reads the small ``alpha_list`` / ``beta_list`` / ``mean_norm_mi_list`` datasets from
the plane's ``*_decrosstalk.h5`` (never the multi-GB ``data`` movie) and reshapes the
flattened objective into an (n_epochs, N, N) grid.

Two on-disk formats are supported (auto-detected from the flattened length):
  - **dense** (original CellPose pipeline): 31x31 grid over 0.00..0.30 at 0.01, no NaN.
  - **sparse** (speed-up branch, coarse-to-fine): 37x37 grid over 0.00..0.36 at 0.01 with
    fine (0.01) resolution near the minimum, coarse (0.04) elsewhere, and NaN at the
    unevaluated points.
Both are square grids at ``GRID_INTERVAL`` starting from 0, so the axis is
``np.arange(N) * GRID_INTERVAL`` and the grid maximum is ``(N - 1) * GRID_INTERVAL``.

Grid convention (from decrosstalk_roi_image.py): the search is
``for alpha in GRID: for beta in GRID: ...`` so the flat index is
``alpha_idx * N + beta_idx``; reshape(N, N) gives ``grid[alpha_idx, beta_idx]``.
``mean_norm_mi`` is normalized so ``grid[0, 0] == 1.0`` (raw, un-unmixed pair), and the
chosen (alpha, beta) per epoch is the (nan-aware) argmin. Use nan-aware ops downstream
(``np.nanargmin`` / ``np.nanmin``) so both formats work identically.
"""

from __future__ import annotations

import numpy as np

from . import io

GRID_INTERVAL = 0.01
# Legacy dense-format defaults (kept for reference/back-compat). Prefer the per-asset
# ``grid_n`` / ``grid_max`` / ``axis`` returned by load_grid_arrays, which handle both
# the 31x31 (0-0.30) and 37x37 (0-0.36) formats.
GRID_MAX = 0.30
GRID = np.arange(0, GRID_MAX + GRID_INTERVAL, GRID_INTERVAL)  # 31 values, 0.00..0.30
GRID_N = len(GRID)  # 31


def grid_axis(n: int) -> np.ndarray:
    """The (alpha or beta) coordinate axis for an n-per-side grid: 0, 0.01, ..., (n-1)*0.01."""
    return np.arange(n) * GRID_INTERVAL


def _augment(d: dict) -> dict:
    """Add grid geometry (grid_n / grid_max / axis) derived from the grid's side length,
    so callers never assume a fixed 31x31."""
    n = int(d["grid"].shape[-1])
    d["grid_n"] = n
    d["grid_max"] = float((n - 1) * GRID_INTERVAL)
    d["axis"] = grid_axis(n)
    return d


def _read_grid_from_h5(asset: io.AssetRef, plane: int) -> dict:
    h5 = io.plane_file(asset, plane, "decrosstalk.h5")
    alpha = np.asarray(io.read_h5_dataset(h5, "alpha_list"), dtype=float)
    beta = np.asarray(io.read_h5_dataset(h5, "beta_list"), dtype=float)
    mm = np.asarray(io.read_h5_dataset(h5, "mean_norm_mi_list"), dtype=float)
    per_epoch = mm.shape[-1]
    n = int(round(per_epoch ** 0.5))
    if n * n != per_epoch:
        raise ValueError(
            f"mean_norm_mi_list per-epoch length {per_epoch} is not a square grid "
            f"({asset.name} plane {plane})"
        )
    grid = mm.reshape(mm.shape[0], n, n)
    return _augment({"alpha": alpha, "beta": beta, "grid": grid})


def _build_asset_cache(asset: io.AssetRef, cache) -> None:
    """Read all planes' grid arrays in one pass and write a single per-asset npz
    (one NFS write per asset, not per plane — small-file creates on /scratch are slow)."""
    data = {}
    for p in range(io.N_PLANES):
        try:
            d = _read_grid_from_h5(asset, p)
        except Exception:
            continue
        data[f"alpha_{p}"] = d["alpha"]
        data[f"beta_{p}"] = d["beta"]
        data[f"grid_{p}"] = d["grid"]
    if data:
        np.savez(cache, **data)


def load_grid_arrays(asset: io.AssetRef, plane: int, use_cache: bool = True) -> dict:
    """Return per-epoch grid arrays for one plane.

    Keys:
      - ``alpha`` (n_epochs,)      chosen alpha per epoch (pre-rounding)
      - ``beta``  (n_epochs,)      chosen beta per epoch
      - ``grid``  (n_epochs, N, N) normalized-MI objective, [alpha_idx, beta_idx]
                                   (NaN at unevaluated points for the sparse format)
      - ``grid_n``   int           grid side length (31 dense / 37 sparse)
      - ``grid_max`` float         max alpha/beta on the grid ((N-1)*0.01)
      - ``axis`` (N,)              alpha/beta coordinate axis

    These datasets live inside the multi-GB ``*_decrosstalk.h5`` movie; opening it
    repeatedly (batch scans) is slow, so all 8 planes are extracted once and cached
    to a single per-asset ``grids.npz`` under ``io.CACHE_DIR``.
    """
    if use_cache:
        cache = io.cache_path(asset, "grids.npz")
        if not cache.exists():
            _build_asset_cache(asset, cache)
        if cache.exists():
            z = np.load(cache)
            if f"alpha_{plane}" in z.files:
                return _augment({"alpha": z[f"alpha_{plane}"], "beta": z[f"beta_{plane}"],
                                 "grid": z[f"grid_{plane}"]})
    return _read_grid_from_h5(asset, plane)


def chosen_from_grid(grid_epoch: np.ndarray) -> tuple[float, float]:
    """(alpha, beta) at the (nan-aware) argmin of one epoch's (N, N) objective grid."""
    ai, bi = np.unravel_index(int(np.nanargmin(grid_epoch)), grid_epoch.shape)
    axis = grid_axis(grid_epoch.shape[0])
    return float(axis[ai]), float(axis[bi])


def mi_reduction(grid_epoch: np.ndarray) -> float:
    """1 - min(objective): fraction of ROI mutual information removed at the argmin.

    Diagnostic only (the estimator optimizes this quantity — see plan §2). ~0 means
    nothing was removed (no crosstalk / not applied); larger means more removed.
    Nan-aware so it works on the sparse grid.
    """
    return 1.0 - float(np.nanmin(grid_epoch))


def boundary_hit(alpha: float, beta: float, grid_max: float = GRID_MAX,
                 tol: float = 1e-9) -> bool:
    """True if the chosen alpha or beta sits at the grid maximum (search saturated).

    Pass the per-asset ``grid_max`` (from load_grid_arrays) — 0.30 for the dense format,
    0.36 for the sparse format; the default is the legacy dense max.
    """
    return (abs(alpha - grid_max) <= tol) or (abs(beta - grid_max) <= tol)
