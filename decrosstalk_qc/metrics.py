"""QC features for decrosstalk quality, derived from the stored per-epoch grid
arrays (`alpha_list`/`beta_list`/`mean_norm_mi_list`) and, optionally, images.

Motivated by manual-labeling observations (see docs/labeling_observations.md):
- β is the paired-subtraction knob (`recon = [(1−β)·signal − β·paired]/(1−α−β)`);
  high β over-subtracts → over-correction. High α alone is benign for that plane.
- The pipeline applies the epoch-AVERAGED (ᾱ,β̄); epoch variance of the optimum →
  mixed over/under across epochs → "unsure".
- Within a pair, good = balanced low α≈β; bad = asymmetric, one direction high.
- Cross-plane reciprocity α_A≈β_B: data-integrity check (secondary).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

from . import arrays, images, io

_TOL = 1e-9


def plane_features(asset: io.AssetRef, plane: int) -> dict:
    """Per-plane features from the per-epoch coefficient/grid arrays (fast, no movie)."""
    g = arrays.load_grid_arrays(asset, plane)
    a, b, grid = g["alpha"], g["beta"], g["grid"]
    grid_max = g["grid_max"]
    n = len(a)
    # nan-aware: the sparse (37x37) grid has NaN at unevaluated points
    mi_red = np.array([1.0 - float(np.nanmin(grid[i])) for i in range(n)])
    return {
        "n_epochs": n,
        # magnitude
        "alpha_mean": float(a.mean()),
        "beta_mean": float(b.mean()),           # PRIMARY: over-correction knob
        "ab_max": float(max(a.mean(), b.mean())),
        # epoch instability (→ unsure/mixed)
        "alpha_std": float(a.std()),
        "beta_std": float(b.std()),
        "alpha_range": float(a.max() - a.min()),
        "beta_range": float(b.max() - b.min()),
        # within-plane directional asymmetry (leak-in β vs leak-out α)
        "within_asym": float(abs(a.mean() - b.mean())),
        # grid-search health
        "boundary_hit_frac": float(np.mean((a >= grid_max - _TOL) |
                                           (b >= grid_max - _TOL))),
        # MI reduction (diagnostic, circular — context only)
        "mi_reduction_mean": float(mi_red.mean()),
        "mi_reduction_min": float(mi_red.min()),
    }


_LQ_KEYS = ("lam_min", "lam_max", "a_star", "b_star", "se_a", "se_b", "sigma", "depth", "snr")


def _fit_basin_quadratic(av, bv, zv, depth):
    """Fit z ~ c0 + c1 a + c2 b + c3 a^2 + c4 b^2 + c5 ab to points (av, bv, zv) and return
    the landscape-quality metric dict. `depth` (global basin depth) is passed in so it's
    consistent across regions. np.nan dict if degenerate / too few points."""
    if len(zv) < 6:
        return {k: np.nan for k in _LQ_KEYS}
    X = np.column_stack([np.ones_like(av), av, bv, av ** 2, bv ** 2, av * bv])
    coef, *_ = np.linalg.lstsq(X, zv, rcond=None)
    resid = zv - X @ coef
    sigma = float(np.sqrt((resid ** 2).sum() / max(len(zv) - 6, 1)))
    _, c1, c2, c3, c4, c5 = coef
    evals = np.linalg.eigvalsh(np.array([[2 * c3, c5], [c5, 2 * c4]]))
    lam_min, lam_max = float(evals[0]), float(evals[1])

    def _vertex(c):
        cc1, cc2, cc3, cc4, cc5 = c
        Hm = np.array([[2 * cc3, cc5], [cc5, 2 * cc4]])
        try:
            # grad z = [c1,c2] + Hm @ v = 0  ->  v = -Hm^-1 [c1,c2]
            return -np.linalg.solve(Hm, np.array([cc1, cc2]))
        except np.linalg.LinAlgError:
            return np.array([np.nan, np.nan])

    v = _vertex(coef[1:])
    a_star, b_star = float(v[0]), float(v[1])
    # propagated SE of the vertex via delta method (numeric Jacobian wrt c1..c5)
    try:
        cov5 = (sigma ** 2 * np.linalg.inv(X.T @ X))[1:, 1:]
        eps, J, base = 1e-6, np.zeros((2, 5)), coef[1:].copy()
        for k in range(5):
            cp = base.copy(); cp[k] += eps
            J[:, k] = (_vertex(cp) - _vertex(base)) / eps
        cov_v = J @ cov5 @ J.T
        se_a, se_b = float(np.sqrt(max(cov_v[0, 0], 0))), float(np.sqrt(max(cov_v[1, 1], 0)))
    except np.linalg.LinAlgError:
        se_a = se_b = np.nan
    snr = depth / sigma if sigma > 0 else np.nan
    return dict(lam_min=lam_min, lam_max=lam_max, a_star=a_star, b_star=b_star,
                se_a=se_a, se_b=se_b, sigma=sigma, depth=depth, snr=snr)


def landscape_quality(grid_epoch, region="coarse", grid_interval=None,
                      coarse_step=0.04, fine_window=0.05):
    """Curvature / flatness / SNR of one epoch's MI objective basin, from a 2D quadratic
    fit to either the COARSE or the FINE grid points.

    The normalized-MI objective z(alpha, beta) (~1 at the origin, dipping to a minimum) is
    fit to  z ~ c0 + c1 a + c2 b + c3 a^2 + c4 b^2 + c5 a b. Region selection:
      region="coarse": the coarse lattice (every coarse_step/grid_interval-th point over the
                       whole 0..grid_max range) -> GLOBAL bowl shape.
      region="fine"  : all points within +/-fine_window of the grid argmin (the dense 0.01
                       block; for the sparse coarse-to-fine format this is the stored fine
                       block) -> LOCAL basin curvature near the minimum.

    Returns metric VALUES only (no pass/warn decision):
      lam_min, lam_max : Hessian [[2c3,c5],[c5,2c4]] eigenvalues (curvature)
      a_star, b_star   : fitted vertex  v = -H^-1 [c1,c2]
      se_a, se_b       : propagated vertex SE (delta method on sigma^2 (X'X)^-1)
      sigma            : RMS fit residual (landscape noise / non-quadraticity)
      depth            : 1 - nanmin(grid)  (global basin depth; same for both regions)
      snr              : depth / sigma
    np.nan where the fit is degenerate (Hessian singular / too few points).
    """
    gi = arrays.GRID_INTERVAL if grid_interval is None else grid_interval
    G = np.asarray(grid_epoch, dtype=float)
    n = G.shape[0]
    ax_full = np.arange(n) * gi
    depth = 1.0 - float(np.nanmin(G))
    if region == "coarse":
        step = max(int(round(coarse_step / gi)), 1)
        idx = np.arange(0, n, step)
        A, B = np.meshgrid(ax_full[idx], ax_full[idx], indexing="ij")
        Z = G[np.ix_(idx, idx)]
    elif region == "fine":
        i0, j0 = np.unravel_index(int(np.nanargmin(G)), G.shape)
        w = max(int(round(fine_window / gi)), 1)
        ii = np.arange(max(0, i0 - w), min(n, i0 + w + 1))
        jj = np.arange(max(0, j0 - w), min(n, j0 + w + 1))
        A, B = np.meshgrid(ax_full[ii], ax_full[jj], indexing="ij")
        Z = G[np.ix_(ii, jj)]
    else:
        raise ValueError(f"region must be 'coarse' or 'fine', got {region!r}")
    m = np.isfinite(Z)
    if int(m.sum()) < 6:
        return {k: np.nan for k in _LQ_KEYS}
    return _fit_basin_quadratic(A[m], B[m], Z[m], depth)


def plane_landscape_quality(asset: io.AssetRef, plane: int) -> dict:
    """Per-plane landscape-quality: epoch-mean of ``landscape_quality`` over all epochs,
    from the stored per-epoch MI grid, for BOTH the coarse and fine fits. Keys suffixed
    ``_coarse`` / ``_fine`` (e.g. ``se_b_coarse``, ``se_b_fine``). Format-aware (dense
    31x31 or sparse 37x37)."""
    g = arrays.load_grid_arrays(asset, plane)
    grid, gi = g["grid"], arrays.GRID_INTERVAL
    out = {}
    for region in ("coarse", "fine"):
        per = [landscape_quality(grid[e], region=region, grid_interval=gi)
               for e in range(grid.shape[0])]
        for k in _LQ_KEYS:
            vals = np.array([p[k] for p in per], dtype=float)
            out[f"{k}_{region}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    out["n_epochs"] = grid.shape[0]
    return out


def _cell_mask(img, dilate=2):
    """Boolean mask of cell footprints from basic_segmentation, with holes filled (so the
    dim nucleus inside a detected cell rim is included, not left as a low-value pixel) and
    a small dilation for a margin. Used to EXCLUDE cells before estimating background."""
    from .repro_estimator import basic_segmentation
    m = basic_segmentation(img) > 0
    m = ndimage.binary_fill_holes(m)
    if dilate:
        m = ndimage.binary_dilation(m, iterations=dilate)
    return m


def background_correlation(sig_mean, pai_mean, block=16, min_valid=0.3,
                           gauss_sigma=30, dilate=2):
    """Low-frequency background correlation between a plane and its paired plane (both
    full-session mean FOVs, same registration frame), with CELLS REMOVED.

    The decrosstalk MI model assumes the vasculature-shadow / illumination background is
    shared between the two planes; for far-apart (deep) pairs this can break. We estimate
    the background after masking out segmented cells (basic_segmentation, holes filled so
    dim nuclei are excluded too, dilated) in EITHER plane, then correlate:
      - ``bg_corr``       : per-block MEDIAN of the non-cell pixels (downsample) -> the
                            neuropil/vasculature background. Primary metric.
      - ``bg_corr_gauss`` : naive heavy-Gaussian low-pass, no cell removal (cells smear
                            in). Reference for how much cell-removal matters.
    Invalid pixels (<=0, warped border) are also masked; blocks with < ``min_valid`` valid
    (non-cell, in-FOV) fraction are dropped, so no motion crop is needed. Pearson r over the
    retained blocks; ~1 = shared background (assumption holds), lower = patterns differ.

    Returns dict(bg_corr, bg_corr_gauss, n_blocks).
    """
    s = np.asarray(sig_mean, dtype=float)
    p = np.asarray(pai_mean, dtype=float)
    H, W = s.shape
    h, w = (H // block) * block, (W // block) * block
    s, p = s[:h, :w], p[:h, :w]
    infov = np.isfinite(s) & np.isfinite(p) & (s > 0) & (p > 0)
    cells = _cell_mask(s, dilate) | _cell_mask(p, dilate)  # cell in either plane
    valid = infov & ~cells
    nb = (h // block, w // block)
    frac = valid.reshape(nb[0], block, nb[1], block).mean(axis=(1, 3))
    goodblk = frac >= min_valid

    def _coarse_median(img):
        a = np.where(valid, img, np.nan).reshape(nb[0], block, nb[1], block)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # all-nan blocks -> nan
            return np.nanmedian(a, axis=(1, 3))

    def _coarse_gauss(img):
        g = ndimage.gaussian_filter(np.where(infov, img, 0.0), gauss_sigma)
        return g.reshape(nb[0], block, nb[1], block).mean(axis=(1, 3))

    def _corr(x, y):
        m = goodblk & np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[m], y[m])[0, 1]) if int(m.sum()) >= 10 else np.nan

    return {
        "bg_corr": _corr(_coarse_median(s), _coarse_median(p)),
        "bg_corr_gauss": _corr(_coarse_gauss(s), _coarse_gauss(p)),
        "n_blocks": int(goodblk.sum()),
    }


def plane_background_correlation(asset: io.AssetRef, plane: int) -> dict:
    """Background correlation between a plane (PRE mean FOV) and its partner (registered to
    this plane's frame). See background_correlation. Loads full-session mean images."""
    sig = images.load_mean_img(asset, plane, "pre")
    pai = images.load_mean_img(asset, plane, "paired")
    return background_correlation(sig, pai)


def add_pair_features(df):
    """Add cross-plane (pair-level) columns to a per-plane feature DataFrame:
    reciprocity |α_this − β_partner|, and the pair's max β / max asymmetry."""
    import pandas as pd  # noqa: F401  (df is already a DataFrame)

    df = df.copy()
    for col in ("recip_gap", "pair_beta_max", "pair_asym_max", "partner_beta"):
        df[col] = np.nan
    by = df.set_index(["asset_name", "plane"])
    for (asset_name, plane), row in by.iterrows():
        partner = io.pair_of(int(plane))
        if (asset_name, partner) not in by.index:
            continue
        prow = by.loc[(asset_name, partner)]
        mask = (df["asset_name"] == asset_name) & (df["plane"] == plane)
        # reciprocity: this plane's α should match the partner's β
        df.loc[mask, "recip_gap"] = abs(row["alpha_mean"] - prow["beta_mean"])
        df.loc[mask, "partner_beta"] = prow["beta_mean"]
        df.loc[mask, "pair_beta_max"] = max(row["beta_mean"], prow["beta_mean"])
        df.loc[mask, "pair_asym_max"] = max(row["within_asym"], prow["within_asym"])
    return df


def histogram_imbalance(asset: io.AssetRef, plane: int) -> dict:
    """Tier-2 (loads EMF): pixel-intensity imbalance between the signal plane (PRE)
    and its paired plane, per epoch. Hypothesized driver of high/asymmetric β.

    Returns median-ratio and normalized histogram L1 distance, aggregated over epochs.
    """
    pre = images.load_emf(asset, plane, "pre")       # (n_epochs, H, W)
    paired = images.load_emf(asset, plane, "paired")
    n = pre.shape[0]
    med_ratio, hist_l1 = [], []
    lo = min(pre.min(), paired.min())
    hi = max(np.percentile(pre, 99.5), np.percentile(paired, 99.5))
    edges = np.linspace(lo, hi, 64)
    for i in range(n):
        ps, pp = pre[i], paired[i]
        ms, mp = np.median(ps), np.median(pp)
        med_ratio.append(float(ms / mp) if mp else np.nan)
        hs, _ = np.histogram(ps, bins=edges, density=True)
        hp, _ = np.histogram(pp, bins=edges, density=True)
        hs = hs / (hs.sum() or 1); hp = hp / (hp.sum() or 1)
        hist_l1.append(float(np.abs(hs - hp).sum()))
    med_ratio = np.array(med_ratio, float)
    return {
        "hist_med_ratio_mean": float(np.nanmean(med_ratio)),
        "hist_med_ratio_std": float(np.nanstd(med_ratio)),
        "hist_l1_mean": float(np.mean(hist_l1)),
        "hist_l1_std": float(np.std(hist_l1)),
    }
