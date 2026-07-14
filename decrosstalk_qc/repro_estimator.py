"""Faithful reproduction of the decrosstalk alpha/beta estimator, with the ONLY change
being the segmenter: CellPose -> `get_and_plot_basic_segmentation` (movie-qc).

Everything downstream is copied verbatim from `aind-ophys-decrosstalk-roi-images`
(`decrosstalk_roi_image.py`): dendrite/border filtering, cross-plane IoU>0.7 dedup
(keep brighter), top-15-intensity, bounding-box (area x2), and the 31x31 normalized-MI
grid search minimized across ROI boxes, per epoch, averaged. Purpose: test whether a
fast classical segmenter yields the same (alpha, beta) as the CellPose pipeline — if
so, the QC/estimation can skip CellPose (~22 s/plane + torch) entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import filters, measure
from skimage.metrics import normalized_mutual_information as _nmi

from . import io, images

GRID_MAX = 0.36  # matches pipeline speed-up branch (was 0.30; raised to avoid clipping heavy planes)
GRID_INTERVAL = 0.01
PIX_SIZE = 0.78
MOTION_BUFFER = 5
DENDRITE_DIAM_UM = 10
NUM_TOP_ROIS = 15
OVERLAP_THRESHOLD = 0.7
AREA_EXT = 2


# --- basic segmentation (from movie-qc get_and_plot_basic_segmentation), adapted to a
#     single mean image and returning a CellPose-style integer-labeled mask ---
def basic_segmentation(mean_img, min_object_size=100, max_object_size=300, sigma=30):
    neuropil = ndimage.gaussian_filter(mean_img, sigma=sigma)
    hp = mean_img - neuropil
    thr = filters.threshold_otsu(hp)
    binary = hp > thr
    lab = measure.label(binary)
    out = np.zeros_like(lab)
    n = 0
    for region in measure.regionprops(lab):
        if min_object_size < region.area < max_object_size:
            n += 1
            out[lab == region.label] = n
    return out


# --- verbatim helpers from decrosstalk_roi_image.py ---
def filter_dendrite(masks, dendrite_diameter_pix):
    r = dendrite_diameter_pix / 2
    area_threshold = np.pi * r ** 2
    out = masks.copy()
    for roi_id in range(1, int(masks.max()) + 1):
        m = masks == roi_id
        if m.sum() < area_threshold:
            out[m] = 0
    return out


def filter_border_roi(masks, buffer_pix=5):
    border = np.zeros(masks.shape, dtype=bool)
    border[:buffer_pix, :] = border[-buffer_pix:, :] = 1
    border[:, :buffer_pix] = border[:, -buffer_pix:] = 1
    out = masks.copy()
    for roi_id in range(1, int(masks.max()) + 1):
        m = masks == roi_id
        if np.any(m & border):
            out[m] = 0
    return out


def reorder_mask(mask):
    out = np.zeros_like(mask)
    for i, ind in enumerate(np.setdiff1d(np.unique(mask), 0)):
        out[mask == ind] = i + 1
    return out


def get_ranks_roi_inds(img, mask):
    inds = np.setdiff1d(np.unique(mask), 0)
    mi = np.array([img[mask == i].mean() for i in inds])
    ranks = np.zeros_like(mi)
    ranks[np.argsort(mi)[::-1]] = np.arange(len(mi))
    return ranks, inds


def get_top_intensity_mask(img, mask, num_top_rois=15):
    inds = np.setdiff1d(np.unique(mask), 0)
    if len(inds) < num_top_rois:
        return mask
    mi = [img[mask == i].mean() for i in inds]
    top_ids = inds[np.argsort(mi)[::-1][:num_top_rois]]
    out = mask.copy()
    for i in inds:
        if i not in top_ids:
            out[mask == i] = 0
    return out


def get_bounding_box(masks, area_extension_factor=2):
    ext = np.sqrt(area_extension_factor * np.pi / 4)
    inds = np.setdiff1d(np.unique(masks), 0)
    bb = np.zeros((len(inds), *masks.shape), dtype=np.uint16)
    for i, mi in enumerate(inds):
        y, x = np.where(masks == mi)
        yl, xl = y.max() - y.min(), x.max() - x.min()
        by = [max(0, int(round(y.min() - ext * yl / 2))),
              min(masks.shape[0], int(round(y.max() + ext * yl / 2)))]
        bx = [max(0, int(round(x.min() - ext * xl / 2))),
              min(masks.shape[1], int(round(x.max() + ext * xl / 2)))]
        bb[i, by[0]:by[1], bx[0]:bx[1]] = mi
    return bb


def top_masks_basic(signal_mean, paired_mean):
    """get_signal_paired_top_masks with basic_segmentation instead of CellPose."""
    dpx = DENDRITE_DIAM_UM / PIX_SIZE
    sm = reorder_mask(filter_border_roi(filter_dendrite(basic_segmentation(signal_mean), dpx), MOTION_BUFFER))
    pm = reorder_mask(filter_border_roi(filter_dendrite(basic_segmentation(paired_mean), dpx), MOTION_BUFFER))
    ns, npd = int(sm.max()), int(pm.max())
    if ns and npd:
        ov = np.zeros((ns, npd))
        for i in range(1, ns + 1):
            for j in range(1, npd + 1):
                inter = ((sm == i) & (pm == j)).sum()
                uni = ((sm == i) | (pm == j)).sum()
                ov[i - 1, j - 1] = inter / uni if uni else 0
        sr, _ = get_ranks_roi_inds(signal_mean, sm)
        pr, _ = get_ranks_roi_inds(paired_mean, pm)
        for si, pi in zip(*np.where(ov > OVERLAP_THRESHOLD)):
            if sr[si] < pr[pi]:
                pm[pm == pi + 1] = 0
            else:
                sm[sm == si + 1] = 0
    return (get_top_intensity_mask(signal_mean, sm, NUM_TOP_ROIS),
            get_top_intensity_mask(paired_mean, pm, NUM_TOP_ROIS))


# --- crop range (verbatim logic) ---
def _plane_range(asset, plane):
    p = io.plane_decrosstalk_dir(asset, plane).parent / "motion_correction" / f"VISp_{plane}_motion_transform.csv"
    df = pd.read_csv(str(p))
    max_y = int(np.ceil(max(df.y.max(), 1))); min_y = int(np.floor(min(df.y.min(), 0)))
    max_x = int(np.ceil(max(df.x.max(), 1))); min_x = int(np.floor(min(df.x.min(), 0)))
    return [-min_y, -max_y], [-min_x, -max_x]


def _crop_range(asset, plane):
    ry_s, rx_s = _plane_range(asset, plane)
    ry_p, rx_p = _plane_range(asset, io.pair_of(plane))
    ry = [max(ry_s[0], ry_p[0]), min(ry_s[1], ry_p[1])]
    rx = [max(rx_s[0], rx_p[0]), min(rx_s[1], rx_p[1])]
    return ry, rx


def _grid_search(signal_mean, paired_mean, bb_masks):
    """Full 37x37 grid (0-0.36 at 0.01, verbatim objective). Slow (~80 s/epoch); reference."""
    grid = np.arange(0, GRID_MAX + GRID_INTERVAL, GRID_INTERVAL)
    return _search_over(signal_mean, paired_mean, bb_masks, grid, grid)[:2]


def _objective(signal_mean, paired_mean, box_idx, mi_raw, data, shape, a, b):
    unmix = np.linalg.inv([[1 - a, b], [a, 1 - b]])
    rec = unmix @ data
    rs = rec[0].reshape(shape); rp = rec[1].reshape(shape)
    mi = np.array([_nmi(rs[iy, ix], rp[iy, ix]) for (iy, ix) in box_idx])
    return float((mi / mi_raw).mean())


def _search_over(signal_mean, paired_mean, bb_masks, alphas, betas):
    box_idx = [np.where(m) for m in bb_masks]
    mi_raw = np.array([_nmi(signal_mean[iy, ix], paired_mean[iy, ix]) for (iy, ix) in box_idx])
    data = np.vstack((signal_mean.ravel(), paired_mean.ravel()))
    shape = signal_mean.shape
    best = None
    for a in alphas:
        for b in betas:
            v = _objective(signal_mean, paired_mean, box_idx, mi_raw, data, shape, a, b)
            if best is None or v < best[0]:
                best = (v, float(a), float(b))
    return best[1], best[2], best[0]


def _grid_search_fast(signal_mean, paired_mean, bb_masks, coarse=0.04, win=0.05):
    """Coarse-to-fine: coarse grid (step `coarse`) then fine (0.01) in +/-`win` around
    the coarse argmin. Matches the pipeline speed-up branch (coarse 0.04 over 0-0.36,
    fine 0.01 +/-0.05). Verified to match _grid_search (session02 verify gate)."""
    cg = np.round(np.arange(0, GRID_MAX + coarse, coarse), 4)
    cg = cg[cg <= GRID_MAX + 1e-9]
    a0, b0, _ = _search_over(signal_mean, paired_mean, bb_masks, cg, cg)
    fa = np.round(np.arange(max(0, a0 - win), min(GRID_MAX, a0 + win) + GRID_INTERVAL, GRID_INTERVAL), 4)
    fb = np.round(np.arange(max(0, b0 - win), min(GRID_MAX, b0 + win) + GRID_INTERVAL, GRID_INTERVAL), 4)
    fa = fa[fa <= GRID_MAX + 1e-9]; fb = fb[fb <= GRID_MAX + 1e-9]
    a, b, _ = _search_over(signal_mean, paired_mean, bb_masks, fa, fb)
    return a, b


def estimate_ab(asset: io.AssetRef, plane: int, epochs=None, fast=True, per_epoch=False):
    """Reproduce (alpha, beta) with basic segmentation. Returns (alpha_mean, beta_mean)
    rounded to 2 decimals (like the pipeline), plus n epochs used. With per_epoch=True,
    also returns the per-epoch (alpha, beta) lists (unrounded).
    fast=True uses the coarse-to-fine grid (verified to match the full grid)."""
    ry, rx = _crop_range(asset, plane)
    sig = images.load_emf(asset, plane, "pre")
    pai = images.load_emf(asset, plane, "paired")
    E = sig.shape[0]
    eps = range(E) if epochs is None else epochs
    A, B = [], []
    for e in eps:
        s = sig[e][ry[0] + MOTION_BUFFER:ry[1] - MOTION_BUFFER, rx[0] + MOTION_BUFFER:rx[1] - MOTION_BUFFER]
        p = pai[e][ry[0] + MOTION_BUFFER:ry[1] - MOTION_BUFFER, rx[0] + MOTION_BUFFER:rx[1] - MOTION_BUFFER]
        stm, ptm = top_masks_basic(s, p)
        bb = np.concatenate([get_bounding_box(stm, AREA_EXT), get_bounding_box(ptm, AREA_EXT)])
        if len(bb) == 0:
            continue
        a, b = (_grid_search_fast if fast else _grid_search)(s, p, bb)
        A.append(a); B.append(b)
    res = [round(float(np.mean(A)), 2), round(float(np.mean(B)), 2), len(A)]
    if per_epoch:
        res += [A, B]
    return tuple(res)
