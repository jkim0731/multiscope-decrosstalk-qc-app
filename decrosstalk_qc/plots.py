"""One-page landscape QC figure: per-epoch MI landscapes + coefficient stability + pair
symmetry (reciprocity).

Design rationale (session02 `bad_plane_analysis.ipynb`): landscape-shape metrics
(curvature / flatness / SNR) do NOT discriminate good from bad and were dropped. What
remains is a lean sanity view to catch DRASTIC estimation failures:
  - per-epoch landscapes (spot a wandering / multi-modal minimum),
  - α/β stability (argmin scatter on the α-β plane + α/β vs epoch),
  - pair symmetry / reciprocity (α_p≈β_q, β_p≈α_q) — catches a one-plane segmentation /
    registration breakdown that per-plane views cannot.

`render_landscape_page` is the shared renderer (mirrored in the pipeline's
decrosstalk_roi_image.py); `plot_landscape_page` is the asset-level wrapper.
"""
from __future__ import annotations

import math

import numpy as np

from . import arrays, io

COEF_MAX = 0.36     # fixed coefficient-value axis (grid max on the speed-up branch) -> figures comparable across sessions
RECIP_FLAG = 0.05   # reciprocity-gap flag line (above the observed max ~0.043; tunable)


def render_landscape_page(grid, alpha, beta, grid_interval=0.01, title="", applied=None,
                          partner=None, coef_max=COEF_MAX, recip_flag=RECIP_FLAG, save=None):
    """Render one plane's landscape QC page.

    Parameters
    ----------
    grid : (n_epochs, N, N) per-epoch normalized-MI objective (NaN allowed for the sparse
           coarse-to-fine format); grid[e] indexed [alpha_idx, beta_idx].
    alpha, beta : (n_epochs,) per-epoch chosen coefficients.
    grid_interval : grid spacing (0.01).
    title : figure suptitle.
    applied : optional (alpha, beta) actually applied (e.g. reciprocity-averaged), marked on
              the stability panels.
    partner : optional (alpha_q, beta_q) per-epoch coefficients of the PARTNER plane. When
              given, a pair-symmetry row is added (reciprocity α_p≈β_q, β_p≈α_q).
    coef_max : fixed upper limit for all coefficient-value axes (comparability).
    recip_flag : reciprocity-gap flag threshold (line on the gap-vs-epoch panel).
    save : optional path; if given, save the figure (Agg) and close it.

    Layout: top block = per-epoch landscapes (square, 2 rows), then a 2x2 (or 1x2 without
    partner) analysis block: argmin stability, α/β vs epoch, [pair symmetry, reciprocity
    gap vs epoch]. Legends sit OUTSIDE the axes; α/β axes are square and fixed to [0, coef_max].
    """
    import matplotlib
    if save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(grid, dtype=float)
    alpha, beta = np.asarray(alpha, float), np.asarray(beta, float)
    n, N = grid.shape[0], grid.shape[1]
    gm = (N - 1) * grid_interval
    ep = np.arange(n)
    finite = grid[np.isfinite(grid)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    ncols = math.ceil(n / 2)
    has_partner = partner is not None
    if has_partner:
        aq, bq = np.asarray(partner[0], float), np.asarray(partner[1], float)
        E = min(n, len(aq))

    bot_rows = 2 if has_partner else 1
    fig = plt.figure(figsize=(1.6 * ncols + 2.4, 4.0 + 2.5 * bot_rows), layout="constrained")
    fig.suptitle(title, fontsize=11)
    sf_land, sf_bot = fig.subfigures(2, 1, height_ratios=[1.5, 1.3 * bot_rows])

    # --- per-epoch landscapes: 2 rows x ncols, square panels, native data extent ---
    axl = sf_land.subplots(2, ncols, squeeze=False)
    sf_land.suptitle("per-epoch MI landscapes (min = r+)", fontsize=9)
    im = None
    for e in range(2 * ncols):
        ax = axl[e // ncols][e % ncols]
        if e < n:
            im = ax.imshow(grid[e].T, origin="lower", extent=[0, gm, 0, gm], vmin=vmin,
                           vmax=vmax, cmap="viridis", aspect="auto")
            if np.isfinite(grid[e]).any():
                ai, bi = np.unravel_index(int(np.nanargmin(grid[e])), grid[e].shape)
                ax.plot(ai * grid_interval, bi * grid_interval, "r+", ms=7)
            ax.set_box_aspect(1)
            ax.set_title(f"ep{e}", fontsize=7); ax.tick_params(labelsize=5)
            if e % ncols == 0: ax.set_ylabel("β", fontsize=7)
            if e // ncols == 1 or n <= ncols: ax.set_xlabel("α", fontsize=7)
        else:
            ax.axis("off")
    if im is not None:
        sf_land.colorbar(im, ax=axl, shrink=0.7, pad=0.01, label="norm. MI (basin=low)")

    axb = sf_bot.subplots(bot_rows, 2, squeeze=False)

    # (0,0) argmin stability -- SQUARE, fixed axes, legend outside right
    ax = axb[0][0]
    ax.scatter(alpha, beta, c=ep, cmap="plasma", s=38, edgecolor="k", linewidth=0.3, zorder=3)
    ax.scatter(alpha.mean(), beta.mean(), marker="*", s=180, c="lime", edgecolor="k",
               zorder=4, label="epoch mean")
    if applied is not None:
        ax.scatter([applied[0]], [applied[1]], marker="X", s=110, c="red", edgecolor="k",
                   zorder=5, label="applied")
    ax.set_xlim(0, coef_max); ax.set_ylim(0, coef_max); ax.set_box_aspect(1)
    ax.set_xlabel("α*", fontsize=8); ax.set_ylabel("β*", fontsize=8)
    ax.set_title(f"argmin stability (σα={alpha.std():.3f}, σβ={beta.std():.3f})", fontsize=8)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # (0,1) alpha/beta vs epoch -- y fixed to [0, coef_max], legend outside right
    ax = axb[0][1]
    ax.plot(ep, alpha, "o-", label="α*"); ax.plot(ep, beta, "s-", label="β*")
    if applied is not None:
        ax.axhline(applied[0], color="C0", ls="--", lw=1, alpha=0.6)
        ax.axhline(applied[1], color="C1", ls="--", lw=1, alpha=0.6)
    ax.set_ylim(0, coef_max); ax.set_xlabel("epoch", fontsize=8); ax.set_ylabel("coefficient", fontsize=8)
    ax.set_title("α / β vs epoch", fontsize=8); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    if has_partner:
        # (1,0) pair symmetry scatter -- SQUARE: this plane (α_p,β_p) vs partner flipped (β_q,α_q)
        ax = axb[1][0]
        ax.scatter(alpha[:E], beta[:E], s=42, facecolor="none", edgecolor="C0", linewidth=1.4,
                   label="this plane (α_p,β_p)")
        ax.scatter(bq[:E], aq[:E], marker="x", s=42, c="C3", label="partner flipped (β_q,α_q)")
        ax.plot([0, coef_max], [0, coef_max], "k:", linewidth=0.6)
        ax.set_xlim(0, coef_max); ax.set_ylim(0, coef_max); ax.set_box_aspect(1)
        ax.set_xlabel("α_p  (leak p→q)", fontsize=8); ax.set_ylabel("β_p  (leak q→p)", fontsize=8)
        gap = 0.5 * (np.mean(np.abs(alpha[:E] - bq[:E])) + np.mean(np.abs(beta[:E] - aq[:E])))
        ax.set_title(f"pair symmetry (recip gap={gap:.3f}, {'OK' if gap < recip_flag else 'FLAG'})", fontsize=8)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

        # (1,1) reciprocity gap vs epoch, legend outside right
        ax = axb[1][1]
        ax.plot(ep[:E], np.abs(alpha[:E] - bq[:E]), "o-", label="|α_p − β_q|")
        ax.plot(ep[:E], np.abs(beta[:E] - aq[:E]), "s-", label="|β_p − α_q|")
        ax.axhline(recip_flag, color="r", ls="--", linewidth=1, label=f"{recip_flag:g} flag")
        ax.set_ylim(0, max(recip_flag * 1.6, float(np.abs(beta[:E] - aq[:E]).max()) * 1.15))
        ax.set_xlabel("epoch", fontsize=8); ax.set_ylabel("reciprocity gap", fontsize=8)
        ax.set_title("reciprocity gap vs epoch", fontsize=8); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    if save is not None:
        fig.savefig(save, dpi=100)
        plt.close(fig)
        return save
    return fig


def plot_landscape_page(asset: io.AssetRef, plane: int, save=None, applied=None,
                        with_symmetry=True):
    """Load a plane's stored per-epoch grids (and, for the symmetry panel, its partner's)
    and render its landscape QC page."""
    g = arrays.load_grid_arrays(asset, plane)
    partner = None
    if with_symmetry:
        try:
            gq = arrays.load_grid_arrays(asset, io.pair_of(plane))
            partner = (gq["alpha"], gq["beta"])
        except Exception:
            partner = None
    title = f"{asset.name}  plane {plane}"
    return render_landscape_page(g["grid"], g["alpha"], g["beta"],
                                 grid_interval=arrays.GRID_INTERVAL, title=title,
                                 applied=applied, partner=partner, save=save)
