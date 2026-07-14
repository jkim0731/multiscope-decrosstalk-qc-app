"""Image loading for decrosstalk QC: PRE / POST / registered-to-PAIRED.

Three roles for a signal plane ``p``, all in ``p``'s frame, shape (n_epochs, H, W):
  - ``pre``    motion-corrected signal   ``VISp_p_registered_episodic_mean_fov.h5``
  - ``post``   decrosstalked signal      ``VISp_p_decrosstalk_episodic_mean_fov.h5``
  - ``paired`` the *partner* plane's crosstalk contribution, registered into ``p``'s
    frame — ``VISp_{pair_of(p)}_registered_to_pair_episodic_mean_fov.h5``. NB this is
    the **partner's** file, per each plane's `paired_emf` metadata; the pipeline uses
    exactly this file as the paired input when decrosstalking ``p``.

The "full-session mean" image (plan §2) is the mean over the episodic-mean-FOV stack
(~10 epochs, each an average of ~1000 frames sampled across the session) — a
quantitative, float proxy for the whole-session average, consistent across all three
roles. The 8-bit ``*_avg_img.png`` are also available for instant previews.

Remote (S3) reads are cached as ``.npy`` under ``CACHE_DIR``; local ``/data`` reads
go straight from disk (already fast).
"""

from __future__ import annotations

import numpy as np

from . import io

# role -> episodic-mean-FOV h5 suffix
EMF_SUFFIX = {
    "pre": "registered_episodic_mean_fov.h5",
    "post": "decrosstalk_episodic_mean_fov.h5",
    "paired": "registered_to_pair_episodic_mean_fov.h5",
}
# role -> full-session average preview PNG
AVG_PNG_SUFFIX = {
    "pre": "registered_avg_img.png",
    "post": "decrosstalk_avg_img.png",
    "paired": "registered_to_pair_avg_img.png",
}
ROLES = tuple(EMF_SUFFIX)


def _check_role(kind: str) -> None:
    if kind not in EMF_SUFFIX:
        raise ValueError(f"kind must be one of {ROLES}, got {kind!r}")


def _file_plane(plane: int, kind: str) -> int:
    """Which VISp_<n> file holds a role for signal ``plane``.

    ``paired`` lives in the *partner* plane's ``registered_to_pair`` file
    (``VISp_{pair_of(plane)}_...``); ``pre``/``post`` are the plane's own files.
    """
    return io.pair_of(plane) if kind == "paired" else plane


def load_emf(asset: io.AssetRef, plane: int, kind: str, use_cache: bool = True) -> np.ndarray:
    """Episodic-mean-FOV stack ``(n_epochs, H, W)`` float for signal ``plane`` and role."""
    _check_role(kind)
    path = io.plane_file(asset, _file_plane(plane, kind), EMF_SUFFIX[kind])

    if asset.is_local:
        return np.asarray(io.read_h5_dataset(path, "data"), dtype=float)

    # remote: cache the (small) stack locally as .npy
    cache = io.cache_path(asset, f"emf_p{plane}_{kind}.npy")
    if use_cache and cache.exists():
        return np.load(cache)
    arr = np.asarray(io.read_h5_dataset(path, "data"), dtype=float)
    if use_cache:
        np.save(cache, arr)
    return arr


def load_mean_img(asset: io.AssetRef, plane: int, kind: str, use_cache: bool = True) -> np.ndarray:
    """Full-session mean image ``(H, W)`` = mean over the EMF stack epochs.

    The 20 MB EMF stack is transient here (averaged then released); only the 2 MB
    mean is returned, so callers can cache the mean cheaply.
    """
    return load_emf(asset, plane, kind, use_cache=use_cache).mean(axis=0)


def n_epochs(asset: io.AssetRef, plane: int, kind: str = "pre") -> int:
    """Number of epochs in a plane's EMF stack (cheap: reads only the shape)."""
    path = io.plane_file(asset, _file_plane(plane, kind), EMF_SUFFIX[kind])
    return int(io.read_h5_shape(path, "data")[0])


def load_avg_png(asset: io.AssetRef, plane: int, kind: str) -> np.ndarray:
    """8-bit full-session average preview PNG ``(H, W)`` (instant; not quantitative)."""
    _check_role(kind)
    import imageio.v3 as iio

    path = io.plane_file(asset, _file_plane(plane, kind), AVG_PNG_SUFFIX[kind])
    if io._is_remote(path):
        import fsspec

        with fsspec.open(str(path), "rb") as fobj:
            img = iio.imread(fobj)
    else:
        img = iio.imread(str(path))
    if img.ndim == 3:  # RGB(A) -> luminance
        img = img[..., :3].mean(axis=-1)
    return np.asarray(img, dtype=float)
