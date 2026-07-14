# multiscope-decrosstalk-qc-app

Interactive **Qt visual-QC app** and supporting metrics for multiplane-ophys (2-photon
mesoscope) **decrosstalk** outputs. It loads PRE / POST / paired-registered mean-FOV
images and the per-epoch α/β grid (mutual-information) landscapes from *processed* data
assets and lets you flip through pairs/epochs, adjust contrast, and label pairs
(`good` / `under-corrected` / `over-corrected` / `unsure`).

This is the **app package only**. To run it against attached data assets in a Code Ocean
capsule (the usual workflow), use the companion capsule repo
[`multiscope-decrosstalk-qc-using-gui-app`](https://github.com/AllenNeuralDynamics/multiscope-decrosstalk-qc-using-gui-app),
which builds the Qt desktop environment, installs this package, and attaches the assets.

> Built for the Code Ocean environment: at import the package redirects temp/cache to
> `/scratch` (`decrosstalk_qc.io.use_scratch_tmp`), and it resolves *attached* assets from
> `/root/capsule/data`. See "Data model" below.

## Install

```bash
pip install git+https://github.com/jkim0731/multiscope-decrosstalk-qc-app.git
# S3 / non-attached asset lookup (optional):
pip install "git+https://github.com/jkim0731/multiscope-decrosstalk-qc-app.git#egg=multiscope-decrosstalk-qc-app[remote]"
```

In a conda/mamba environment that already provides Qt (`pyqt=5`), install without letting
pip pull a conflicting PyQt5 wheel:

```bash
pip install numpy pandas scipy scikit-image h5py matplotlib pyqtgraph fsspec s3fs
pip install --no-deps git+https://github.com/jkim0731/multiscope-decrosstalk-qc-app.git
```

## Run the GUI

```bash
python -m decrosstalk_qc.app --help
python -m decrosstalk_qc.app                       # all attached processed assets under /data
python -m decrosstalk_qc.app --subject 782149      # restrict to one subject's attached assets
python -m decrosstalk_qc.app --assets NAME1 NAME2   # specific processed asset names
python -m decrosstalk_qc.app --assets-file list.txt # one processed asset name per line
python -m decrosstalk_qc.app --exemplars           # the 3 built-in Phase-1 exemplars
# choose the labeler name and where labels are saved (also editable in the app):
python -m decrosstalk_qc.app --labeler <labeler_name> --labels-path /scratch/labels.csv
```

The console entry point `multiscope-decrosstalk-qc` is equivalent to
`python -m decrosstalk_qc.app`. The GUI needs a display (the companion capsule provides a
virtual desktop).

## Data model

- **Attached assets** (default): each processed asset is a directory under
  `/root/capsule/data/`; the app reads the small `alpha_list` / `beta_list` /
  `mean_norm_mi_list` datasets and the episodic-mean-FOV images per plane (never the
  multi-GB movie).
- **Non-attached assets**: resolved by name from the Code Ocean data catalog / S3 via the
  optional `aind-session` dependency.

## Package layout

```
decrosstalk_qc/
  io.py         asset resolution (attached /data or S3), h5 readers, /scratch tmp redirect
  images.py     PRE / POST / paired mean-FOV + episodic-mean-FOV loaders (cached)
  arrays.py     per-epoch α/β + MI-objective grid (auto-detects 31x31 dense / 37x37 sparse)
  labels.py     LabelStore (CSV of good/under/over/unsure labels)
  metrics.py    coefficient / landscape-quality / background-correlation features
  plots.py      one-page landscape QC figure (landscapes + stability + pair symmetry)
  repro_estimator.py  faithful re-estimation of α/β (basic segmentation + fast grid)
  scan.py       batch per-plane feature table
  app/          Qt GUI: main.py (CLI+launch), model.py (state), viewer.py (window)
```
