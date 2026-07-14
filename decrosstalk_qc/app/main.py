"""Entry point for the decrosstalk visual-QC app.

Examples (on the CO cloud workstation, with a display)::

    cd /root/capsule/code
    python -m decrosstalk_qc.app                 # ALL attached processed sessions
    python -m decrosstalk_qc.app --exemplars     # just the 3 Phase-1 exemplars
    python -m decrosstalk_qc.app --subject 782149
    python -m decrosstalk_qc.app --assets NAME1 NAME2 ...

Labeler and the labels-CSV location can also be set live in the app (top of the
control panel); the CLI flags below just set the initial values.
"""

from __future__ import annotations

import argparse

from .. import io
from ..labels import DEFAULT_PATH, LabelStore
from .model import ImageProvider, ViewerModel

# Phase-1 exemplars (plan §7): all-zero (calibration), good (0.03–0.06), heavy (>0.2)
EXEMPLARS = [
    "multiplane-ophys_800792_2025-07-29_12-31-47_processed_2025-07-30_17-12-17",
    "multiplane-ophys_782149_2025-03-31_12-23-33_processed_2025-09-11_21-09-23",
    "multiplane-ophys_800995_2025-08-22_12-35-43_processed_2025-10-10_22-56-38",
]


def build_worklist(args) -> list[io.AssetRef]:
    if args.assets_file:
        names = [ln.strip() for ln in open(args.assets_file)
                 if ln.strip() and not ln.startswith("#")]
        if not names:
            raise SystemExit(f"No asset names in {args.assets_file}")
        return [io.asset_from_name(n) for n in names]
    if args.assets:
        return [io.asset_from_name(n) for n in args.assets]
    if args.exemplars:
        return [io.asset_from_name(n) for n in EXEMPLARS]
    assets = io.find_processed_assets(subject_id=args.subject)
    if not assets:
        raise SystemExit(
            f"No attached processed assets{' for subject ' + args.subject if args.subject else ''} "
            f"under {io.DATA_DIR}"
        )
    return assets


def main(argv=None):
    parser = argparse.ArgumentParser(description="Decrosstalk visual QC viewer")
    parser.add_argument("--assets", nargs="+", help="specific processed asset names to load")
    parser.add_argument("--assets-file", help="file with one processed asset name per line "
                        "(# comments ok) — used to restrict the worklist, e.g. high-β candidates")
    parser.add_argument("--subject", help="restrict to one subject's attached assets")
    parser.add_argument("--exemplars", action="store_true",
                        help="load only the 3 Phase-1 exemplars")
    parser.add_argument("--labeler", help="initial labeler name (editable in app; default: anonymous)")
    parser.add_argument("--labels-path", default=str(DEFAULT_PATH),
                        help=f"initial labels CSV (changeable in app; default: {DEFAULT_PATH})")
    args = parser.parse_args(argv)

    assets = build_worklist(args)
    model = ViewerModel(assets, ImageProvider())
    store = LabelStore(path=args.labels_path, labeler=args.labeler)
    print(f"labels -> {store.path}  (labeler: {store.labeler})")

    from PyQt5 import QtWidgets  # imported here so build_worklist works headless

    from .viewer import MainWindow

    app = QtWidgets.QApplication([])
    win = MainWindow(model, store)
    win.resize(1400, 760)
    win.show()
    try:
        app.exec_()
    finally:
        io.clear_cache()  # empty the ephemeral image cache on exit (labels/figures kept)


if __name__ == "__main__":
    main()
