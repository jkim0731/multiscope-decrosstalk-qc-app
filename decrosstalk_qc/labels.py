"""Durable good/bad/unsure labels for decrosstalk QC, one label per (asset, plane).

Append-only CSV under ``/scratch/decrosstalk_qc/labels/`` (artifacts live in
``/scratch``, never under ``code/``). Each label action appends a row; the *latest*
row for an (asset_name, plane) is the current label, so a re-label just appends and
history is preserved.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import io

LABELS = ("good", "under-corrected", "over-corrected", "unsure")
DEFAULT_LABELER = "anonymous"
DEFAULT_PATH = io.ARTIFACT_DIR / "labels" / "decrosstalk_qc_labels.csv"
FIELDS = ["timestamp", "labeler", "asset_id", "asset_name", "plane", "label", "note"]


class LabelStore:
    """Read/append good/bad/unsure labels keyed by (asset_name, plane)."""

    def __init__(self, path=DEFAULT_PATH, labeler: Optional[str] = None):
        self.path = Path(path)
        self.labeler = (labeler or "").strip() or DEFAULT_LABELER
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._latest: dict[tuple[str, int], dict] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    plane = int(row["plane"])
                except (KeyError, ValueError):
                    continue
                self._latest[(row["asset_name"], plane)] = row

    def set_label(self, asset: io.AssetRef, plane: int, label: str, note: str = "") -> dict:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {label!r}")
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "labeler": self.labeler,
            "asset_id": asset.id or "",
            "asset_name": asset.name,
            "plane": plane,
            "label": label,
            "note": note or "",
        }
        write_header = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)
        self._latest[(asset.name, plane)] = row
        return row

    def get(self, asset_name: str, plane: int) -> Optional[dict]:
        return self._latest.get((asset_name, plane))

    def get_label(self, asset_name: str, plane: int) -> Optional[str]:
        row = self.get(asset_name, plane)
        return row["label"] if row else None

    def counts(self) -> dict:
        c = {k: 0 for k in LABELS}
        for row in self._latest.values():
            if row["label"] in c:
                c[row["label"]] += 1
        return c

    def to_df(self):
        """Latest label per (asset_name, plane) as a tidy DataFrame."""
        import pandas as pd

        rows = list(self._latest.values())
        return pd.DataFrame(rows, columns=FIELDS) if rows else pd.DataFrame(columns=FIELDS)
