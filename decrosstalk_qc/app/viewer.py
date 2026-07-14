"""PyQt5 + pyqtgraph visual-QC viewer for decrosstalk (Phase 1).

Two panels = the two planes of a pair, side by side (left = plane A, right = its
partner B). Both panels show the same role at once, toggled in place:
  - **space** flips PRE ↔ POST
  - **p** toggles PAIRED on/off — the partner plane's `registered_to_pair` (its
    crosstalk contribution in this plane's frame); off returns to PRE/POST.

So: space to see what decrosstalk changed; p to see the crosstalk source and check
(vs POST) whether it was removed. Each image is normalized to its own robust range
so PRE and POST display at matching brightness (the unmixing rescales the signal
globally by ~1/(1−α−β); a shared scale would make POST look much brighter).

Nav: ←/→ pair · ↑/↓ session · s source(mean/epoch) · [ ] epoch. Panels are pinned
to image bounds (mouse zoom disabled), so flips never resize. Logic lives in
:mod:`decrosstalk_qc.app.model` (Qt-free). Run via :mod:`decrosstalk_qc.app.main`.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from ..labels import LABELS, LabelStore
from .model import ViewerModel, contrast_levels

pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("background", "k")
pg.setConfigOption("foreground", "w")

ALL_ROLES = ("pre", "post", "paired")
LABEL_COLORS = {
    "good": "#5c6", "under-corrected": "#fb3", "over-corrected": "#4af",
    "unsure": "#aaa", None: "#888",
}
BTN_TEXT = {"good": "Good", "under-corrected": "Under", "over-corrected": "Over", "unsure": "Unsure"}
# keyboard: 1/2/3/4 label LEFT plane, 5/6/7/8 label RIGHT plane (order = LABELS)
LABEL_KEYS = {}
for _side, _base in ((0, QtCore.Qt.Key_1), (1, QtCore.Qt.Key_5)):
    for _i, _lab in enumerate(LABELS):
        LABEL_KEYS[_base + _i] = (_side, _lab)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, model: ViewerModel, store: LabelStore = None):
        super().__init__()
        self.model = model
        self.store = store or LabelStore()
        self.base = "pre"  # "pre" | "post", toggled by space
        self.show_paired = False  # toggled by p
        self.hi_pct = 99.0
        self._levels_cache: dict = {}
        self._gen = 0  # bumped on every navigation; stale prefetch/NMI jobs self-cancel
        self.setWindowTitle("Decrosstalk visual QC")
        self._build_ui()
        self._sync_controls_from_model()
        self.refresh(fit=True)
        self._schedule_prefetch()

    @property
    def role(self) -> str:
        return "paired" if self.show_paired else self.base

    # --- UI construction ---
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self._build_controls(), 0)
        layout.addWidget(self._build_panes(), 1)
        self.status = self.statusBar()

    def _build_panes(self) -> QtWidgets.QWidget:
        self.glw = pg.GraphicsLayoutWidget()
        self.vbs, self.items, self.hdr = [], [], []
        for c in range(2):
            vb = self.glw.addViewBox(row=0, col=c)
            vb.setAspectLocked(True)
            vb.invertY(True)
            vb.setMouseEnabled(x=False, y=False)  # no zoom/pan
            vb.setMenuEnabled(False)
            vb.disableAutoRange()
            it = pg.ImageItem(autoDownsample=True)
            vb.addItem(it)
            txt = pg.TextItem(anchor=(0, 0), fill=(0, 0, 0, 150))
            txt.setPos(6, 6)
            vb.addItem(txt, ignoreBounds=True)
            self.vbs.append(vb)
            self.items.append(it)
            self.hdr.append(txt)
        gl = self.glw.ci.layout
        for c in range(2):
            gl.setColumnStretchFactor(c, 1)  # equal-width panels
        return self.glw

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMaximumWidth(300)
        form = QtWidgets.QFormLayout(w)

        # labeler name (editable; default anonymous)
        self.labeler_edit = QtWidgets.QLineEdit(self.store.labeler)
        self.labeler_edit.setPlaceholderText("anonymous")
        self.labeler_edit.editingFinished.connect(self._on_labeler)
        form.addRow("Labeler", self.labeler_edit)

        # save location (editable via file dialog) — full-width so the path fits
        save_row = QtWidgets.QWidget()
        sh = QtWidgets.QHBoxLayout(save_row)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.setSpacing(3)
        self.path_lbl = QtWidgets.QLabel()
        self.path_lbl.setStyleSheet("color:#888")
        change_btn = QtWidgets.QPushButton("Change…")
        change_btn.setMaximumWidth(72)
        change_btn.clicked.connect(self._change_save_location)
        sh.addWidget(self.path_lbl, 1)
        sh.addWidget(change_btn, 0)
        form.addRow(QtWidgets.QLabel("Save to"))
        form.addRow(save_row)
        self._update_path_label()

        self.session_combo = QtWidgets.QComboBox()
        for a in self.model.assets:
            self.session_combo.addItem(a.name)
        self.session_combo.currentIndexChanged.connect(self._on_session)
        form.addRow("Session", self.session_combo)

        self.pair_spin = QtWidgets.QSpinBox()
        self.pair_spin.setRange(0, self.model.n_pairs - 1)
        self.pair_spin.valueChanged.connect(self._on_pair)
        form.addRow("Pair", self.pair_spin)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["mean", "epoch"])
        self.source_combo.currentTextChanged.connect(self._on_source)
        form.addRow("Source", self.source_combo)

        self.epoch_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.epoch_slider.setRange(0, max(0, self.model.n_epochs - 1))
        self.epoch_slider.valueChanged.connect(self._on_epoch)
        self.epoch_slider.setEnabled(False)
        form.addRow("Epoch", self.epoch_slider)

        self.pct_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pct_slider.setRange(800, 1000)  # /10 -> 80.0..100.0 percentile
        self.pct_slider.setValue(int(self.hi_pct * 10))
        self.pct_slider.valueChanged.connect(self._on_pct)
        form.addRow("Contrast %ile", self.pct_slider)

        # per-plane label controls (left = plane A, right = plane B)
        self.lbl_caps = []
        self.lbl_btns = []  # [side] -> {label: button}
        for side in range(2):
            cap = QtWidgets.QLabel()
            row = QtWidgets.QWidget()
            hb = QtWidgets.QHBoxLayout(row)
            hb.setContentsMargins(0, 0, 0, 0)
            hb.setSpacing(3)
            btns = {}
            for lab in LABELS:
                b = QtWidgets.QPushButton(BTN_TEXT[lab])
                b.setToolTip(lab)
                b.setCheckable(True)
                b.clicked.connect(lambda _=False, s=side, l=lab: self._label_side(s, l))
                hb.addWidget(b)
                btns[lab] = b
            self.lbl_caps.append(cap)
            self.lbl_btns.append(btns)
            form.addRow(cap)   # caption on its own full-width row...
            form.addRow(row)   # ...buttons span the full panel width below it

        self.note_edit = QtWidgets.QLineEdit()
        self.note_edit.setPlaceholderText("optional note (saved with next label)")
        form.addRow("Note", self.note_edit)

        self.info_lbl = QtWidgets.QLabel()
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setTextFormat(QtCore.Qt.RichText)
        form.addRow(self.info_lbl)

        hint = QtWidgets.QLabel(
            "<b>keys</b>: <b>space</b> PRE↔POST · <b>p</b> PAIRED on/off<br>"
            "←/→ pair · ↑/↓ session · <b>s</b> source · [ ] epoch · <b>n</b> NMI<br>"
            "label (good/under/over/unsure): <b>1-4</b> left · <b>5-8</b> right"
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        return w

    # --- control callbacks ---
    def _sync_controls_from_model(self):
        self.session_combo.blockSignals(True)
        self.session_combo.setCurrentIndex(self.model.asset_idx)
        self.session_combo.blockSignals(False)
        self.pair_spin.blockSignals(True)
        self.pair_spin.setValue(self.model.pair_idx)
        self.pair_spin.blockSignals(False)

    def _on_session(self, idx):
        if self.model.set_asset(idx):
            self._gen += 1
            self._reset_epoch_range()
            self.note_edit.clear()
            self.refresh(fit=True)
            self._schedule_prefetch()

    def _on_pair(self, idx):
        if self.model.set_pair(idx):
            self._gen += 1
            self._reset_epoch_range()
            self.note_edit.clear()
            self.refresh(fit=True)
            self._schedule_prefetch()

    def _on_source(self, source):
        if self.model.set_source(source):
            self._gen += 1
            self.epoch_slider.setEnabled(source == "epoch")
            self.refresh(fit=False)
            self._schedule_prefetch()

    def _on_epoch(self, epoch):
        if self.model.set_epoch(epoch):
            self.refresh(fit=False)

    def _on_pct(self, val):
        self.hi_pct = val / 10.0
        self._levels_cache.clear()
        self.refresh(fit=False)

    def _reset_epoch_range(self):
        self.epoch_slider.blockSignals(True)
        self.epoch_slider.setRange(0, max(0, self.model.n_epochs - 1))
        self.epoch_slider.setValue(self.model.epoch)
        self.epoch_slider.blockSignals(False)

    # --- cooperative background prefetch (keeps nav responsive) ---
    def _schedule_prefetch(self):
        """After painting the visible role, load the *other* roles for this pair in
        the background — one per event-loop tick, so flips become cache hits. Starts
        after a short idle delay so rapid nav (holding an arrow) isn't slowed by
        prefetch ticks, and self-cancels if the user navigates (gen changes). NMI is
        NOT prefetched (too slow); it is on-demand via the `n` key."""
        gen = self._gen
        QtCore.QTimer.singleShot(120, lambda: self._begin_prefetch(gen))

    def _begin_prefetch(self, gen: int):
        if gen != self._gen:
            return
        jobs = [(plane, role) for plane in self.model.pair
                for role in ALL_ROLES if role != self.role]
        self._run_prefetch(gen, jobs)

    def _run_prefetch(self, gen: int, jobs: list):
        if gen != self._gen or not jobs:
            return
        plane, role = jobs[0]
        try:
            self.model.image(plane, role)  # load + cache (≈130 ms), yields after
        except Exception:
            pass
        QtCore.QTimer.singleShot(0, lambda: self._run_prefetch(gen, jobs[1:]))

    # --- rendering ---
    def _fit(self, shape):
        h, w = shape
        rect = QtCore.QRectF(0, 0, w, h)
        for vb in self.vbs:
            vb.setRange(rect, padding=0)

    def _levels(self, plane: int, role: str, img):
        key = (self.model.asset.name, plane, role, self.model.source, self.model.epoch, self.hi_pct)
        if key not in self._levels_cache:
            self._levels_cache[key] = contrast_levels(img, hi_pct=self.hi_pct)
        return self._levels_cache[key]

    def refresh(self, fit: bool = True):
        role = self.role
        shape = None
        for it, plane in zip(self.items, self.model.pair):
            try:
                img = self.model.image(plane, role)  # only the 2 VISIBLE images
                it.setImage(img, levels=self._levels(plane, role, img), autoLevels=False)
                shape = img.shape
            except Exception as exc:
                it.clear()
                self.status.showMessage(f"load error plane {plane} [{role}]: {exc}")
        self._update_overlays()
        self._update_labels_ui()
        if fit and shape:
            self._fit(shape)
        self._update_info()

    def _update_overlays(self):
        role = self.role
        for txt, plane in zip(self.hdr, self.model.pair):
            label = "PAIRED" if role == "paired" else role.upper()
            note = " (partner→this plane)" if role == "paired" else ""
            qc = self.store.get_label(self.model.asset.name, plane)
            tag = ""
            if qc:
                tag = f" · <b style='color:{LABEL_COLORS[qc]}'>{qc.upper()}</b>"
            txt.setHtml(
                f"<div style='font-size:9pt'>plane {plane} · "
                f"<b style='color:#6cf'>{label}</b>{note}{tag}</div>"
            )

    def _on_labeler(self):
        self.store.labeler = self.labeler_edit.text().strip() or "anonymous"
        self.labeler_edit.setText(self.store.labeler)

    def _update_path_label(self):
        p = str(self.store.path)
        shown = p if len(p) <= 30 else "…" + p[-29:]
        self.path_lbl.setText(shown)
        self.path_lbl.setToolTip(p)

    def _change_save_location(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save labels to", str(self.store.path), "CSV files (*.csv)"
        )
        if fname:
            self._set_store_path(fname)

    def _set_store_path(self, path: str):
        """Point the label store at a new CSV (keeping the labeler); loads any labels
        already in that file and refreshes the UI."""
        self.store = LabelStore(path=path, labeler=self.store.labeler)
        self._update_path_label()
        self._update_labels_ui()
        self._update_overlays()
        self._update_info()

    def _label_side(self, side: int, label: str):
        plane = self.model.pair[side]
        self.store.set_label(self.model.asset, plane, label, self.note_edit.text().strip())
        self._update_labels_ui()
        self._update_overlays()
        self._update_info()

    def _update_labels_ui(self):
        for side, plane in enumerate(self.model.pair):
            cur = self.store.get_label(self.model.asset.name, plane)
            col = LABEL_COLORS[cur]
            self.lbl_caps[side].setText(
                f"plane {plane} ({'L' if side == 0 else 'R'}): "
                f"<b style='color:{col}'>{cur.upper() if cur else '—'}</b>"
            )
            self.lbl_caps[side].setTextFormat(QtCore.Qt.RichText)
            for lab, btn in self.lbl_btns[side].items():
                btn.setChecked(lab == cur)

    def _update_info(self):
        i = self.model.info()
        rows = []
        for p in i["planes"]:
            al, be = i["alpha_beta"][p]
            a = "—" if al is None else f"{al:.2f}"
            b = "—" if be is None else f"{be:.2f}"
            nmi = i["nmi"][p]
            nmi_txt = (f"NMI {nmi[0]:.3f}→{nmi[1]:.3f}" if nmi
                       else "<span style='color:#666'>NMI: press n</span>")
            rows.append(f"<b>plane {p}</b> α={a} β={b}<br>&nbsp;{nmi_txt}")
        ep = "" if i["epoch"] is None else f" · epoch {i['epoch']}"
        showing = "PAIRED" if self.role == "paired" else self.base.upper()
        c = self.store.counts()
        tally = " ".join(
            f"<span style='color:{LABEL_COLORS[k]}'>{BTN_TEXT[k].lower()}:{c[k]}</span>" for k in LABELS
        )
        self.info_lbl.setText(
            "<br>".join(rows)
            + f"<br><span style='color:#888'>showing {showing} · {i['source']}{ep}<br>"
            f"pair {i['pair_idx'] + 1}/{self.model.n_pairs} · "
            f"asset {i['asset_idx'] + 1}/{self.model.n_assets}</span>"
            f"<br>labeled — {tally}"
        )
        self.status.showMessage(i["asset"])

    # --- keyboard ---
    def keyPressEvent(self, e):
        k = e.key()
        nav = False
        if k == QtCore.Qt.Key_Space:
            self.show_paired = False
            self.base = "post" if self.base == "pre" else "pre"
            self.refresh(fit=False)
            return
        elif k == QtCore.Qt.Key_P:
            self.show_paired = not self.show_paired
            self.refresh(fit=False)
            return
        elif k in LABEL_KEYS:
            side, label = LABEL_KEYS[k]
            self._label_side(side, label)
            return
        elif k == QtCore.Qt.Key_Right:
            nav = self.model.next_pair()
        elif k == QtCore.Qt.Key_Left:
            nav = self.model.prev_pair()
        elif k == QtCore.Qt.Key_Up:
            nav = self.model.prev_asset()
        elif k == QtCore.Qt.Key_Down:
            nav = self.model.next_asset()
        elif k == QtCore.Qt.Key_S:
            if self.model.toggle_source():
                self._gen += 1
                self.source_combo.blockSignals(True)
                self.source_combo.setCurrentText(self.model.source)
                self.source_combo.blockSignals(False)
                self.epoch_slider.setEnabled(self.model.source == "epoch")
                self.refresh(fit=False)
                self._schedule_prefetch()
            return
        elif k == QtCore.Qt.Key_BracketRight:
            if self.model.source == "epoch" and self.model.next_epoch():
                self._reset_epoch_range()
                self.refresh(fit=False)
            return
        elif k == QtCore.Qt.Key_BracketLeft:
            if self.model.source == "epoch" and self.model.prev_epoch():
                self._reset_epoch_range()
                self.refresh(fit=False)
            return
        elif k == QtCore.Qt.Key_N:
            self._compute_nmi_current()
            return
        else:
            super().keyPressEvent(e)
            return
        if nav:
            self._gen += 1
            self._sync_controls_from_model()
            self._reset_epoch_range()
            self.note_edit.clear()
            self.refresh(fit=True)
            self._schedule_prefetch()

    def _compute_nmi_current(self):
        """On-demand NMI for the current pair (≈0.7 s/plane) — triggered by `n`."""
        self.status.showMessage("computing NMI…")
        for plane in self.model.pair:
            self.model.compute_nmi(plane)
        self._update_info()
