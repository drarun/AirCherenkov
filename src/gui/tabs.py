"""Workflow tabs for Array Config, Simulation, Model Training, and IRF/Spectral Analysis."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.collections import PatchCollection
import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from analysis.irf import save_irfs
from analysis.spectral_fit import (
    CRAB_F0_MAGIC,
    CRAB_F0_VERITAS,
    CRAB_INDEX_MAGIC,
    CRAB_INDEX_VERITAS,
    power_law_flux,
)
from gui.array_models import (
    DetectorArrayModel,
    get_preset_model,
    load_detector_model_json,
    save_detector_model_json,
)
from gui.sim_monitor import SimulationMonitorWidget
from gui.workers import (
    AnalysisIRFSpectralWorker,
    CORE_CONTAINED_BIAS_PCT,
    CORE_CONTAINED_E_TEV,
    CORE_CONTAINED_RES_PCT,
    EvaluateModelWorker,
    SimulationWorker,
    TrainingWorker,
    VERITAS_BASELINE_BIAS_PCT,
    VERITAS_BASELINE_E_TEV,
    VERITAS_BASELINE_RES_PCT,
    VERITAS_BENCHMARK_TIERS,
    compute_irf_and_spectrum_from_arrays,
    compute_skymap_from_inference_arrays,
    describe_model_and_dataset,
    evaluate_existing_model_on_dataset,
    format_baseline_comparison,
    load_upgraded_gnn_benchmark_evaluation,
)
from sim.fadc import FADCConfig


def _style_dark_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, alpha=0.22, which="both", color="#64748b")
    if title:
        ax.set_title(title, color="#f1f5f9", fontsize=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color="#cbd5e1", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color="#cbd5e1", fontsize=9)


class ArrayConfigTab(QWidget):
    """Tab 1: Array presets (VERITAS, HESS, MAGIC), custom telescope editing, and detector I/O."""

    array_model_changed = Signal(object)
    raw_data_imported = Signal(str)

    COLUMNS = [
        "X (m)",
        "Y (m)",
        "Z (m)",
        "Mirror R (m)",
        "Focal L (m)",
        "Reflectivity",
        "QE",
        "Cam Rings",
        "Pixel Deg",
    ]

    def __init__(self, initial_model: DetectorArrayModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model = initial_model
        self._build_ui()
        self._populate_from_model(self.model)

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(8)

        # Preset & file bar
        preset_box = QGroupBox("Array Configuration & Detector Models")
        preset_l = QVBoxLayout(preset_box)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Experiment Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["VERITAS", "HESS", "MAGIC", "Custom"])
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        row1.addWidget(self.preset_combo, stretch=1)

        self.btn_load_json = QPushButton("Load Detector (.json)")
        self.btn_load_json.clicked.connect(self._load_model_json)
        self.btn_save_json = QPushButton("Save Detector (.json)")
        self.btn_save_json.clicked.connect(self._save_model_json)
        self.btn_import_raw = QPushButton("Load Experiment Data (.vbf/.h5/.root)")
        self.btn_import_raw.clicked.connect(self._import_raw_data)
        row1.addWidget(self.btn_load_json)
        row1.addWidget(self.btn_save_json)
        preset_l.addLayout(row1)
        preset_l.addWidget(self.btn_import_raw)
        left_v.addWidget(preset_box)

        # Telescope table
        tel_box = QGroupBox("Telescope Array Layout & Optics")
        tel_v = QVBoxLayout(tel_box)
        self.tel_table = QTableWidget(0, len(self.COLUMNS))
        self.tel_table.setHorizontalHeaderLabels(self.COLUMNS)
        self.tel_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tel_v.addWidget(self.tel_table)

        btn_row = QHBoxLayout()
        self.btn_add_tel = QPushButton("+ Add Telescope")
        self.btn_add_tel.clicked.connect(self._add_telescope_row)
        self.btn_rem_tel = QPushButton("- Remove Selected")
        self.btn_rem_tel.clicked.connect(self._remove_telescope_row)
        btn_row.addWidget(self.btn_add_tel)
        btn_row.addWidget(self.btn_rem_tel)
        btn_row.addStretch(1)
        tel_v.addLayout(btn_row)
        left_v.addWidget(tel_box, stretch=1)

        # Readout & Trigger group
        det_box = QGroupBox("FADC Readout & Array Trigger")
        form = QFormLayout(det_box)
        self.spin_alt = QDoubleSpinBox(); self.spin_alt.setRange(0.0, 6000.0); self.spin_alt.setSuffix(" m")
        self.spin_bins = QSpinBox(); self.spin_bins.setRange(4, 128)
        self.spin_bw = QDoubleSpinBox(); self.spin_bw.setRange(0.1, 20.0); self.spin_bw.setSuffix(" ns")
        self.spin_nsb = QDoubleSpinBox(); self.spin_nsb.setRange(0.0, 50.0); self.spin_nsb.setDecimals(2)
        self.spin_ped = QDoubleSpinBox(); self.spin_ped.setRange(0.0, 20.0); self.spin_ped.setDecimals(2)
        self.spin_sat = QDoubleSpinBox(); self.spin_sat.setRange(10.0, 100000.0)
        self.spin_lg = QDoubleSpinBox(); self.spin_lg.setRange(1.0, 100.0)
        self.spin_trig_pe = QDoubleSpinBox(); self.spin_trig_pe.setRange(0.5, 500.0); self.spin_trig_pe.setSuffix(" p.e.")
        self.spin_trig_pix = QSpinBox(); self.spin_trig_pix.setRange(1, 20)
        self.spin_trig_tels = QSpinBox(); self.spin_trig_tels.setRange(1, 32)

        form.addRow("Observatory Altitude:", self.spin_alt)
        form.addRow("FADC Time Bins / Width:", self._pair(self.spin_bins, self.spin_bw))
        form.addRow("NSB Rate / Pedestal Std:", self._pair(self.spin_nsb, self.spin_ped))
        form.addRow("Saturation / Low-Gain Factor:", self._pair(self.spin_sat, self.spin_lg))
        form.addRow("Trigger PE / Min Pix / Min Tels:", self._triple(self.spin_trig_pe, self.spin_trig_pix, self.spin_trig_tels))
        left_v.addWidget(det_box)

        self.btn_apply = QPushButton("Apply Detector Configuration to Workspace")
        self.btn_apply.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 8px;")
        self.btn_apply.clicked.connect(self._apply_changes)
        left_v.addWidget(self.btn_apply)

        # Right side: Live Array + Camera Footprint Preview
        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(6.5, 7.0), facecolor="#0b1120")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_v.addWidget(self.toolbar)
        right_v.addWidget(self.canvas, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([640, 620])
        root.addWidget(splitter)

    @staticmethod
    def _pair(w1: QWidget, w2: QWidget) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w1)
        h.addWidget(w2)
        return box

    @staticmethod
    def _triple(w1: QWidget, w2: QWidget, w3: QWidget) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w1)
        h.addWidget(w2)
        h.addWidget(w3)
        return box

    def _populate_from_model(self, model: DetectorArrayModel) -> None:
        self.preset_combo.blockSignals(True)
        idx = self.preset_combo.findText(model.preset_name)
        self.preset_combo.setCurrentIndex(idx if idx >= 0 else self.preset_combo.findText("Custom"))
        self.preset_combo.blockSignals(False)

        self.spin_alt.setValue(model.observatory_altitude_m)
        self.spin_bins.setValue(model.fadc_config.n_time_bins)
        self.spin_bw.setValue(model.fadc_config.bin_width_ns)
        self.spin_nsb.setValue(model.fadc_config.nsb_rate)
        self.spin_ped.setValue(model.fadc_config.pedestal_std)
        self.spin_sat.setValue(model.fadc_config.saturation_limit)
        self.spin_lg.setValue(model.fadc_config.low_gain_factor)
        self.spin_trig_pe.setValue(model.trigger_threshold_pe)
        self.spin_trig_pix.setValue(model.trigger_min_pixels)
        self.spin_trig_tels.setValue(model.trigger_min_telescopes)

        self.tel_table.setRowCount(0)
        keys = [
            "x_tel",
            "y_tel",
            "z_tel",
            "mirror_radius",
            "focal_length",
            "mirror_reflectivity",
            "quantum_efficiency",
            "n_rings",
            "pixel_size",
        ]
        for spec in model.telescopes_spec:
            row = self.tel_table.rowCount()
            self.tel_table.insertRow(row)
            for col_i, k in enumerate(keys):
                val = spec[k]
                txt = str(int(val)) if k == "n_rings" else f"{float(val):.3g}"
                self.tel_table.setItem(row, col_i, QTableWidgetItem(txt))

        self._update_preview(model)

    def _read_model_from_widgets(self) -> DetectorArrayModel:
        fadc = FADCConfig(
            n_time_bins=int(self.spin_bins.value()),
            bin_width_ns=float(self.spin_bw.value()),
            nsb_rate=float(self.spin_nsb.value()),
            pedestal_std=float(self.spin_ped.value()),
            saturation_limit=float(self.spin_sat.value()),
            low_gain_factor=float(self.spin_lg.value()),
        )
        specs = []
        keys = [
            "x_tel",
            "y_tel",
            "z_tel",
            "mirror_radius",
            "focal_length",
            "mirror_reflectivity",
            "quantum_efficiency",
            "n_rings",
            "pixel_size",
        ]
        for row in range(self.tel_table.rowCount()):
            spec: Dict[str, float | int] = {}
            for col_i, k in enumerate(keys):
                item = self.tel_table.item(row, col_i)
                raw = item.text() if item else "0"
                spec[k] = int(float(raw)) if k == "n_rings" else float(raw)
            specs.append(spec)

        return DetectorArrayModel(
            preset_name=self.preset_combo.currentText(),
            observatory_altitude_m=float(self.spin_alt.value()),
            trigger_threshold_pe=float(self.spin_trig_pe.value()),
            trigger_min_pixels=int(self.spin_trig_pix.value()),
            trigger_min_telescopes=int(self.spin_trig_tels.value()),
            fadc_config=fadc,
            telescopes_spec=specs,
        )

    @Slot(str)
    def _on_preset_selected(self, name: str) -> None:
        if name in ("VERITAS", "HESS", "MAGIC"):
            self.model = get_preset_model(name)
            self._populate_from_model(self.model)
            self.array_model_changed.emit(self.model)

    @Slot()
    def _add_telescope_row(self) -> None:
        row = self.tel_table.rowCount()
        self.tel_table.insertRow(row)
        defaults = ["0.0", "0.0", f"{self.spin_alt.value():.1f}", "6.0", "12.0", "0.82", "0.20", "12", "0.15"]
        for c, val in enumerate(defaults):
            self.tel_table.setItem(row, c, QTableWidgetItem(val))
        self.preset_combo.setCurrentText("Custom")

    @Slot()
    def _remove_telescope_row(self) -> None:
        row = self.tel_table.currentRow()
        if row >= 0 and self.tel_table.rowCount() > 1:
            self.tel_table.removeRow(row)
            self.preset_combo.setCurrentText("Custom")

    @Slot()
    def _apply_changes(self) -> None:
        try:
            self.model = self._read_model_from_widgets()
            # Validate by building array
            self.model.build_telescope_array()
            self._update_preview(self.model)
            self.array_model_changed.emit(self.model)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Detector Configuration", str(exc))

    @Slot()
    def _save_model_json(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(
            self, "Save Detector Model", f"data/{self.model.preset_name.lower()}_detector.json", "JSON (*.json)"
        )
        if fpath:
            self.model = self._read_model_from_widgets()
            save_detector_model_json(self.model, fpath)

    @Slot()
    def _load_model_json(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(self, "Load Detector Model", "data", "JSON (*.json)")
        if fpath:
            self.model = load_detector_model_json(fpath)
            self._populate_from_model(self.model)
            self.array_model_changed.emit(self.model)

    @Slot()
    def _import_raw_data(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Import Raw Experiment or Batch Data",
            "data",
            "Supported (*.vbf *.h5 *.hdf5 *.root *.pt *.npz);;All Files (*.*)",
        )
        if fpath:
            self.raw_data_imported.emit(fpath)

    def _update_preview(self, model: DetectorArrayModel) -> None:
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

        _style_dark_ax(
            ax1,
            title=f"{model.preset_name} Ground Array ({len(model.telescopes_spec)} Telescopes, Alt={model.observatory_altitude_m:.0f} m)",
            xlabel="East / X (m)",
            ylabel="North / Y (m)",
        )
        xs = [float(t["x_tel"]) for t in model.telescopes_spec]
        ys = [float(t["y_tel"]) for t in model.telescopes_spec]
        for idx, t in enumerate(model.telescopes_spec):
            x, y, r = float(t["x_tel"]), float(t["y_tel"]), float(t["mirror_radius"])
            circ = Circle((x, y), radius=max(r, 4.0), facecolor="#38bdf8", edgecolor="#f8fafc", alpha=0.85, lw=1.5)
            ax1.add_patch(circ)
            ax1.text(x, y + max(r, 5.0) + 4.0, f"T{idx + 1} (D={2 * r:.0f}m)", color="#e2e8f0", ha="center", fontsize=8.5)
        if xs and ys:
            pad = max(35.0, 0.3 * max(np.ptp(xs), np.ptp(ys), 50.0))
            ax1.set_xlim(min(xs) - pad, max(xs) + pad)
            ax1.set_ylim(min(ys) - pad, max(ys) + pad)
        ax1.set_aspect("equal")

        cam = model.primary_camera()
        fov_deg = 2.0 * cam.n_rings * cam.pixel_size
        _style_dark_ax(
            ax2,
            title=f"Primary Camera Geometry ({cam.n_pixels} Hex Pixels, FoV ~ {fov_deg:.2f}°)",
            xlabel="Camera X (deg)",
            ylabel="Camera Y (deg)",
        )
        rad = cam.pixel_size / np.sqrt(3.0)
        patches = [
            RegularPolygon((px, py), numVertices=6, radius=rad, orientation=np.radians(30))
            for px, py in zip(cam.pixel_x, cam.pixel_y)
        ]
        pc = PatchCollection(patches, facecolor="#1e293b", edgecolor="#38bdf8", linewidth=0.45, alpha=0.9)
        ax2.add_collection(pc)
        lim = (cam.n_rings + 1.2) * cam.pixel_size
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_aspect("equal")

        self.fig.tight_layout()
        self.canvas.draw_idle()


class SimulationTab(QWidget):
    """Tab 2: GPU-accelerated MC simulation generator with live energy/core telemetry."""

    simulation_status = Signal(int, str)
    dataset_ready = Signal(str)
    open_in_viewer_requested = Signal(str)

    def __init__(self, array_model: DetectorArrayModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.array_model = array_model
        self.worker: Optional[SimulationWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Monte Carlo Shower & Ray-Trace Parameters")
        form = QFormLayout(grp)

        self.spin_ng = QSpinBox(); self.spin_ng.setRange(0, 500000); self.spin_ng.setValue(500)
        self.spin_nh = QSpinBox(); self.spin_nh.setRange(0, 500000); self.spin_nh.setValue(500)
        self.spin_emin = QDoubleSpinBox(); self.spin_emin.setRange(10.0, 100000.0); self.spin_emin.setValue(100.0); self.spin_emin.setSuffix(" GeV")
        self.spin_emax = QDoubleSpinBox(); self.spin_emax.setRange(20.0, 500000.0); self.spin_emax.setValue(30000.0); self.spin_emax.setSuffix(" GeV")
        self.spin_g_idx = QDoubleSpinBox(); self.spin_g_idx.setRange(0.5, 5.0); self.spin_g_idx.setValue(2.0); self.spin_g_idx.setSingleStep(0.1)
        self.spin_hemin = QDoubleSpinBox(); self.spin_hemin.setRange(10.0, 500000.0); self.spin_hemin.setValue(300.0); self.spin_hemin.setSuffix(" GeV")
        self.spin_hemax = QDoubleSpinBox(); self.spin_hemax.setRange(20.0, 1000000.0); self.spin_hemax.setValue(90000.0); self.spin_hemax.setSuffix(" GeV")
        self.spin_h_idx = QDoubleSpinBox(); self.spin_h_idx.setRange(0.5, 5.0); self.spin_h_idx.setValue(2.7); self.spin_h_idx.setSingleStep(0.1)
        self.spin_zen = QDoubleSpinBox(); self.spin_zen.setRange(0.0, 75.0); self.spin_zen.setValue(30.0); self.spin_zen.setSuffix(" deg")
        self.spin_azi = QDoubleSpinBox(); self.spin_azi.setRange(-180.0, 360.0); self.spin_azi.setValue(0.0); self.spin_azi.setSuffix(" deg")
        self.spin_wobble = QDoubleSpinBox(); self.spin_wobble.setRange(0.0, 5.0); self.spin_wobble.setValue(0.5); self.spin_wobble.setSuffix(" deg")
        self.combo_wobble = QComboBox(); self.combo_wobble.addItems(["cardinal", "continuous"])
        self.spin_fov = QDoubleSpinBox(); self.spin_fov.setRange(0.0, 10.0); self.spin_fov.setValue(2.5); self.spin_fov.setSuffix(" deg")
        self.spin_imp = QDoubleSpinBox(); self.spin_imp.setRange(50.0, 3000.0); self.spin_imp.setValue(750.0); self.spin_imp.setSuffix(" m")
        self.spin_batch = QSpinBox(); self.spin_batch.setRange(1, 2000); self.spin_batch.setValue(100)
        self.spin_save = QSpinBox(); self.spin_save.setRange(10, 10000); self.spin_save.setValue(500)
        self.combo_dev = QComboBox(); self.combo_dev.addItems(["auto", "cuda", "cpu"])

        out_row = QWidget()
        out_h = QHBoxLayout(out_row)
        out_h.setContentsMargins(0, 0, 0, 0)
        self.edit_out = QLineEdit("data/gui_sim/raw")
        self.edit_out.editingFinished.connect(
            lambda: self.set_monitored_directory(self.edit_out.text().strip())
        )
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_out)
        out_h.addWidget(self.edit_out, stretch=1)
        out_h.addWidget(btn_browse)

        self.chk_append = QCheckBox("Concatenate (+N events) onto existing simulation batches")
        self.chk_append.setChecked(False)
        from sim.provenance import SIM_SCHEMA_VERSION
        self.edit_sim_version = QLineEdit(SIM_SCHEMA_VERSION)
        self.edit_sim_version.setMaximumWidth(140)
        self.edit_sim_version.setToolTip("Version tag recorded in sim_manifest.json and each saved event")

        form.addRow("Target Gammas / Hadrons:", ArrayConfigTab._pair(self.spin_ng, self.spin_nh))
        form.addRow("Gamma E_min / E_max:", ArrayConfigTab._pair(self.spin_emin, self.spin_emax))
        form.addRow("Gamma Index / Impact R:", ArrayConfigTab._pair(self.spin_g_idx, self.spin_imp))
        form.addRow("Hadron E_min / E_max:", ArrayConfigTab._pair(self.spin_hemin, self.spin_hemax))
        form.addRow("Hadron Index / Diffuse FoV:", ArrayConfigTab._pair(self.spin_h_idx, self.spin_fov))
        form.addRow("Pointing Zenith / Azimuth:", ArrayConfigTab._pair(self.spin_zen, self.spin_azi))
        form.addRow("Wobble Offset / Mode:", ArrayConfigTab._pair(self.spin_wobble, self.combo_wobble))
        form.addRow("Batch / Save Every / Device:", ArrayConfigTab._triple(self.spin_batch, self.spin_save, self.combo_dev))
        form.addRow("Output Directory:", out_row)
        form.addRow("Append / Sim Version:", ArrayConfigTab._pair(self.chk_append, self.edit_sim_version))
        left_v.addWidget(grp)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start / Resume Simulation")
        self.btn_start.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 7px;")
        self.btn_start.clicked.connect(self.start_simulation)
        self.btn_stop = QPushButton("Pause & Save Checkpoint")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_simulation)
        self.btn_open_viewer = QPushButton("Open in Event Viewer")
        self.btn_open_viewer.clicked.connect(lambda: self.open_in_viewer_requested.emit(self.edit_out.text().strip()))
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_open_viewer)
        left_v.addLayout(btn_row)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 8.5))
        left_v.addWidget(self.log_box, stretch=1)

        # Right side live monitor & plots
        default_monitor_dir = (
            "data/corsika_crab_30deg/raw"
            if os.path.isdir("data/corsika_crab_30deg/raw")
            else self.edit_out.text().strip()
        )
        self.monitor_widget = SimulationMonitorWidget(initial_dir=default_monitor_dir, parent=self)
        self.monitor_widget.cohort_data_ready.connect(self._on_batch_completed)

        right_map = QWidget()
        right_v = QVBoxLayout(right_map)
        right_v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(6.5, 7.0), facecolor="#0b1120")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_v.addWidget(self.toolbar)
        right_v.addWidget(self.canvas, stretch=1)
        self._init_empty_plots()

        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.monitor_widget, "Live Progress & Timeseries")
        self.right_tabs.addTab(right_map, "Live Energy & Core Map")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.right_tabs)
        splitter.setSizes([520, 740])
        root.addWidget(splitter)

    def _init_empty_plots(self) -> None:
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        _style_dark_ax(ax1, "Live Triggered Energy Spectrum", "Energy (GeV)", "Triggered Count")
        _style_dark_ax(ax2, "Live Ground Impact Cores", "Core X (m)", "Core Y (m)")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def set_array_model(self, model: DetectorArrayModel) -> None:
        self.array_model = model

    def set_monitored_directory(self, path: str) -> None:
        if path:
            self.monitor_widget.set_directory(path)

    def shutdown(self) -> None:
        self.monitor_widget.shutdown()

    @Slot()
    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Simulation Output Directory", "data")
        if d:
            self.edit_out.setText(d)
            self.set_monitored_directory(d)

    def build_params(self) -> Dict[str, Any]:
        return {
            "num_gammas": self.spin_ng.value(),
            "num_hadrons": self.spin_nh.value(),
            "e_min": self.spin_emin.value(),
            "e_max": self.spin_emax.value(),
            "spectral_index": self.spin_g_idx.value(),
            "hadron_e_min": self.spin_hemin.value(),
            "hadron_e_max": self.spin_hemax.value(),
            "hadron_spectral_index": self.spin_h_idx.value(),
            "zenith_deg": self.spin_zen.value(),
            "azimuth_deg": self.spin_azi.value(),
            "wobble_offset": self.spin_wobble.value(),
            "wobble_mode": self.combo_wobble.currentText(),
            "diffuse_fov": self.spin_fov.value(),
            "impact_radius": self.spin_imp.value(),
            "batch_size": self.spin_batch.value(),
            "save_every": self.spin_save.value(),
            "device": self.combo_dev.currentText(),
            "output_dir": self.edit_out.text().strip(),
            "append_mode": bool(self.chk_append.isChecked()),
            "sim_version": self.edit_sim_version.text().strip(),
        }

    @Slot()
    def start_simulation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        params = self.build_params()
        self.set_monitored_directory(params["output_dir"])
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_box.appendPlainText("Starting MC Simulation...")
        self.worker = SimulationWorker(self.array_model, params, parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log_box.appendPlainText)
        self.worker.batch_completed.connect(self._on_batch_completed)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    @Slot()
    def stop_simulation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.log_box.appendPlainText("Pause requested — finishing active batch and saving checkpoint...")

    @Slot(int, str, dict)
    def _on_progress(self, pct: int, msg: str, _meta: dict) -> None:
        self.simulation_status.emit(pct, msg)

    @Slot(dict)
    def _on_batch_completed(self, payload: dict) -> None:
        energies = np.asarray(payload["energies"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        ix = np.asarray(payload["impact_x"], dtype=np.float64)
        iy = np.asarray(payload["impact_y"], dtype=np.float64)
        valid = np.isfinite(energies) & (energies > 0.0) & np.isfinite(ix) & np.isfinite(iy)
        if not np.any(valid):
            return
        energies, labels, ix, iy = energies[valid], labels[valid], ix[valid], iy[valid]

        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

        _style_dark_ax(ax1, "Live Triggered Energy Distribution", "True Energy (GeV)", "Events")
        e_lo = max(float(np.min(energies)), 1.0)
        e_hi = max(float(np.max(energies)), e_lo * 1.25)
        bins = np.logspace(np.log10(e_lo), np.log10(e_hi), 25)
        g_mask = labels == 1
        h_mask = labels == 0
        counts_list = []
        if np.any(g_mask):
            cg, _ = np.histogram(energies[g_mask], bins=bins)
            counts_list.append(int(np.max(cg)))
            ax1.hist(energies[g_mask], bins=bins, alpha=0.75, color="#38bdf8", label=f"Gammas ({int(np.sum(g_mask)):,})")
        if np.any(h_mask):
            ch, _ = np.histogram(energies[h_mask], bins=bins)
            counts_list.append(int(np.max(ch)))
            ax1.hist(energies[h_mask], bins=bins, alpha=0.6, color="#f43f5e", label=f"Hadrons ({int(np.sum(h_mask)):,})")
        ax1.set_xscale("log")
        if counts_list and max(counts_list) > 20:
            ax1.set_yscale("log")
        ax1.legend(fontsize=8.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        _style_dark_ax(ax2, "Live Triggered Ground Core Impact Map", "Impact X (m)", "Impact Y (m)")
        if len(energies) > 6000:
            stride = max(1, len(energies) // 6000)
            g_s, h_s = g_mask[::stride], h_mask[::stride]
            ix_s, iy_s = ix[::stride], iy[::stride]
        else:
            g_s, h_s, ix_s, iy_s = g_mask, h_mask, ix, iy
        if np.any(g_s):
            ax2.scatter(ix_s[g_s], iy_s[g_s], s=10, alpha=0.65, color="#38bdf8", label=f"Gamma core ({int(np.sum(g_mask)):,})")
        if np.any(h_s):
            ax2.scatter(ix_s[h_s], iy_s[h_s], s=10, alpha=0.5, color="#f43f5e", label=f"Hadron core ({int(np.sum(h_mask)):,})")
        tx = [float(t["x_tel"]) for t in self.array_model.telescopes_spec]
        ty = [float(t["y_tel"]) for t in self.array_model.telescopes_spec]
        ax2.scatter(tx, ty, s=90, marker="^", color="#fbbf24", edgecolor="#0f172a", zorder=5, label="Telescopes")
        ax2.set_aspect("equal")
        ax2.legend(fontsize=8.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    @Slot(str)
    def _on_finished(self, out_dir: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.dataset_ready.emit(out_dir)

    @Slot(str)
    def _on_failed(self, err: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.simulation_status.emit(0, "Error")
        self.log_box.appendPlainText(f"ERROR: {err}")


class TrainingTab(QWidget):
    """Tab 3: Train Stereo ML Suite / SpatiotemporalGNN or Evaluate Any Existing Model & Dataset."""

    _TIER_KEYS = ("fiducial_s80", "fiducial_all", "std_s80", "core_contained", "full_throw")

    training_status = Signal(int, str)
    model_trained = Signal(str, object)
    dataset_path_changed = Signal(str)
    model_path_changed = Signal(str)
    analysis_ready = Signal(dict)

    def __init__(self, array_model: DetectorArrayModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.array_model = array_model
        self.worker: Optional[TrainingWorker] = None
        self.eval_worker: Optional[EvaluateModelWorker] = None
        self.latest_bundle: Optional[Dict[str, Any]] = None
        self.viewing_benchmark: bool = False
        self.evaluation_params_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Event Reconstruction, Error Statistics & Model Evaluation")
        form = QFormLayout(grp)

        data_row = QWidget()
        dh = QHBoxLayout(data_row); dh.setContentsMargins(0, 0, 0, 0)
        self.edit_data = QLineEdit("data/corsika_crab_30deg/raw")
        self.edit_data.editingFinished.connect(lambda: self.dataset_path_changed.emit(self.edit_data.text().strip()))
        b_data = QPushButton("Browse...")
        b_data.clicked.connect(self._browse_data)
        dh.addWidget(self.edit_data, stretch=1); dh.addWidget(b_data)

        default_model = (
            "data/spatiotemporal_gnn_corsika.pt"
            if os.path.exists("data/spatiotemporal_gnn_corsika.pt")
            else "data/stereo_ml_bundle.pt"
        )
        model_row = QWidget()
        mh = QHBoxLayout(model_row); mh.setContentsMargins(0, 0, 0, 0)
        self.edit_model = QLineEdit(default_model)
        self.edit_model.editingFinished.connect(lambda: self.model_path_changed.emit(self.edit_model.text().strip()))
        self.btn_browse_load_model = QPushButton("Load .pt...")
        self.btn_browse_load_model.clicked.connect(self._browse_load_model)
        b_mod = QPushButton("Save As...")
        b_mod.clicked.connect(self._browse_model)
        mh.addWidget(self.edit_model, stretch=1)
        mh.addWidget(self.btn_browse_load_model)
        mh.addWidget(b_mod)

        self.combo_backend = QComboBox()
        self.combo_backend.addItems(
            [
                "Stereoscopic ML Suite + Stage-2 Calibrator (VERITAS / HESS / MAGIC)",
                "SpatiotemporalGNN (21-feat GAT + Halo + CaloSkip, VERITAS)",
            ]
        )
        self.combo_backend.currentIndexChanged.connect(self._on_backend_changed)

        self.spin_rf_est = QSpinBox(); self.spin_rf_est.setRange(10, 1000); self.spin_rf_est.setValue(120)
        self.spin_rf_depth = QSpinBox(); self.spin_rf_depth.setRange(3, 40); self.spin_rf_depth.setValue(14)
        self.spin_size2 = QDoubleSpinBox(); self.spin_size2.setRange(0.0, 2000.0); self.spin_size2.setValue(80.0); self.spin_size2.setSuffix(" p.e.")
        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(1, 200); self.spin_epochs.setValue(15)
        self.spin_batch = QSpinBox(); self.spin_batch.setRange(4, 512); self.spin_batch.setValue(64)
        self.spin_lr = QDoubleSpinBox(); self.spin_lr.setRange(1e-5, 0.1); self.spin_lr.setDecimals(5); self.spin_lr.setValue(0.0003)

        self.chk_debias_stage2 = QCheckBox("Stage-2 Power-Law Endpoint Debiaser (<225 GeV, >2 TeV)")
        self.chk_debias_stage2.setChecked(True)
        self.chk_debias_stage2.toggled.connect(self._on_benchmark_option_changed)

        self.chk_resume_sgd = QCheckBox("Continue SGD from existing Model Checkpoint (.pt)")
        self.chk_resume_sgd.setChecked(False)

        self.combo_cut_tier = QComboBox()
        self.combo_cut_tier.addItems([VERITAS_BENCHMARK_TIERS[k]["label"] for k in self._TIER_KEYS])
        self.combo_cut_tier.currentIndexChanged.connect(self._on_benchmark_option_changed)

        form.addRow("Dataset Dir:", data_row)
        form.addRow("Model Architecture:", self.combo_backend)
        form.addRow("Model Checkpoint (.pt):", model_row)
        form.addRow("RF Trees / Max Depth:", ArrayConfigTab._pair(self.spin_rf_est, self.spin_rf_depth))
        form.addRow("Validation Size_2nd Cut:", self.spin_size2)
        form.addRow("GNN Epochs / Batch Size:", ArrayConfigTab._pair(self.spin_epochs, self.spin_batch))
        form.addRow("GNN Learning Rate:", self.spin_lr)
        form.addRow("GNN SGD Mode:", self.chk_resume_sgd)
        form.addRow("Energy Calibrator:", self.chk_debias_stage2)
        form.addRow("VERITAS Cut Tier:", self.combo_cut_tier)
        left_v.addWidget(grp)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Train Reconstruction & Energy Model")
        self.btn_start.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 7px;")
        self.btn_start.clicked.connect(self.start_training)
        self.btn_stop = QPushButton("Stop Task")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_training)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        left_v.addLayout(btn_row)

        self.btn_eval_model = QPushButton("Evaluate Existing Model on Dataset (5-Fold CV + IRFs + Sky Map)")
        self.btn_eval_model.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 7px;")
        self.btn_eval_model.clicked.connect(self.evaluate_existing_model)
        left_v.addWidget(self.btn_eval_model)

        bench_row = QHBoxLayout()
        self.btn_load_gnn = QPushButton("Load Pre-Trained Upgraded GNN (27k CORSIKA Crab)")
        self.btn_load_gnn.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px;")
        self.btn_load_gnn.clicked.connect(self.load_pretrained_gnn_benchmark)
        self.btn_open_report = QPushButton("Benchmark Report PDF")
        self.btn_open_report.clicked.connect(self._open_benchmark_report)
        bench_row.addWidget(self.btn_load_gnn, stretch=2)
        bench_row.addWidget(self.btn_open_report, stretch=1)
        left_v.addLayout(bench_row)

        self.lbl_kpi = QLabel("No model trained yet in this session.")
        self.lbl_kpi.setWordWrap(True)
        self.lbl_kpi.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 8px; color: #e2e8f0;"
        )
        left_v.addWidget(self.lbl_kpi)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 8.5))
        left_v.addWidget(self.log_box, stretch=1)

        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(7.2, 7.2), facecolor="#0b1120")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_v.addWidget(self.toolbar)
        right_v.addWidget(self.canvas, stretch=1)
        self._init_empty_plots()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([520, 740])
        root.addWidget(splitter)

    def _init_empty_plots(self) -> None:
        self.fig.clear()
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)
        _style_dark_ax(ax1, "Loss / VERITAS Cut-Tier Comparison", "Epoch / Tier", "Loss / sigma_68 (%)")
        _style_dark_ax(ax2, "Energy Resolution (7-Bin Matched + 14-Bin Fine)", "True Energy (TeV)", "Resolution sigma_68 (%)")
        _style_dark_ax(ax3, "Signed Energy Bias (7-Bin Matched + 14-Bin Fine)", "True Energy (TeV)", "Signed Bias (%)")
        _style_dark_ax(ax4, "Calibrated E_reco vs E_true", "True Energy (GeV)", "Reco Energy (GeV)")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def set_array_model(self, model: DetectorArrayModel) -> None:
        self.array_model = model

    def set_dataset_path(self, path: str) -> None:
        self.edit_data.setText(path)

    def _selected_tier_key(self) -> str:
        idx = self.combo_cut_tier.currentIndex()
        if 0 <= idx < len(self._TIER_KEYS):
            return self._TIER_KEYS[idx]
        return "fiducial_s80"

    @Slot()
    def _on_benchmark_option_changed(self) -> None:
        if self.viewing_benchmark:
            self.load_pretrained_gnn_benchmark()

    @Slot()
    def load_pretrained_gnn_benchmark(self) -> None:
        tier = self._selected_tier_key()
        debias = bool(self.chk_debias_stage2.isChecked())
        res = load_upgraded_gnn_benchmark_evaluation(
            npz_path="data/eval_predictions_corsika_upgraded.npz",
            model_path="data/spatiotemporal_gnn_corsika.pt",
            tier=tier,
            debias_endpoints=debias,
        )
        self.viewing_benchmark = True
        self.latest_bundle = res["bundle"]
        m = res["metrics"]
        self.edit_model.setText(res["model_path"])
        self._on_metrics(m)
        self.training_status.emit(100, f"Upgraded GNN loaded ({tier})")
        self.model_trained.emit(res["model_path"], self.latest_bundle)
        b = m["binned"]
        band_str = ", ".join(
            f"{ec:.2f}TeV: {r * 100.0:.2f}% ({bi * 100.0:+.2f}%)"
            for ec, r, bi in zip(b["bin_centers_tev"], b["resolution_68"], b["bias"])
        )
        cv_folds = int(m.get("cv_folds", 1))
        cv_tag = f", {cv_folds}-Fold CV" if cv_folds > 1 else ""
        self.log_box.appendPlainText(
            f"[Benchmark] Loaded Upgraded SpatiotemporalGNN ({m['n_total']:,} Crab 30-deg events{cv_tag}, "
            f"Stage-2 debias={m['stage2_debiased']}, tier={m['tier']})\n"
            f"  Overall sigma_68={m['overall_res68'] * 100.0:.2f}% | Median bias={m['overall_bias'] * 100.0:+.2f}% | "
            f"ROC AUC={m['roc_auc']:.4f}\n"
            f"  7-Bin Matched: {band_str}\n"
            f"  vs Madhavan (2013): {format_baseline_comparison(b)}"
        )
        cached = self.latest_bundle.get("cached_results")
        if cached is not None:
            try:
                p = self.evaluation_params_provider() if self.evaluation_params_provider is not None else {}
                n_thrown = int(p.get("n_thrown_gammas", 0))
                if n_thrown <= 0:
                    for cand in ("data/corsika_crab_30deg/raw/sim_state.json", "data/corsika_crab_30deg/sim_state.json"):
                        if os.path.exists(cand):
                            with open(cand, "r", encoding="utf-8") as fh:
                                n_thrown = int(json.load(fh).get("gammas_thrown", 0))
                            break
                if n_thrown > 0:
                    n_thrown = max(
                        int(round(n_thrown * float(cached.get("val_fraction", 0.2)))),
                        int(np.sum(cached["true_class"] == 1.0)),
                    )
                analysis_out = compute_irf_and_spectrum_from_arrays(
                    results=cached,
                    n_thrown_gammas=n_thrown,
                    impact_radius_m=float(p.get("impact_radius_m", 750.0)),
                    e_min_gev=float(p.get("e_min_gev", 100.0)),
                    e_max_gev=float(p.get("e_max_gev", 30000.0)),
                    gen_index=float(p.get("gen_index", 2.0)),
                    gamma_cut=float(p.get("gamma_cut", 0.5)),
                    obs_time_s=float(p.get("obs_time_s", 18000.0)),
                    target_f0=float(p.get("target_f0", CRAB_F0_VERITAS)),
                    target_alpha=float(p.get("target_alpha", CRAB_INDEX_VERITAS)),
                    spectrum_mode=str(p.get("spectrum_mode", "crab_reweighted")),
                    theta_cut_deg=float(p.get("theta_cut_deg", 0.16)),
                    bg_method=str(p.get("bg_method", "ring")),
                    fov_half_width_deg=float(p.get("fov_half_width_deg", 1.8)),
                    bin_size_deg=float(p.get("bin_size_deg", 0.05)),
                )
                analysis_out["cached_inference"] = cached
                analysis_out["n_thrown_gammas"] = max(n_thrown, int(np.sum(cached["true_class"] == 1.0)))
                self.analysis_ready.emit(analysis_out)
            except Exception:
                pass

    @Slot()
    def _open_benchmark_report(self) -> None:
        for rel in (
            "reports/AirCherenkov_Energy_Resolution_Report.pdf",
            "reports/figures/energy_resolution_comparison.png",
            "reports/AirCherenkov_Energy_Resolution_Report.tex",
        ):
            if os.path.exists(rel):
                abs_p = os.path.abspath(rel)
                QDesktopServices.openUrl(QUrl.fromLocalFile(abs_p))
                self.log_box.appendPlainText(f"Opened benchmark artifact: {abs_p}")
                return
        self.log_box.appendPlainText(
            "Benchmark report PDF/PNG not found under reports/. Run report build first if needed."
        )

    @Slot(int)
    def _on_backend_changed(self, idx: int) -> None:
        if idx == 1 and "stereo_ml_bundle" in self.edit_model.text():
            self.edit_model.setText("data/spatiotemporal_gnn_gui.pt")
        elif idx == 0 and "spatiotemporal_gnn" in self.edit_model.text():
            self.edit_model.setText("data/stereo_ml_bundle.pt")

    @Slot()
    def _browse_data(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Simulation Dataset Directory", "data")
        if d:
            self.edit_data.setText(d)
            self.dataset_path_changed.emit(d)

    @Slot()
    def _browse_load_model(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Load Trained Model Checkpoint (.pt)", "data", "PyTorch (*.pt *.pth)"
        )
        if fpath:
            self.edit_model.setText(fpath)
            lower = os.path.basename(fpath).lower()
            if "stereo_ml" in lower or "bundle" in lower:
                self.combo_backend.blockSignals(True)
                self.combo_backend.setCurrentIndex(0)
                self.combo_backend.blockSignals(False)
            elif "gnn" in lower:
                self.combo_backend.blockSignals(True)
                self.combo_backend.setCurrentIndex(1)
                self.combo_backend.blockSignals(False)
            self.model_path_changed.emit(fpath)

    @Slot()
    def _browse_model(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(self, "Model Checkpoint Path", self.edit_model.text(), "PyTorch (*.pt)")
        if fpath:
            self.edit_model.setText(fpath)
            self.model_path_changed.emit(fpath)

    def build_params(self) -> Dict[str, Any]:
        backend = "gnn" if self.combo_backend.currentIndex() == 1 else "stereo_ml"
        return {
            "backend": backend,
            "data_root": self.edit_data.text().strip(),
            "model_path": self.edit_model.text().strip(),
            "rf_estimators": self.spin_rf_est.value(),
            "rf_max_depth": self.spin_rf_depth.value(),
            "size_2nd_cut": self.spin_size2.value(),
            "epochs": self.spin_epochs.value(),
            "batch_size": self.spin_batch.value(),
            "learning_rate": self.spin_lr.value(),
            "resume_training": bool(self.chk_resume_sgd.isChecked()),
            "debias_endpoints": bool(self.chk_debias_stage2.isChecked()),
            "tier": self._selected_tier_key(),
        }

    @Slot()
    def evaluate_existing_model(self) -> None:
        if (self.worker is not None and self.worker.isRunning()) or (
            self.eval_worker is not None and self.eval_worker.isRunning()
        ):
            return
        self.viewing_benchmark = False
        params = self.build_params()
        if self.evaluation_params_provider is not None:
            try:
                extra = self.evaluation_params_provider()
                if isinstance(extra, dict):
                    params.update(extra)
            except Exception:
                pass
        self.btn_start.setEnabled(False)
        self.btn_eval_model.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.log_box.appendPlainText(
            f"Evaluating model {params['model_path']} on dataset {params['data_root']}..."
        )
        self.eval_worker = EvaluateModelWorker(self.array_model, params, parent=self)
        self.eval_worker.progress.connect(self.training_status.emit)
        self.eval_worker.log.connect(self.log_box.appendPlainText)
        self.eval_worker.finished_ok.connect(self._on_eval_finished)
        self.eval_worker.failed.connect(self._on_eval_failed)
        self.eval_worker.start()

    @Slot(dict)
    def _on_eval_finished(self, payload: dict) -> None:
        self.btn_start.setEnabled(True)
        self.btn_eval_model.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.latest_bundle = payload.get("bundle")
        m = payload["metrics"]
        self._on_metrics(m)
        b = m.get("binned", {})
        if len(b.get("bin_centers_tev", [])) > 0:
            band_str = ", ".join(
                f"{ec:.2f}TeV: {r * 100.0:.2f}% ({bi * 100.0:+.2f}%)"
                for ec, r, bi in zip(b["bin_centers_tev"], b["resolution_68"], b["bias"])
            )
            cv_folds = int(m.get("cv_folds", 1))
            cv_tag = f"{cv_folds}-Fold CV, " if cv_folds > 1 else ""
            self.log_box.appendPlainText(
                f"[Eval] {cv_tag}N_total={m.get('n_total', m.get('n_val', 0)):,}, "
                f"N_eval={m.get('n_eval', 0):,} | sigma_68={m['overall_res68'] * 100.0:.2f}%, "
                f"Bias={m['overall_bias'] * 100.0:+.2f}%\n"
                f"  7-Bin Matched: {band_str}\n"
                f"  vs Madhavan (2013): {format_baseline_comparison(b)}"
            )
        self.model_trained.emit(payload["model_path"], self.latest_bundle)
        if "analysis_out" in payload and payload["analysis_out"]:
            self.analysis_ready.emit(payload["analysis_out"])

    @Slot(str)
    def _on_eval_failed(self, err: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_eval_model.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.training_status.emit(0, "Error")
        self.log_box.appendPlainText(f"ERROR: {err}")

    @Slot()
    def start_training(self) -> None:
        if (self.worker is not None and self.worker.isRunning()) or (
            self.eval_worker is not None and self.eval_worker.isRunning()
        ):
            return
        self.viewing_benchmark = False
        params = self.build_params()
        if (
            params["backend"] == "stereo_ml"
            and os.path.normpath(params["model_path"]) == os.path.normpath("data/spatiotemporal_gnn_corsika.pt")
        ):
            params["model_path"] = "data/stereo_ml_bundle.pt"
            self.edit_model.setText("data/stereo_ml_bundle.pt")
        self.btn_start.setEnabled(False)
        self.btn_eval_model.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_box.appendPlainText(f"Starting training ({params['backend']})...")
        self.worker = TrainingWorker(self.array_model, params, parent=self)
        self.worker.progress.connect(self.training_status.emit)
        self.worker.log.connect(self.log_box.appendPlainText)
        self.worker.epoch_metrics.connect(self._on_metrics)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    @Slot()
    def stop_training(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.log_box.appendPlainText("Stopping training gracefully...")

    @Slot(dict)
    def _on_metrics(self, m: dict) -> None:
        res_pct = m["overall_res68"] * 100.0
        auc = m.get("roc_auc", float("nan"))
        th68 = m.get("theta_68_deg", None)
        core_rms = m.get("core_rms_m", None)
        te = np.asarray(m.get("true_e_gev", []), dtype=np.float64)
        re = np.asarray(m.get("reco_e_gev", []), dtype=np.float64)
        overall_bias = m.get("overall_bias", None)
        if overall_bias is None and len(te) > 0:
            overall_bias = float(np.median((re - te) / np.clip(te, 1e-6, None)))

        cv_folds = int(m.get("cv_folds", 1))
        if m.get("backend") in ("gnn_eval", "stereo_ml_eval"):
            parts = [f"{cv_folds}-Fold CV"] if cv_folds > 1 else []
            parts.extend([
                f"Total={m.get('n_total', m['n_val'])}",
                f"Eval={m.get('n_eval', m['n_val'])}",
                f"sigma_68={res_pct:.2f}%",
            ])
        else:
            parts = ([f"{cv_folds}-Fold CV"] if cv_folds > 1 else [f"Train={m['n_train']}"]) + [
                f"Val={m['n_val']}",
                f"sigma_68={res_pct:.2f}%",
            ]
        if overall_bias is not None:
            parts.append(f"Bias={overall_bias * 100.0:+.2f}%")
        if "stage2_debiased" in m:
            parts.append(f"Stage-2={'ON' if m['stage2_debiased'] else 'OFF'}")
        parts.append(f"ROC AUC={auc:.3f}")
        if th68 is not None and core_rms is not None:
            parts.append(f"theta_68={th68:.3f} deg | Core RMS={core_rms:.1f} m")
        elif th68 is not None:
            parts.append(f"theta_68={th68:.3f} deg")
        if m.get("tier_label"):
            parts.append(f"Tier={m['tier_label']}")
        prov = describe_model_and_dataset(
            model_path=str(m.get("model_path", "") or self.edit_model.text().strip()),
            data_path=str(m.get("data_path", "") or self.edit_data.text().strip()),
            bundle={"backend": m.get("backend", ""), "source_backend": m.get("source_backend", "")},
            array_preset=self.array_model.preset_name,
        )
        self.lbl_kpi.setText(
            f"Model: {prov['model_title']}  ->  Dataset: {prov['dataset_title']}  |  Array: {self.array_model.preset_name}\n"
            + " | ".join(parts)
        )

        self.fig.clear()
        eval_tag = f"{cv_folds}-Fold CV" if cv_folds > 1 else "Validation Split"
        self.fig.suptitle(
            f"{prov['model_title']}  ->  {prov['dataset_title']}  ({eval_tag})",
            color="#e2e8f0",
            fontsize=10,
            fontweight="bold",
            y=0.992,
        )
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)

        if m.get("backend") == "gnn_benchmark":
            _style_dark_ax(ax1, "VERITAS Cut-Tier Comparison (sigma_68)", "Overall 68% Res (%)", "")
            short_names = {
                "fiducial_s80": "Fid + S2>=80",
                "fiducial_all": "Fid All",
                "std_s80": "Std + S2>=80",
                "core_contained": "Core R<=150",
                "full_throw": "Full 750m",
            }
            active_tier = m.get("tier", "fiducial_s80")
            keys = list(self._TIER_KEYS)
            vals = [
                res_pct if k == active_tier else float(VERITAS_BENCHMARK_TIERS[k]["overall_res68"]) * 100.0
                for k in keys
            ]
            labels = [short_names.get(k, k) for k in keys]
            colors = ["#34d399" if k == active_tier else "#475569" for k in keys]
            y_pos = np.arange(len(keys))
            bars = ax1.barh(y_pos, vals, color=colors, edgecolor="#94a3b8", linewidth=0.6)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels, color="#cbd5e1", fontsize=8)
            ax1.invert_yaxis()
            ax1.set_xlim(0.0, max(50.0, max(vals) * 1.18))
            for bar, val in zip(bars, vals):
                ax1.text(val + 0.8, bar.get_y() + bar.get_height() * 0.5, f"{val:.2f}%", va="center", color="#e2e8f0", fontsize=7.5)
        elif m.get("backend") in ("gnn_eval", "stereo_ml_eval"):
            _style_dark_ax(ax1, f"Gamma/Hadron ROC (AUC={auc:.3f})", "False Positive Rate", "True Positive Rate")
            ax1.plot(m["fpr"], m["tpr"], color="#22d3ee", lw=2, label="Evaluated Model")
            ax1.plot([0, 1], [0, 1], "w--", alpha=0.4)
            ax1.legend(fontsize=8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
        else:
            _style_dark_ax(ax1, "Loss Convergence", "Epoch", "Loss")
            ep = m.get("epochs", [1])
            ax1.plot(ep, m.get("train_losses", [0]), "o-", color="#38bdf8", label="Train")
            ax1.plot(ep, m.get("val_losses", [0]), "s-", color="#f59e0b", label="Validation")
            ax1.legend(fontsize=8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        _style_dark_ax(ax2, "Energy Resolution (7-Bin Matched + 14-Bin Fine)", "True Energy (TeV)", "Resolution sigma_68 (%)")
        b = m["binned"]
        model_short = m.get("model_arch", "Model")
        if len(model_short) > 26:
            model_short = "SpatiotemporalGNN" if "gnn" in model_short.lower() else model_short[:26]
        cal_suffix = "+S2" if m.get("stage2_debiased", False) else ""
        model_label = f"{model_short}{cal_suffix} (7-Bin Matched)"
        if len(b["bin_centers_tev"]) > 0:
            res_y = b["resolution_68"] * 100.0
            res_err = b.get("res_err", None)
            if res_err is not None and len(res_err) == len(res_y) and np.any(np.isfinite(res_err)):
                ax2.errorbar(
                    b["bin_centers_tev"],
                    res_y,
                    yerr=np.nan_to_num(res_err, nan=0.0) * 100.0,
                    fmt="o-",
                    color="#34d399",
                    lw=2.0,
                    capsize=3,
                    label=model_label,
                )
            else:
                ax2.plot(b["bin_centers_tev"], res_y, "o-", color="#34d399", lw=2.0, label=model_label)
        fine = b.get("fine", {})
        if len(fine.get("bin_centers_tev", [])) > 0:
            ax2.plot(
                fine["bin_centers_tev"],
                fine["resolution_68"] * 100.0,
                ".:",
                color="#a7f3d0",
                alpha=0.65,
                lw=1.2,
                markersize=4,
                label=f"{model_short}{cal_suffix} (14-Bin Fine)",
            )
        ax2.plot(VERITAS_BASELINE_E_TEV, VERITAS_BASELINE_RES_PCT, "s--", color="#f43f5e", lw=1.4, label="VERITAS Baseline (Madhavan 2013, Fig 4.18)")
        ax2.plot(CORE_CONTAINED_E_TEV, CORE_CONTAINED_RES_PCT, "^:", color="#f59e0b", lw=1.3, label="Core-Contained (R<=150m)")
        ax2.set_xscale("log")
        ymax = max([35.0, float(np.max(VERITAS_BASELINE_RES_PCT))] + ([float(np.nanmax(b["resolution_68"] * 100.0))] if len(b["resolution_68"]) > 0 else []))
        ax2.set_ylim(0.0, min(65.0, ymax * 1.15))
        ax2.legend(fontsize=6.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        _style_dark_ax(ax3, f"Signed Bias (7-Bin Matched, AUC={auc:.3f})", "True Energy (TeV)", "Signed Bias (%)")
        ax3.axhspan(-10.0, 10.0, color="#22c55e", alpha=0.12, label="±10% Target Band")
        ax3.axhline(0.0, color="#94a3b8", ls=":", lw=1.0)
        if len(b["bin_centers_tev"]) > 0:
            bias_y = b["bias"] * 100.0
            bias_err = b.get("bias_err", None)
            if bias_err is not None and len(bias_err) == len(bias_y) and np.any(np.isfinite(bias_err)):
                ax3.errorbar(
                    b["bin_centers_tev"],
                    bias_y,
                    yerr=np.nan_to_num(bias_err, nan=0.0) * 100.0,
                    fmt="o-",
                    color="#38bdf8",
                    lw=2.0,
                    capsize=3,
                    label=model_label,
                )
            else:
                ax3.plot(b["bin_centers_tev"], bias_y, "o-", color="#38bdf8", lw=2.0, label=model_label)
        if len(fine.get("bin_centers_tev", [])) > 0:
            ax3.plot(
                fine["bin_centers_tev"],
                fine["bias"] * 100.0,
                ".:",
                color="#bae6fd",
                alpha=0.65,
                lw=1.2,
                markersize=4,
                label=f"{model_short}{cal_suffix} (14-Bin Fine)",
            )
        ax3.plot(VERITAS_BASELINE_E_TEV, VERITAS_BASELINE_BIAS_PCT, "s--", color="#f43f5e", alpha=0.85, lw=1.4, label="VERITAS Baseline (Madhavan 2013, Fig 4.18)")
        ax3.set_xscale("log")
        ax3.set_ylim(-20.0, 20.0)
        ax3.legend(fontsize=6.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        _style_dark_ax(ax4, "Calibrated E_reco vs E_true", "True E (GeV)", "Reco E (GeV)")
        valid = np.isfinite(te) & np.isfinite(re) & (te > 1e-3) & (re > 1e-3)
        if np.any(valid):
            te_v, re_v = te[valid], re[valid]
            if len(te_v) > 2500:
                step = max(1, len(te_v) // 2500)
                te_p, re_p = te_v[::step], re_v[::step]
            else:
                te_p, re_p = te_v, re_v
            ax4.scatter(te_p, re_p, s=6, alpha=0.38, color="#38bdf8")
            lo = max(10.0, min(float(te_v.min()), float(re_v.min())))
            hi = max(float(te_v.max()), float(re_v.max()))
            ax4.plot([lo, hi], [lo, hi], "--", color="#f8fafc", lw=1.2, label="1:1")
            ax4.plot([lo, hi], [0.8 * lo, 0.8 * hi], ":", color="#94a3b8", lw=0.9, label="±20%")
            ax4.plot([lo, hi], [1.2 * lo, 1.2 * hi], ":", color="#94a3b8", lw=0.9)
            ax4.set_xscale("log")
            ax4.set_yscale("log")
            ax4.legend(fontsize=7, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        self.fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
        self.canvas.draw_idle()

    @Slot(dict)
    def _on_finished(self, payload: dict) -> None:
        self.btn_start.setEnabled(True)
        self.btn_eval_model.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.latest_bundle = payload.get("bundle")
        self.model_trained.emit(payload["model_path"], self.latest_bundle)

    @Slot(str)
    def _on_failed(self, err: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_eval_model.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.training_status.emit(0, "Error")
        self.log_box.appendPlainText(f"ERROR: {err}")


class IRFSpectralTab(QWidget):
    """Tab 4: Effective Area, Energy Migration Matrix, and Simulated vs Reconstructed Spectra."""

    analysis_status = Signal(int, str)
    analysis_completed = Signal(dict)
    dataset_path_changed = Signal(str)
    model_path_changed = Signal(str)

    def __init__(self, array_model: DetectorArrayModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.array_model = array_model
        self.worker: Optional[AnalysisIRFSpectralWorker] = None
        self.in_memory_bundle: Optional[Dict[str, Any]] = None
        self.last_output: Optional[Dict[str, Any]] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("IRF & Spectral Reconstruction Configuration")
        form = QFormLayout(grp)

        test_row = QWidget()
        th = QHBoxLayout(test_row); th.setContentsMargins(0, 0, 0, 0)
        self.edit_test = QLineEdit("data/corsika_crab_30deg/raw")
        self.edit_test.editingFinished.connect(lambda: self.dataset_path_changed.emit(self.edit_test.text().strip()))
        b_test = QPushButton("Browse...")
        b_test.clicked.connect(self._browse_test)
        th.addWidget(self.edit_test, stretch=1); th.addWidget(b_test)

        mod_row = QWidget()
        mh = QHBoxLayout(mod_row); mh.setContentsMargins(0, 0, 0, 0)
        default_irf_mod = (
            "data/spatiotemporal_gnn_corsika.pt"
            if not os.path.exists("data/stereo_ml_bundle.pt") and os.path.exists("data/spatiotemporal_gnn_corsika.pt")
            else "data/stereo_ml_bundle.pt"
        )
        self.edit_model = QLineEdit(default_irf_mod)
        self.edit_model.editingFinished.connect(lambda: self.model_path_changed.emit(self.edit_model.text().strip()))
        b_mod = QPushButton("Browse...")
        b_mod.clicked.connect(self._browse_model)
        mh.addWidget(self.edit_model, stretch=1); mh.addWidget(b_mod)

        self.chk_use_session = QCheckBox("Prefer in-memory trained / preloaded model if available")
        self.chk_use_session.setChecked(True)

        self.spin_thrown = QSpinBox(); self.spin_thrown.setRange(0, 10000000); self.spin_thrown.setValue(0)
        self.spin_thrown.setSpecialValueText("Auto (sim_state.json)")
        self.spin_imp = QDoubleSpinBox(); self.spin_imp.setRange(50.0, 3000.0); self.spin_imp.setValue(750.0); self.spin_imp.setSuffix(" m")
        self.spin_emin = QDoubleSpinBox(); self.spin_emin.setRange(10.0, 100000.0); self.spin_emin.setValue(100.0); self.spin_emin.setSuffix(" GeV")
        self.spin_emax = QDoubleSpinBox(); self.spin_emax.setRange(50.0, 500000.0); self.spin_emax.setValue(30000.0); self.spin_emax.setSuffix(" GeV")
        self.spin_gen_idx = QDoubleSpinBox(); self.spin_gen_idx.setRange(0.5, 5.0); self.spin_gen_idx.setValue(2.0); self.spin_gen_idx.setSingleStep(0.1)

        self.combo_spec = QComboBox()
        self.combo_spec.addItems(
            [
                "VERITAS Crab Nebula (reweighted)",
                "MAGIC Crab Nebula (reweighted)",
                "Custom Power-Law (reweighted)",
                "Raw MC Thrown Spectrum",
            ]
        )
        self.combo_spec.currentIndexChanged.connect(self._on_spec_preset_changed)

        self.spin_obs_hr = QDoubleSpinBox(); self.spin_obs_hr.setRange(0.01, 500.0); self.spin_obs_hr.setValue(5.0); self.spin_obs_hr.setSuffix(" hr")
        self.spin_f0 = QDoubleSpinBox(); self.spin_f0.setDecimals(10); self.spin_f0.setRange(1e-12, 1.0); self.spin_f0.setValue(CRAB_F0_VERITAS); self.spin_f0.setSingleStep(1e-8)
        self.spin_alpha = QDoubleSpinBox(); self.spin_alpha.setRange(1.0, 5.0); self.spin_alpha.setValue(CRAB_INDEX_VERITAS); self.spin_alpha.setSingleStep(0.05)

        cut_row = QWidget()
        ch = QHBoxLayout(cut_row); ch.setContentsMargins(0, 0, 0, 0)
        self.slider_cut = QSlider(Qt.Orientation.Horizontal)
        self.slider_cut.setRange(5, 95)
        self.slider_cut.setValue(50)
        self.slider_cut.valueChanged.connect(self._on_cut_changed)
        self.lbl_cut = QLabel("0.50")
        ch.addWidget(self.slider_cut, stretch=1); ch.addWidget(self.lbl_cut)

        form.addRow("Test Dataset Dir:", test_row)
        form.addRow("Model Checkpoint:", mod_row)
        form.addRow("", self.chk_use_session)
        form.addRow("Thrown Gammas / Impact R:", ArrayConfigTab._pair(self.spin_thrown, self.spin_imp))
        form.addRow("MC E_min / E_max:", ArrayConfigTab._pair(self.spin_emin, self.spin_emax))
        form.addRow("MC Gen Index / Obs Time:", ArrayConfigTab._pair(self.spin_gen_idx, self.spin_obs_hr))
        form.addRow("Source Spectrum Mode:", self.combo_spec)
        form.addRow("Target f0 / Index alpha:", ArrayConfigTab._pair(self.spin_f0, self.spin_alpha))
        form.addRow("Gamma Score Cut:", cut_row)
        left_v.addWidget(grp)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Compute IRFs & Spectra")
        self.btn_run.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold; padding: 7px;")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_save_irf = QPushButton("Save IRFs (.npz)")
        self.btn_save_irf.setEnabled(False)
        self.btn_save_irf.clicked.connect(self._save_irfs)
        self.btn_export = QPushButton("Export Plot")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_plot)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_save_irf)
        btn_row.addWidget(self.btn_export)
        left_v.addLayout(btn_row)

        self.lbl_summary = QLabel("Run analysis to generate Effective Area, Energy Migration, and Simulated vs. Reconstructed Spectra.")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setFont(QFont("Consolas", 9))
        self.lbl_summary.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 8px; color: #e2e8f0;"
        )
        left_v.addWidget(self.lbl_summary)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 8.5))
        left_v.addWidget(self.log_box, stretch=1)

        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8.0, 7.2), facecolor="#0b1120")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_v.addWidget(self.toolbar)
        right_v.addWidget(self.canvas, stretch=1)
        self._init_empty_plots()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([470, 790])
        root.addWidget(splitter)

    def _init_empty_plots(self) -> None:
        self.fig.clear()
        titles = [
            ("Effective Area A_eff(E_true)", "True Energy (TeV)", "Effective Area (m^2)"),
            ("Energy Migration P(E_reco | E_true)", "Reco Energy (TeV)", "True Energy (TeV)"),
            ("Forward-Folded Counts", "Reco Energy (TeV)", "Counts"),
            ("Simulated vs Reconstructed Energy Spectrum", "Energy (TeV)", "dN/dE (m^-2 s^-1 TeV^-1)"),
        ]
        for idx, (t, xl, yl) in enumerate(titles, start=1):
            ax = self.fig.add_subplot(2, 2, idx)
            _style_dark_ax(ax, t, xl, yl)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def set_array_model(self, model: DetectorArrayModel) -> None:
        self.array_model = model

    def set_test_data(self, path: str) -> None:
        self.edit_test.setText(path)

    def set_active_model(self, model_path: str, bundle: Optional[Dict[str, Any]]) -> None:
        self.edit_model.setText(model_path)
        self.in_memory_bundle = bundle

    @Slot(dict)
    def set_analysis_output(self, out: Dict[str, Any]) -> None:
        """Accept precomputed IRF/spectral/skymap results from unified evaluation or preloading."""
        if "aeff" in out and "fit" in out:
            self._on_finished(out)
        else:
            self.last_output = out
            if "gamma_cut" in out:
                val = max(5, min(95, int(round(float(out["gamma_cut"]) * 100.0))))
                self.slider_cut.blockSignals(True)
                self.slider_cut.setValue(val)
                self.lbl_cut.setText(f"{val / 100.0:.2f}")
                self.slider_cut.blockSignals(False)
            self.analysis_completed.emit(out)

    @Slot()
    def _browse_test(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Test Dataset Directory", "data")
        if d:
            self.edit_test.setText(d)
            self.dataset_path_changed.emit(d)

    @Slot()
    def _browse_model(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Select Trained Model (.pt)", "data", "PyTorch (*.pt *.pth)")
        if f:
            self.edit_model.setText(f)
            self.in_memory_bundle = None
            self.model_path_changed.emit(f)

    @Slot(int)
    def _on_spec_preset_changed(self, idx: int) -> None:
        if idx == 0:
            self.spin_f0.setValue(CRAB_F0_VERITAS)
            self.spin_alpha.setValue(CRAB_INDEX_VERITAS)
        elif idx == 1:
            self.spin_f0.setValue(CRAB_F0_MAGIC)
            self.spin_alpha.setValue(CRAB_INDEX_MAGIC)
        self._recompute_live_if_cached()

    @Slot(int)
    def _on_cut_changed(self, val: int) -> None:
        self.lbl_cut.setText(f"{val / 100.0:.2f}")
        self._recompute_live_if_cached()

    def _spectrum_mode(self) -> str:
        idx = self.combo_spec.currentIndex()
        if idx == 3:
            return "raw_mc"
        if idx == 2:
            return "custom_reweighted"
        return "crab_reweighted"

    def _recompute_live_if_cached(self) -> None:
        if self.last_output is None or "cached_inference" not in self.last_output:
            return
        try:
            updated = compute_irf_and_spectrum_from_arrays(
                results=self.last_output["cached_inference"],
                n_thrown_gammas=self.last_output["n_thrown_gammas"],
                impact_radius_m=self.spin_imp.value(),
                e_min_gev=self.spin_emin.value(),
                e_max_gev=self.spin_emax.value(),
                gen_index=self.spin_gen_idx.value(),
                gamma_cut=self.slider_cut.value() / 100.0,
                obs_time_s=self.spin_obs_hr.value() * 3600.0,
                target_f0=self.spin_f0.value(),
                target_alpha=self.spin_alpha.value(),
                spectrum_mode=self._spectrum_mode(),
            )
            updated["cached_inference"] = self.last_output["cached_inference"]
            updated["n_thrown_gammas"] = self.last_output["n_thrown_gammas"]
            for pkey in ("model_arch", "model_path", "model_title", "data_path", "dataset_title", "array_preset", "provenance_banner"):
                if pkey in self.last_output:
                    updated[pkey] = self.last_output[pkey]
                    if isinstance(updated.get("skymap"), dict):
                        updated["skymap"][pkey] = self.last_output[pkey]
            self._on_finished(updated)
        except Exception:
            pass

    def build_params(self) -> Dict[str, Any]:
        return {
            "test_data": self.edit_test.text().strip(),
            "model_path": self.edit_model.text().strip(),
            "n_thrown_gammas": self.spin_thrown.value(),
            "impact_radius_m": self.spin_imp.value(),
            "e_min_gev": self.spin_emin.value(),
            "e_max_gev": self.spin_emax.value(),
            "gen_index": self.spin_gen_idx.value(),
            "gamma_cut": self.slider_cut.value() / 100.0,
            "obs_time_s": self.spin_obs_hr.value() * 3600.0,
            "target_f0": self.spin_f0.value(),
            "target_alpha": self.spin_alpha.value(),
            "spectrum_mode": self._spectrum_mode(),
        }

    @Slot()
    def run_analysis(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        params = self.build_params()
        bundle = self.in_memory_bundle if self.chk_use_session.isChecked() else None
        self.btn_run.setEnabled(False)
        self.worker = AnalysisIRFSpectralWorker(self.array_model, params, in_memory_bundle=bundle, parent=self)
        self.worker.progress.connect(self.analysis_status.emit)
        self.worker.log.connect(self.log_box.appendPlainText)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    @Slot(dict)
    def _on_finished(self, out: dict) -> None:
        self.btn_run.setEnabled(True)
        self.btn_save_irf.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.last_output = out

        aeff = out["aeff"]
        mig = out["migration"]
        fit = out["fit"]
        peak_aeff = float(np.max(aeff["aeff"]))
        peak_e = float(aeff["bin_centres"][np.argmax(aeff["aeff"])]) / 1000.0
        sky_info = ""
        if "skymap" in out:
            sm = out["skymap"]
            sky_info = (
                f"\nSource Li & Ma S = {sm['source_significance_lima']:.2f} sigma "
                f"(N_on={sm['source_n_on']:.0f}, N_off={sm['source_n_off']:.0f}, alpha={sm['source_alpha']:.3f})"
            )

        prov = describe_model_and_dataset(
            model_path=str(out.get("model_path", "") or self.edit_model.text().strip()),
            data_path=str(out.get("data_path", "") or self.edit_test.text().strip()),
            bundle=self.in_memory_bundle,
            array_preset=self.array_model.preset_name,
        )
        self.lbl_summary.setText(
            f"Model: {prov['model_title']}  ->  Dataset: {prov['dataset_title']}\n"
            f"Array: {self.array_model.preset_name} | Selected Gammas: {out['n_gammas_selected']}/{out['n_gammas_total']} (Cut >= {out['gamma_cut']:.2f})\n"
            f"Peak Effective Area: {peak_aeff:,.0f} m^2 at {peak_e:.2f} TeV\n"
            f"Simulated Spectrum:  f0 = {out['sim_f0']:.3e},  alpha = {out['sim_alpha']:.3f}\n"
            f"Reconstructed Fit:   f0 = {fit['f0']:.3e} +/- {fit['f0_err']:.2e},  alpha = {fit['alpha']:.3f} +/- {fit['alpha_err']:.3f}"
            f"{sky_info}"
        )

        self.fig.clear()
        self.fig.suptitle(
            f"IRF & Spectral Reconstruction  |  {prov['model_title']}  ->  {prov['dataset_title']}",
            color="#e2e8f0",
            fontsize=10,
            fontweight="bold",
            y=0.992,
        )
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)

        bc_tev = out["bin_centres_tev"]
        be_tev = out["bin_edges_tev"]

        # 1. Effective Area
        _style_dark_ax(ax1, "Effective Area A_eff(E_true)", "True Energy (TeV)", "Effective Area (m^2)")
        pos_a = aeff["aeff"] > 0
        if np.any(pos_a):
            ax1.errorbar(
                bc_tev[pos_a],
                aeff["aeff"][pos_a],
                yerr=aeff["aeff_err"][pos_a],
                fmt="o-",
                color="#38bdf8",
                capsize=3,
                lw=2,
                label=f"{self.array_model.preset_name} A_eff",
            )
        ax1.axhline(1e5, color="#94a3b8", ls="--", alpha=0.6, label="VERITAS ~1e5 m^2 ref")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.legend(fontsize=8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        # 2. Migration Matrix
        _style_dark_ax(ax2, "Energy Migration P(E_reco | E_true)", "Reco Energy (TeV)", "True Energy (TeV)")
        im = ax2.pcolormesh(be_tev, be_tev, mig["matrix"], cmap="inferno", norm=LogNorm(vmin=1e-3, vmax=1.0))
        ax2.plot([be_tev[0], be_tev[-1]], [be_tev[0], be_tev[-1]], "w--", lw=1.3, alpha=0.75)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        cb = self.fig.colorbar(im, ax=ax2)
        cb.ax.tick_params(colors="#94a3b8", labelsize=7.5)

        # 3. Forward-Folded Counts
        _style_dark_ax(ax3, "Forward-Folded Count Spectrum", "Reconstructed Energy (TeV)", "Counts")
        n_obs = out["n_obs"]
        n_pred = fit["n_pred"]
        obs_mask = n_obs > 0
        if np.any(obs_mask):
            ax3.errorbar(
                bc_tev[obs_mask],
                n_obs[obs_mask],
                yerr=np.sqrt(np.maximum(out["raw_reco_counts"][obs_mask], 1.0))
                * (n_obs[obs_mask] / np.maximum(out["raw_reco_counts"][obs_mask], 1.0)),
                fmt="o",
                color="#f8fafc",
                capsize=3,
                label="Observed (Reco)",
            )
        ax3.step(be_tev, np.append(n_pred, n_pred[-1]), where="post", color="#f43f5e", lw=2, label="Forward-Fold Fit")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.legend(fontsize=8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        # 4. Simulated vs Reconstructed Differential Energy Spectrum
        _style_dark_ax(ax4, "Simulated vs Reconstructed Energy Spectrum", "Energy (TeV)", "dN/dE (m^-2 s^-1 TeV^-1)")
        e_curve = np.logspace(np.log10(be_tev[0]), np.log10(be_tev[-1]), 200)
        ax4.plot(
            e_curve,
            power_law_flux(e_curve, out["sim_f0"], out["sim_alpha"]),
            "--",
            color="#34d399",
            lw=2.0,
            label=f"Simulated Truth (alpha={out['sim_alpha']:.2f})",
        )
        vt = out["valid_true"]
        if np.any(vt):
            ax4.plot(
                bc_tev[vt],
                out["true_binned_flux"][vt],
                "s",
                color="#34d399",
                alpha=0.65,
                markersize=5,
                label="True Binned Flux",
            )
        vu = out["valid_unfold"]
        if np.any(vu):
            ax4.errorbar(
                bc_tev[vu],
                out["unfolded_flux"][vu],
                yerr=out["unfolded_err"][vu],
                fmt="o",
                color="#38bdf8",
                capsize=3,
                markersize=5,
                label="Unfolded Reco Flux",
            )
        ax4.plot(
            e_curve,
            power_law_flux(e_curve, fit["f0"], fit["alpha"]),
            "-",
            color="#f43f5e",
            lw=2.0,
            label=f"Reco Fit (alpha={fit['alpha']:.2f}+/-{fit['alpha_err']:.2f})",
        )
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.legend(fontsize=7.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        self.fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
        self.canvas.draw_idle()
        self.analysis_completed.emit(out)

    @Slot(str)
    def _on_failed(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.analysis_status.emit(0, "Error")
        self.log_box.appendPlainText(f"ERROR: {err}")

    @Slot()
    def _save_irfs(self) -> None:
        if self.last_output is None:
            return
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory for irfs.npz", "data")
        if d:
            path = save_irfs(self.last_output["aeff"], self.last_output["migration"], output_dir=d)
            self.log_box.appendPlainText(f"Saved IRFs to {path}")

    @Slot()
    def _export_plot(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(self, "Export IRF & Spectral Figure", "data/irf_spectral_summary.png", "PNG (*.png);;PDF (*.pdf)")
        if fpath:
            self.fig.savefig(fpath, dpi=220)
            self.log_box.appendPlainText(f"Exported plot to {fpath}")


class SkyMapSignificanceTab(QWidget):
    """Tab 5: 2D Event Count Maps, Excess Maps, Li & Ma (1983) Significance Maps & theta^2."""

    analysis_status = Signal(int, str)
    dataset_path_changed = Signal(str)
    model_path_changed = Signal(str)

    _BG_METHOD_KEYS = ("ring", "reflected", "mc_hadron")

    def __init__(self, array_model: DetectorArrayModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.array_model = array_model
        self.worker: Optional[AnalysisIRFSpectralWorker] = None
        self.in_memory_bundle: Optional[Dict[str, Any]] = None
        self.cached_inference: Optional[Dict[str, Any]] = None
        self.last_skymap: Optional[Dict[str, Any]] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Sky Map & Li & Ma Significance Configuration")
        form = QFormLayout(grp)

        test_row = QWidget()
        th = QHBoxLayout(test_row); th.setContentsMargins(0, 0, 0, 0)
        self.edit_test = QLineEdit("data/corsika_crab_30deg/raw")
        self.edit_test.editingFinished.connect(lambda: self.dataset_path_changed.emit(self.edit_test.text().strip()))
        b_test = QPushButton("Browse...")
        b_test.clicked.connect(self._browse_test)
        th.addWidget(self.edit_test, stretch=1); th.addWidget(b_test)

        mod_row = QWidget()
        mh = QHBoxLayout(mod_row); mh.setContentsMargins(0, 0, 0, 0)
        default_irf_mod = (
            "data/spatiotemporal_gnn_corsika.pt"
            if not os.path.exists("data/stereo_ml_bundle.pt") and os.path.exists("data/spatiotemporal_gnn_corsika.pt")
            else "data/stereo_ml_bundle.pt"
        )
        self.edit_model = QLineEdit(default_irf_mod)
        self.edit_model.editingFinished.connect(lambda: self.model_path_changed.emit(self.edit_model.text().strip()))
        b_mod = QPushButton("Browse...")
        b_mod.clicked.connect(self._browse_model)
        mh.addWidget(self.edit_model, stretch=1); mh.addWidget(b_mod)

        self.chk_use_session = QCheckBox("Prefer in-memory trained / preloaded model if available")
        self.chk_use_session.setChecked(True)

        cut_row = QWidget()
        ch = QHBoxLayout(cut_row); ch.setContentsMargins(0, 0, 0, 0)
        self.slider_cut = QSlider(Qt.Orientation.Horizontal)
        self.slider_cut.setRange(5, 95)
        self.slider_cut.setValue(50)
        self.slider_cut.valueChanged.connect(self._on_cut_changed)
        self.lbl_cut = QLabel("0.50")
        ch.addWidget(self.slider_cut, stretch=1); ch.addWidget(self.lbl_cut)

        self.spin_theta_cut = QDoubleSpinBox()
        self.spin_theta_cut.setRange(0.03, 0.60)
        self.spin_theta_cut.setValue(0.16)
        self.spin_theta_cut.setSingleStep(0.02)
        self.spin_theta_cut.setDecimals(2)
        self.spin_theta_cut.setSuffix(" deg")
        self.spin_theta_cut.valueChanged.connect(lambda _v: self._recompute_live_if_cached())

        self.combo_bg = QComboBox()
        self.combo_bg.addItems(
            [
                "Ring Background (Li & Ma)",
                "Reflected Wobble Regions (Li & Ma)",
                "MC Truth Hadron Template (Li & Ma)",
            ]
        )
        self.combo_bg.currentIndexChanged.connect(lambda _i: self._recompute_live_if_cached())

        self.combo_count_mode = QComboBox()
        self.combo_count_mode.addItems(
            [
                "Aperture-Integrated N_on Map",
                "Raw Unsmoothed Event Counts",
                "True Gamma Counts (Passing Cut)",
                "True Hadron Counts (Passing Cut)",
            ]
        )
        self.combo_count_mode.currentIndexChanged.connect(lambda _i: self._redraw_last())

        self.spin_fov = QDoubleSpinBox()
        self.spin_fov.setRange(0.6, 3.5)
        self.spin_fov.setValue(1.8)
        self.spin_fov.setSingleStep(0.2)
        self.spin_fov.setSuffix(" deg")
        self.spin_fov.valueChanged.connect(lambda _v: self._recompute_live_if_cached())

        self.spin_bin = QDoubleSpinBox()
        self.spin_bin.setRange(0.02, 0.20)
        self.spin_bin.setValue(0.05)
        self.spin_bin.setSingleStep(0.01)
        self.spin_bin.setDecimals(2)
        self.spin_bin.setSuffix(" deg")
        self.spin_bin.valueChanged.connect(lambda _v: self._recompute_live_if_cached())

        form.addRow("Test Dataset Dir:", test_row)
        form.addRow("Model Checkpoint:", mod_row)
        form.addRow("", self.chk_use_session)
        form.addRow("Gamma Score Cut:", cut_row)
        form.addRow("ON Aperture (theta_cut):", self.spin_theta_cut)
        form.addRow("Background Method:", self.combo_bg)
        form.addRow("Count Map Layer:", self.combo_count_mode)
        form.addRow("FoV Half-Width / Pixel:", ArrayConfigTab._pair(self.spin_fov, self.spin_bin))
        left_v.addWidget(grp)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Compute Sky Maps & Significance")
        self.btn_run.setStyleSheet("background-color: #0891b2; color: white; font-weight: bold; padding: 7px;")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_save = QPushButton("Save Maps (.npz)")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_maps)
        self.btn_export = QPushButton("Export Plot")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_plot)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_export)
        left_v.addLayout(btn_row)

        self.lbl_summary = QLabel(
            "Run analysis (or compute IRFs in Tab 4) to generate Event Count Maps and Li & Ma (1983) Significance Maps."
        )
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setFont(QFont("Consolas", 9))
        self.lbl_summary.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 8px; color: #e2e8f0;"
        )
        left_v.addWidget(self.lbl_summary)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 8.5))
        left_v.addWidget(self.log_box, stretch=1)

        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8.6, 7.6), facecolor="#0b1120")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_v.addWidget(self.toolbar)
        right_v.addWidget(self.canvas, stretch=1)
        self._init_empty_plots()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([470, 810])
        root.addWidget(splitter)

    def _init_empty_plots(self) -> None:
        self.fig.clear()
        titles = [
            ("2D Event Count Map", "Delta RA / Azimuth Offset (deg)", "Delta Dec / Zenith Offset (deg)"),
            ("2D Excess Gamma Counts (N_on - alpha*N_off)", "Delta RA / Azimuth Offset (deg)", "Delta Dec / Zenith Offset (deg)"),
            ("2D Li & Ma (1983) Significance Map (sigma)", "Delta RA / Azimuth Offset (deg)", "Delta Dec / Zenith Offset (deg)"),
            ("theta^2 Distribution (ON vs alpha*OFF)", "theta^2 (deg^2)", "Events"),
        ]
        for idx, (t, xl, yl) in enumerate(titles, start=1):
            ax = self.fig.add_subplot(2, 2, idx)
            _style_dark_ax(ax, t, xl, yl)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def set_array_model(self, model: DetectorArrayModel) -> None:
        self.array_model = model

    def set_test_data(self, path: str) -> None:
        self.edit_test.setText(path)

    def set_active_model(self, model_path: str, bundle: Optional[Dict[str, Any]]) -> None:
        self.edit_model.setText(model_path)
        self.in_memory_bundle = bundle

    def _bg_method_key(self) -> str:
        idx = self.combo_bg.currentIndex()
        if 0 <= idx < len(self._BG_METHOD_KEYS):
            return self._BG_METHOD_KEYS[idx]
        return "ring"

    @Slot(dict)
    def set_analysis_output(self, out: Dict[str, Any]) -> None:
        """Accept analysis outputs pushed from Tab 4 (IRFs & Spectra) or worker."""
        if "cached_inference" in out:
            self.cached_inference = out["cached_inference"]
        if "gamma_cut" in out:
            val = int(round(float(out["gamma_cut"]) * 100.0))
            val = max(5, min(95, val))
            if self.slider_cut.value() != val:
                self.slider_cut.blockSignals(True)
                self.slider_cut.setValue(val)
                self.lbl_cut.setText(f"{val / 100.0:.2f}")
                self.slider_cut.blockSignals(False)
        if self.cached_inference is not None:
            self._recompute_live_if_cached()
        elif "skymap" in out:
            self._render_skymap(out["skymap"])

    @Slot()
    def _browse_test(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Test Dataset Directory", "data")
        if d:
            self.edit_test.setText(d)
            self.dataset_path_changed.emit(d)

    @Slot()
    def _browse_model(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Select Trained Model (.pt)", "data", "PyTorch (*.pt *.pth)")
        if f:
            self.edit_model.setText(f)
            self.in_memory_bundle = None
            self.model_path_changed.emit(f)

    @Slot(int)
    def _on_cut_changed(self, val: int) -> None:
        self.lbl_cut.setText(f"{val / 100.0:.2f}")
        self._recompute_live_if_cached()

    def _recompute_live_if_cached(self) -> None:
        if self.cached_inference is None:
            return
        try:
            sm = compute_skymap_from_inference_arrays(
                results=self.cached_inference,
                gamma_cut=self.slider_cut.value() / 100.0,
                theta_cut_deg=self.spin_theta_cut.value(),
                bg_method=self._bg_method_key(),
                fov_half_width_deg=self.spin_fov.value(),
                bin_size_deg=self.spin_bin.value(),
            )
            if isinstance(self.last_skymap, dict):
                for pkey in ("model_arch", "model_path", "model_title", "data_path", "dataset_title", "array_preset", "provenance_banner"):
                    if pkey in self.last_skymap and pkey not in sm:
                        sm[pkey] = self.last_skymap[pkey]
            self._render_skymap(sm)
        except Exception:
            pass

    @Slot()
    def _redraw_last(self) -> None:
        if self.last_skymap is not None:
            self._render_skymap(self.last_skymap)

    @Slot()
    def run_analysis(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        params = {
            "test_data": self.edit_test.text().strip(),
            "model_path": self.edit_model.text().strip(),
            "gamma_cut": self.slider_cut.value() / 100.0,
            "theta_cut_deg": self.spin_theta_cut.value(),
            "bg_method": self._bg_method_key(),
            "fov_half_width_deg": self.spin_fov.value(),
            "bin_size_deg": self.spin_bin.value(),
            "skymap_only": True,
        }
        bundle = self.in_memory_bundle if self.chk_use_session.isChecked() else None
        self.btn_run.setEnabled(False)
        self.worker = AnalysisIRFSpectralWorker(self.array_model, params, in_memory_bundle=bundle, parent=self)
        self.worker.progress.connect(self.analysis_status.emit)
        self.worker.log.connect(self.log_box.appendPlainText)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    @Slot(dict)
    def _on_finished(self, out: dict) -> None:
        self.btn_run.setEnabled(True)
        if "cached_inference" in out:
            self.cached_inference = out["cached_inference"]
        if "skymap" in out:
            sm = dict(out["skymap"])
            for pkey in ("model_arch", "model_path", "model_title", "data_path", "dataset_title", "array_preset", "provenance_banner"):
                if pkey in out and pkey not in sm:
                    sm[pkey] = out[pkey]
            self._render_skymap(sm)

    @Slot(str)
    def _on_failed(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.analysis_status.emit(0, "Error")
        self.log_box.appendPlainText(f"ERROR: {err}")

    def _render_skymap(self, sm: Dict[str, Any]) -> None:
        self.last_skymap = sm
        self.btn_save.setEnabled(True)
        self.btn_export.setEnabled(True)

        bg_labels = {
            "ring": "Ring Background",
            "reflected": "Reflected Wobble",
            "mc_hadron": "MC Hadron Template",
        }
        bg_label = bg_labels.get(sm["bg_method"], sm["bg_method"])
        est_note = " [Nominal PSF/Diffuse Dir Fallback]" if sm.get("directions_estimated") else ""
        s_to_b = sm["source_n_excess"] / max(sm["source_n_bkg"], 1e-6) if sm["source_n_bkg"] > 0 else float("inf")
        s_to_b_str = f"{s_to_b:.2f}" if np.isfinite(s_to_b) else "inf"

        prov = describe_model_and_dataset(
            model_path=str(sm.get("model_path", "") or self.edit_model.text().strip()),
            data_path=str(sm.get("data_path", "") or self.edit_test.text().strip()),
            bundle=self.in_memory_bundle,
            array_preset=self.array_model.preset_name,
        )
        self.lbl_summary.setText(
            f"Model: {prov['model_title']}  ->  Dataset: {prov['dataset_title']}\n"
            f"Array: {self.array_model.preset_name} | {bg_label} | Passing Cut: {sm['n_events_passing']} "
            f"(G={sm['n_gammas_passing']}, H={sm['n_hadrons_passing']}){est_note}\n"
            f"ON Aperture (<= {sm['theta_cut_deg']:.2f} deg): N_on = {sm['source_n_on']:.0f} "
            f"(G={sm['n_gammas_in_theta_cut']}, H={sm['n_hadrons_in_theta_cut']}) | "
            f"N_off = {sm['source_n_off']:.0f}, alpha = {sm['source_alpha']:.4f}\n"
            f"Est. Background = {sm['source_n_bkg']:.2f} | Excess = {sm['source_n_excess']:+.1f} | S/B = {s_to_b_str}\n"
            f"Li & Ma Significance (Source): {sm['source_significance_lima']:.2f} sigma  |  "
            f"2D Map Peak: {sm['peak_significance_lima']:.2f} sigma"
        )

        edges = sm["x_edges_deg"]
        extent = [float(edges[0]), float(edges[-1]), float(edges[0]), float(edges[-1])]

        count_mode = self.combo_count_mode.currentIndex()
        if count_mode == 1:
            cnt_map = sm["raw_counts_map"]
            cnt_title = f"Raw Event Count Map (Total={sm['n_events_passing']})"
        elif count_mode == 2:
            cnt_map = sm["raw_gamma_map"]
            cnt_title = f"True Gamma Count Map (N_gamma={sm['n_gammas_passing']})"
        elif count_mode == 3:
            cnt_map = sm["raw_hadron_map"]
            cnt_title = f"True Hadron Count Map (N_had={sm['n_hadrons_passing']})"
        else:
            cnt_map = sm["n_on_map"]
            cnt_title = f"Aperture-Integrated Event Count Map N_on (r<={sm['theta_cut_deg']:.2f} deg)"

        self.fig.clear()
        self.fig.suptitle(
            f"Sky Map & Li & Ma Significance  |  {prov['model_title']}  ->  {prov['dataset_title']}",
            color="#e2e8f0",
            fontsize=10,
            fontweight="bold",
            y=0.992,
        )
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)

        # 1. Event Count Map
        _style_dark_ax(ax1, cnt_title, "Delta Azimuth / Sky X (deg)", "Delta Zenith / Sky Y (deg)")
        im1 = ax1.imshow(cnt_map, origin="lower", extent=extent, cmap="viridis", aspect="equal")
        ax1.add_patch(Circle((0.0, 0.0), sm["theta_cut_deg"], fill=False, edgecolor="#38bdf8", ls="--", lw=1.4))
        if sm["bg_method"] == "ring":
            ax1.add_patch(Circle((0.0, 0.0), sm["ring_r_in_deg"], fill=False, edgecolor="#94a3b8", ls=":", lw=1.0))
            ax1.add_patch(Circle((0.0, 0.0), sm["ring_r_out_deg"], fill=False, edgecolor="#94a3b8", ls=":", lw=1.0))
        ax1.plot(0.0, 0.0, "+", color="#f8fafc", markersize=8, mew=1.5)
        cb1 = self.fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.set_label("Counts", color="#cbd5e1", fontsize=8)
        cb1.ax.tick_params(colors="#94a3b8", labelsize=7.5)

        # 2. Excess Gamma Count Map
        ex_map = sm["excess_map"]
        _style_dark_ax(ax2, "2D Excess Count Map (N_on - alpha * N_off)", "Delta Azimuth / Sky X (deg)", "Delta Zenith / Sky Y (deg)")
        ex_min, ex_max = float(np.min(ex_map)), float(np.max(ex_map))
        if ex_min < 0.0 < ex_max:
            norm2 = TwoSlopeNorm(vmin=ex_min, vcenter=0.0, vmax=ex_max)
            im2 = ax2.imshow(ex_map, origin="lower", extent=extent, cmap="RdBu_r", norm=norm2, aspect="equal")
        else:
            im2 = ax2.imshow(ex_map, origin="lower", extent=extent, cmap="RdBu_r", aspect="equal")
        ax2.add_patch(Circle((0.0, 0.0), sm["theta_cut_deg"], fill=False, edgecolor="#22d3ee", ls="--", lw=1.3))
        ax2.plot(0.0, 0.0, "+", color="#f8fafc", markersize=8, mew=1.5)
        cb2 = self.fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label("Excess Counts", color="#cbd5e1", fontsize=8)
        cb2.ax.tick_params(colors="#94a3b8", labelsize=7.5)

        # 3. 2D Li & Ma Significance Map
        sig_map = sm["significance_map"]
        _style_dark_ax(
            ax3,
            f"Li & Ma Significance Map (Src: {sm['source_significance_lima']:.2f} sigma)",
            "Delta Azimuth / Sky X (deg)",
            "Delta Zenith / Sky Y (deg)",
        )
        sig_lo = min(-2.5, float(np.min(sig_map)))
        sig_hi = max(5.5, float(np.max(sig_map)))
        im3 = ax3.imshow(sig_map, origin="lower", extent=extent, cmap="inferno", vmin=sig_lo, vmax=sig_hi, aspect="equal")
        centres = sm["bin_centres_deg"]
        for lvl, col, lstyle in ((3.0, "#38bdf8", "--"), (5.0, "#34d399", "-")):
            if float(np.max(sig_map)) > lvl > float(np.min(sig_map)):
                ax3.contour(centres, centres, sig_map, levels=[lvl], colors=[col], linestyles=[lstyle], linewidths=[1.3])
        ax3.plot(0.0, 0.0, "+", color="#f8fafc", markersize=8, mew=1.5)
        cb3 = self.fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cb3.set_label("Significance (sigma)", color="#cbd5e1", fontsize=8)
        cb3.ax.tick_params(colors="#94a3b8", labelsize=7.5)

        # 4. theta^2 Distribution (ON vs alpha * OFF)
        _style_dark_ax(ax4, "theta^2 Distribution (ON vs alpha * OFF)", "theta^2 (deg^2)", "Counts / Bin")
        t2_edges = sm["theta2_edges_deg2"]
        t2_c = sm["theta2_centres_deg2"]
        t2_on = sm["theta2_on_counts"]
        t2_off = sm["theta2_off_scaled_counts"]
        ax4.step(t2_edges, np.append(t2_on, t2_on[-1]), where="post", color="#38bdf8", lw=2.0, label="ON Events")
        ax4.errorbar(t2_c, t2_on, yerr=np.sqrt(np.maximum(t2_on, 0.0)), fmt="none", ecolor="#38bdf8", elinewidth=1.0, capsize=2)
        ax4.step(t2_edges, np.append(t2_off, t2_off[-1]), where="post", color="#f59e0b", ls="--", lw=1.8, label="alpha * OFF Bkg")
        ax4.fill_between(t2_edges, 0.0, np.append(t2_off, t2_off[-1]), step="post", color="#f59e0b", alpha=0.18)
        ax4.axvline(sm["theta2_cut_deg2"], color="#34d399", ls=":", lw=1.8, label=f"theta^2 cut = {sm['theta2_cut_deg2']:.3f}")
        ax4.set_xlim(0.0, float(t2_edges[-1]))
        ax4.set_ylim(bottom=0.0)
        ax4.legend(fontsize=7.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        self.fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
        self.canvas.draw_idle()

    @Slot()
    def _save_maps(self) -> None:
        if self.last_skymap is None:
            return
        fpath, _ = QFileDialog.getSaveFileName(self, "Save Sky Maps & Significance (.npz)", "data/skymap_significance.npz", "NumPy Archive (*.npz)")
        if fpath:
            sm = self.last_skymap
            np.savez(
                fpath,
                x_edges_deg=sm["x_edges_deg"],
                y_edges_deg=sm["y_edges_deg"],
                raw_counts_map=sm["raw_counts_map"],
                raw_gamma_map=sm["raw_gamma_map"],
                raw_hadron_map=sm["raw_hadron_map"],
                n_on_map=sm["n_on_map"],
                n_off_map=sm["n_off_map"],
                alpha_map=sm["alpha_map"],
                bkg_map=sm["bkg_map"],
                excess_map=sm["excess_map"],
                significance_map=sm["significance_map"],
                source_n_on=np.array([sm["source_n_on"]]),
                source_n_off=np.array([sm["source_n_off"]]),
                source_alpha=np.array([sm["source_alpha"]]),
                source_significance_lima=np.array([sm["source_significance_lima"]]),
            )
            self.log_box.appendPlainText(f"Saved sky maps and Li & Ma significance to {fpath}")

    @Slot()
    def _export_plot(self) -> None:
        fpath, _ = QFileDialog.getSaveFileName(self, "Export Sky Map Figure", "data/skymap_significance.png", "PNG (*.png);;PDF (*.pdf)")
        if fpath:
            self.fig.savefig(fpath, dpi=220)
            self.log_box.appendPlainText(f"Exported plot to {fpath}")
