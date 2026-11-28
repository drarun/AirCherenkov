"""Main standalone desktop window for AirCherenkov Studio."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Optional

from PySide6.QtCore import Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.array_models import DetectorArrayModel, get_preset_model
from gui.event_viewer import EventViewerWidget
from gui.tabs import (
    ArrayConfigTab,
    IRFSpectralTab,
    SimulationTab,
    SkyMapSignificanceTab,
    TrainingTab,
)
from gui.telemetry import SystemAndPipelineStatusBar
from gui.workers import describe_model_and_dataset


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0b1120;
    color: #e2e8f0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12px;
}
QTabWidget::pane {
    border: 1px solid #1e293b;
    background-color: #0b1120;
}
QTabBar::tab {
    background-color: #111827;
    color: #94a3b8;
    padding: 8px 16px;
    border: 1px solid #1e293b;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #1e293b;
    color: #38bdf8;
    border-color: #334155;
}
QGroupBox {
    border: 1px solid #1e293b;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 600;
    color: #38bdf8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTableWidget {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    color: #f1f5f9;
    padding: 3px 6px;
}
QHeaderView::section {
    background-color: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
    padding: 4px;
    font-weight: 600;
}
QPushButton {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    color: #f1f5f9;
    padding: 5px 10px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #334155;
}
QPushButton:disabled {
    background-color: #0f172a;
    color: #475569;
    border-color: #1e293b;
}
"""


class AirCherenkovMainWindow(QMainWindow):
    """Unified standalone desktop application for AirCherenkov."""

    def __init__(
        self,
        initial_preset: str = "VERITAS",
        preload_events: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("AirCherenkov Studio — Native IACT Simulation, Reconstruction & Analysis")
        self.resize(1460, 920)
        self.setStyleSheet(DARK_STYLESHEET)

        self.array_model: DetectorArrayModel = get_preset_model(initial_preset)
        self._active_bundle: Optional[Dict[str, Any]] = None
        self._last_viewer_loaded_path: Optional[str] = None

        central = QWidget()
        v_layout = QVBoxLayout(central)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.array_tab = ArrayConfigTab(self.array_model)
        self.sim_tab = SimulationTab(self.array_model)
        self.train_tab = TrainingTab(self.array_model)
        self.irf_tab = IRFSpectralTab(self.array_model)
        self.skymap_tab = SkyMapSignificanceTab(self.array_model)
        self.viewer_tab = EventViewerWidget(self.array_model)

        self.train_tab.evaluation_params_provider = self._collect_eval_params

        self.tabs.addTab(self.array_tab, "1. Array & Detector")
        self.tabs.addTab(self.sim_tab, "2. Simulation")
        self.tabs.addTab(self.train_tab, "3. Training & Models")
        self.tabs.addTab(self.irf_tab, "4. IRFs & Spectra")
        self.tabs.addTab(self.skymap_tab, "5. Sky Maps & Significance")
        self.tabs.addTab(self.viewer_tab, "6. Event Viewer")

        self.quick_load_bar = self._build_quick_load_bar()
        self.status_bar_widget = SystemAndPipelineStatusBar()

        v_layout.addWidget(self.quick_load_bar)
        v_layout.addWidget(self.tabs, stretch=1)
        v_layout.addWidget(self.status_bar_widget)
        self.setCentralWidget(central)

        self._wire_signals()
        self._update_active_context_banner()
        if preload_events:
            self._preload_default_events_if_present()

    def _build_quick_load_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background-color: #0f172a; border-bottom: 1px solid #1e293b;")
        vl = QVBoxLayout(bar)
        vl.setContentsMargins(10, 6, 10, 6)
        vl.setSpacing(4)

        hl = QHBoxLayout()
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)

        lbl_d = QLabel("Active Dataset:")
        lbl_d.setStyleSheet("font-weight: 600; color: #38bdf8; border: none;")
        self.edit_global_data = QLineEdit(self.train_tab.edit_data.text())
        self.edit_global_data.setMinimumWidth(220)
        self.edit_global_data.editingFinished.connect(
            lambda: self.set_active_dataset_path(self.edit_global_data.text().strip())
        )
        self.edit_global_data.textChanged.connect(lambda _txt: self._update_active_context_banner())
        self.btn_global_data = QPushButton("Browse Dataset...")
        self.btn_global_data.clicked.connect(self._browse_global_dataset)

        lbl_m = QLabel("Trained Model (.pt):")
        lbl_m.setStyleSheet("font-weight: 600; color: #38bdf8; border: none;")
        self.edit_global_model = QLineEdit(self.train_tab.edit_model.text())
        self.edit_global_model.setMinimumWidth(220)
        self.edit_global_model.editingFinished.connect(
            lambda: self.set_active_model_path(self.edit_global_model.text().strip())
        )
        self.edit_global_model.textChanged.connect(lambda _txt: self._update_active_context_banner())
        self.btn_global_model = QPushButton("Load Model (.pt)...")
        self.btn_global_model.clicked.connect(self._browse_global_model)

        self.btn_eval_all = QPushButton("Evaluate Dataset & Model (Stats + IRFs + Sky Map)")
        self.btn_eval_all.setStyleSheet(
            "background-color: #0284c7; color: white; font-weight: bold; padding: 5px 12px;"
        )

        hl.addWidget(lbl_d)
        hl.addWidget(self.edit_global_data, stretch=1)
        hl.addWidget(self.btn_global_data)
        hl.addWidget(lbl_m)
        hl.addWidget(self.edit_global_model, stretch=1)
        hl.addWidget(self.btn_global_model)
        hl.addWidget(self.btn_eval_all)

        self.lbl_active_context = QLabel("")
        self.lbl_active_context.setWordWrap(True)
        self.lbl_active_context.setStyleSheet(
            "color: #38bdf8; font-weight: 600; font-size: 11px; border: none; padding-top: 1px;"
        )

        vl.addLayout(hl)
        vl.addWidget(self.lbl_active_context)
        return bar

    def _update_active_context_banner(self, bundle: Optional[Dict[str, Any]] = None) -> None:
        if bundle is not None:
            self._active_bundle = bundle
        active_bundle = self._active_bundle
        data_txt = self.edit_global_data.text().strip()
        model_txt = self.edit_global_model.text().strip()
        prov = describe_model_and_dataset(
            model_path=model_txt,
            data_path=data_txt,
            bundle=active_bundle,
            array_preset=self.array_model.preset_name,
        )
        eval_tag = ""
        if isinstance(active_bundle, dict):
            metrics = active_bundle.get("metrics") or {}
            cv_folds = int(metrics.get("cv_folds", 1))
            n_val = int(metrics.get("n_val", 0))
            if cv_folds > 1 and n_val > 0:
                eval_tag = f"  |  Eval: {cv_folds}-Fold CV ({n_val:,} events)"
        self.lbl_active_context.setText(f"Active Reconstruction:  {prov['provenance_banner']}{eval_tag}")

    def _collect_eval_params(self) -> dict:
        p = self.irf_tab.build_params()
        p.pop("test_data", None)
        p.pop("model_path", None)
        p.update(
            {
                "theta_cut_deg": self.skymap_tab.spin_theta_cut.value(),
                "bg_method": self.skymap_tab._bg_method_key(),
                "fov_half_width_deg": self.skymap_tab.spin_fov.value(),
                "bin_size_deg": self.skymap_tab.spin_bin.value(),
            }
        )
        return p

    @Slot()
    def _browse_global_dataset(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Simulation Dataset Directory", "data")
        if d:
            self.set_active_dataset_path(d)

    @Slot()
    def _browse_global_model(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "Load Trained Model (.pt)", "data", "PyTorch (*.pt *.pth)"
        )
        if f:
            self.set_active_model_path(f)

    @Slot(str)
    def set_active_dataset_path(self, path: str) -> None:
        clean = str(path or "").strip()
        if not clean:
            return
        if self.edit_global_data.text() != clean:
            self.edit_global_data.setText(clean)
        if self.train_tab.edit_data.text() != clean:
            self.train_tab.set_dataset_path(clean)
        if self.irf_tab.edit_test.text() != clean:
            self.irf_tab.set_test_data(clean)
        if self.skymap_tab.edit_test.text() != clean:
            self.skymap_tab.set_test_data(clean)
        self.sim_tab.set_monitored_directory(clean)
        try:
            target_to_load: Optional[str] = None
            if os.path.isdir(clean):
                pt_files = sorted(glob.glob(os.path.join(clean, "sim_batch_*.pt")))
                if not pt_files:
                    pt_files = sorted(glob.glob(os.path.join(clean, "raw", "sim_batch_*.pt")))
                if pt_files:
                    target_to_load = pt_files[0]
            elif os.path.isfile(clean) and any(
                clean.lower().endswith(ext) for ext in (".pt", ".h5", ".hdf5", ".npz", ".vbf", ".root")
            ):
                target_to_load = clean
            if target_to_load:
                abs_target = os.path.abspath(target_to_load)
                if abs_target != self._last_viewer_loaded_path:
                    self.viewer_tab.load_path(target_to_load)
                    self._last_viewer_loaded_path = abs_target
        except Exception:
            pass
        self._update_active_context_banner()

    @Slot(str)
    def set_active_model_path(self, model_path: str) -> None:
        clean = str(model_path or "").strip()
        if not clean:
            return
        for tab in (self.irf_tab, self.skymap_tab):
            bundle = getattr(tab, "in_memory_bundle", None)
            if isinstance(bundle, dict) and bundle.get("model_path"):
                if os.path.normcase(os.path.abspath(str(bundle["model_path"]))) != os.path.normcase(
                    os.path.abspath(clean)
                ):
                    tab.in_memory_bundle = None
        if isinstance(self._active_bundle, dict) and self._active_bundle.get("model_path"):
            if os.path.normcase(os.path.abspath(str(self._active_bundle["model_path"]))) != os.path.normcase(
                os.path.abspath(clean)
            ):
                self._active_bundle = None
        if self.edit_global_model.text() != clean:
            self.edit_global_model.setText(clean)
        if self.train_tab.edit_model.text() != clean:
            self.train_tab.edit_model.setText(clean)
        if self.irf_tab.edit_model.text() != clean:
            self.irf_tab.edit_model.setText(clean)
        if self.skymap_tab.edit_model.text() != clean:
            self.skymap_tab.edit_model.setText(clean)
        self.viewer_tab.set_active_model(clean, self._active_bundle)
        self._update_active_context_banner()

    def _wire_signals(self) -> None:
        self.array_tab.array_model_changed.connect(self._on_array_model_changed)
        self.array_tab.raw_data_imported.connect(self._on_raw_data_imported)

        self.sim_tab.simulation_status.connect(self.status_bar_widget.set_sim_status)
        self.sim_tab.dataset_ready.connect(self._on_simulation_dataset_ready)
        self.sim_tab.open_in_viewer_requested.connect(self._on_open_in_viewer_requested)

        self.train_tab.training_status.connect(self.status_bar_widget.set_train_status)
        self.train_tab.model_trained.connect(self._on_model_trained)
        self.train_tab.analysis_ready.connect(self.irf_tab.set_analysis_output)
        self.train_tab.dataset_path_changed.connect(self.set_active_dataset_path)
        self.train_tab.model_path_changed.connect(self.set_active_model_path)

        self.irf_tab.analysis_status.connect(self.status_bar_widget.set_analysis_status)
        self.irf_tab.analysis_completed.connect(self.skymap_tab.set_analysis_output)
        self.irf_tab.dataset_path_changed.connect(self.set_active_dataset_path)
        self.irf_tab.model_path_changed.connect(self.set_active_model_path)

        self.skymap_tab.analysis_status.connect(self.status_bar_widget.set_analysis_status)
        self.skymap_tab.dataset_path_changed.connect(self.set_active_dataset_path)
        self.skymap_tab.model_path_changed.connect(self.set_active_model_path)

        self.btn_eval_all.clicked.connect(self.train_tab.evaluate_existing_model)

    def _preload_default_events_if_present(self) -> None:
        for candidate in ("data/corsika_crab_30deg/raw", "data/stereo_eval_v2"):
            if os.path.isdir(candidate):
                try:
                    pt_files = sorted(glob.glob(os.path.join(candidate, "sim_batch_*.pt")))
                    if pt_files:
                        self.set_active_dataset_path(candidate)
                        break
                except Exception:
                    pass

        if (
            os.path.exists("data/eval_predictions_corsika_upgraded.npz")
            and os.path.exists("data/corsika_crab_30deg/processed/data.pt")
        ):
            try:
                self.train_tab.load_pretrained_gnn_benchmark()
            except Exception:
                pass

    @Slot(object)
    def _on_array_model_changed(self, model: DetectorArrayModel) -> None:
        self.array_model = model
        self.sim_tab.set_array_model(model)
        self.train_tab.set_array_model(model)
        self.irf_tab.set_array_model(model)
        self.skymap_tab.set_array_model(model)
        self.viewer_tab.set_array_model(model)
        self._update_active_context_banner()

    @Slot(str)
    def _on_raw_data_imported(self, path: str) -> None:
        try:
            n_loaded = self.viewer_tab.load_path(path)
            self.tabs.setCurrentWidget(self.viewer_tab)
            self.status_bar_widget.set_analysis_status(100, f"Loaded {n_loaded} raw events")
            self._update_active_context_banner()
        except Exception as exc:
            self.status_bar_widget.set_analysis_status(0, f"Import error: {exc}")

    @Slot(str)
    def _on_simulation_dataset_ready(self, output_dir: str) -> None:
        if os.path.exists(output_dir):
            self.set_active_dataset_path(output_dir)

    @Slot(str)
    def _on_open_in_viewer_requested(self, output_dir: str) -> None:
        if os.path.exists(output_dir):
            self.viewer_tab.load_path(output_dir)
            self.tabs.setCurrentWidget(self.viewer_tab)

    @Slot(str, object)
    def _on_model_trained(self, model_path: str, bundle) -> None:
        self._active_bundle = bundle if isinstance(bundle, dict) else None
        if self.edit_global_model.text() != model_path:
            self.edit_global_model.setText(model_path)
        self.irf_tab.set_active_model(model_path, bundle)
        self.skymap_tab.set_active_model(model_path, bundle)
        self.viewer_tab.set_active_model(model_path, bundle)
        self._update_active_context_banner(self._active_bundle)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.sim_tab.shutdown()
        for tab in (self.sim_tab, self.train_tab, self.irf_tab, self.skymap_tab):
            for attr in ("worker", "eval_worker"):
                w = getattr(tab, attr, None)
                if w is not None and w.isRunning():
                    if hasattr(w, "request_stop"):
                        w.request_stop()
                    w.wait(2500)
        self.status_bar_widget.shutdown()
        super().closeEvent(event)
