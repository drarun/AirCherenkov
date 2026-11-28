"""Native PySide6 Cherenkov Event Viewer with hexagonal cameras and FADC inspector."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from analysis.io import load_events_universal
from gui.array_models import DetectorArrayModel, extract_stereo_features_any_array
from gui.workers import describe_model_and_dataset
from recon.cleaning import double_pass_fadc_clean
from recon.hillas import HillasParameters, compute_hillas
from recon.stereo import reconstruct_event_geometric, tailcut_clean
from sim.camera import Camera
from sim.fadc import restore_low_gain_traces


# 9-stop Viridis LUT interpolated to 256 RGB colors for fast camera updates
_VIRIDIS_STOPS = np.array(
    [
        [68, 1, 84],
        [72, 40, 120],
        [62, 74, 137],
        [49, 104, 142],
        [38, 130, 142],
        [31, 158, 137],
        [53, 183, 121],
        [109, 205, 89],
        [180, 222, 44],
        [253, 231, 37],
    ],
    dtype=np.float64,
)
_LUT_X = np.linspace(0.0, 1.0, len(_VIRIDIS_STOPS))
_LUT_QUERY = np.linspace(0.0, 1.0, 256)
_VIRIDIS_LUT = [
    QColor(
        int(np.interp(q, _LUT_X, _VIRIDIS_STOPS[:, 0])),
        int(np.interp(q, _LUT_X, _VIRIDIS_STOPS[:, 1])),
        int(np.interp(q, _LUT_X, _VIRIDIS_STOPS[:, 2])),
    )
    for q in _LUT_QUERY
]

TEL_COLORS = ["#f87171", "#34d399", "#60a5fa", "#fbbf24", "#c084fc", "#f472b6"]


def _viridis_qcolor(value: float, vmax: float) -> QColor:
    if vmax <= 0.0 or value <= 0.0:
        return QColor(15, 23, 42)
    t = max(0.0, min(1.0, float(value) / float(vmax)))
    idx = min(255, int(t * 255))
    return _VIRIDIS_LUT[idx]


def _sky_vector_to_cam_deg(
    d_sky: np.ndarray, pointing_zenith_deg: float, pointing_azimuth_deg: float
) -> Tuple[float, float]:
    """Inverse of ``recon.stereo.cam_to_sky_vector``."""
    zen = math.radians(pointing_zenith_deg)
    azi = math.radians(pointing_azimuth_deg + 180.0)
    Rz_inv = np.array(
        [
            [math.cos(azi), math.sin(azi), 0.0],
            [-math.sin(azi), math.cos(azi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    Ry_inv = np.array(
        [
            [math.cos(zen), 0.0, -math.sin(zen)],
            [0.0, 1.0, 0.0],
            [math.sin(zen), 0.0, math.cos(zen)],
        ],
        dtype=np.float64,
    )
    v_cam = Ry_inv @ (Rz_inv @ np.asarray(d_sky, dtype=np.float64))
    z = max(float(v_cam[2]), 1e-6)
    return math.degrees(math.atan2(float(v_cam[0]), z)), math.degrees(math.atan2(float(v_cam[1]), z))


class _HexPixelItem(QGraphicsPolygonItem):
    """Interactive hexagonal PMT polygon that notifies parent camera view on click."""

    def __init__(self, polygon: QPolygonF, tel_idx: int, pix_idx: int, click_cb):
        super().__init__(polygon)
        self.tel_idx = tel_idx
        self.pix_idx = pix_idx
        self._click_cb = click_cb
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event) -> None:
        if self._click_cb is not None:
            self._click_cb(self.tel_idx, self.pix_idx)
        super().mousePressEvent(event)


class HexCameraGraphicsView(QGraphicsView):
    """Native Qt hexagonal camera display with Hillas ellipse/axis overlays."""

    pixel_clicked = Signal(int, int)

    def __init__(self, tel_idx: int, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tel_idx = tel_idx
        self.title = title
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setBackgroundBrush(QBrush(QColor("#090d16")))
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("border: 1px solid #1e293b; border-radius: 6px;")

        self._hex_items: List[_HexPixelItem] = []
        self._overlay_items: List[Any] = []
        self._selected_pix: Optional[int] = None
        self._camera_signature: Optional[Tuple[int, float]] = None

    def set_camera_geometry(self, camera: Camera) -> None:
        sig = (camera.n_pixels, float(camera.pixel_size))
        if self._camera_signature == sig and len(self._hex_items) == camera.n_pixels:
            return
        self._camera_signature = sig
        self._scene.clear()
        self._hex_items.clear()
        self._overlay_items.clear()

        r_hex = float(camera.pixel_size) / math.sqrt(3.0) * 0.96
        base_offsets = [
            (r_hex * math.cos(math.pi / 6.0 + k * math.pi / 3.0), r_hex * math.sin(math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ]

        default_pen = QPen(QColor(30, 41, 59), 0.004)
        default_brush = QBrush(QColor(15, 23, 42))

        for pix_idx, (cx, cy) in enumerate(zip(camera.pixel_x, camera.pixel_y)):
            # Negate y in Qt scene coordinates so +Y points upward
            poly = QPolygonF([QPointF(float(cx + dx), float(-(cy + dy))) for dx, dy in base_offsets])
            item = _HexPixelItem(poly, self.tel_idx, pix_idx, self._handle_pixel_click)
            item.setPen(default_pen)
            item.setBrush(default_brush)
            item.setToolTip(f"Tel {self.tel_idx + 1} | Pixel {pix_idx}\n({cx:+.2f}°, {cy:+.2f}°)")
            self._scene.addItem(item)
            self._hex_items.append(item)

        limit = (camera.n_rings + 1.2) * float(camera.pixel_size)
        self._scene.setSceneRect(-limit, -limit, 2.0 * limit, 2.0 * limit)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _handle_pixel_click(self, tel_idx: int, pix_idx: int) -> None:
        self.set_selected_pixel(pix_idx)
        self.pixel_clicked.emit(tel_idx, pix_idx)

    def set_selected_pixel(self, pix_idx: Optional[int]) -> None:
        if self._selected_pix is not None and 0 <= self._selected_pix < len(self._hex_items):
            self._hex_items[self._selected_pix].setPen(QPen(QColor(30, 41, 59), 0.004))
            self._hex_items[self._selected_pix].setZValue(0)
        self._selected_pix = pix_idx
        if pix_idx is not None and 0 <= pix_idx < len(self._hex_items):
            self._hex_items[pix_idx].setPen(QPen(QColor("#ffffff"), 0.02))
            self._hex_items[pix_idx].setZValue(5)

    def update_image(
        self,
        charges: np.ndarray,
        vmax: float,
        hillas: Optional[HillasParameters] = None,
        show_hillas: bool = True,
        gain_flags: Optional[np.ndarray] = None,
    ) -> None:
        for obj in self._overlay_items:
            self._scene.removeItem(obj)
        self._overlay_items.clear()

        for i, item in enumerate(self._hex_items):
            val = float(charges[i]) if i < len(charges) else 0.0
            color = _viridis_qcolor(val, vmax)
            item.setBrush(QBrush(color))
            lg_str = " [LOW GAIN]" if (gain_flags is not None and i < len(gain_flags) and gain_flags[i] > 0.5) else ""
            item.setToolTip(f"Tel {self.tel_idx + 1} | Pixel {i}{lg_str}\nSignal: {val:.1f} p.e.")

        if show_hillas and hillas is not None:
            tel_color = QColor(TEL_COLORS[self.tel_idx % len(TEL_COLORS)])
            cx, cy = float(hillas.centroid_x), float(hillas.centroid_y)
            axis_len = max(float(hillas.length) * 3.0, 0.8)
            dx = axis_len * math.cos(hillas.psi)
            dy = axis_len * math.sin(hillas.psi)

            axis_line = QGraphicsLineItem(cx - dx, -(cy - dy), cx + dx, -(cy + dy))
            pen_line = QPen(tel_color, 0.025, Qt.PenStyle.DashLine)
            axis_line.setPen(pen_line)
            axis_line.setZValue(10)
            self._scene.addItem(axis_line)
            self._overlay_items.append(axis_line)

            # Draw 2-sigma Hillas ellipse
            w_rad = max(float(hillas.width) * 2.0, 0.04)
            l_rad = max(float(hillas.length) * 2.0, 0.04)
            ellipse = QGraphicsEllipseItem(-l_rad, -w_rad, 2.0 * l_rad, 2.0 * w_rad)
            ellipse.setPos(cx, -cy)
            ellipse.setRotation(-math.degrees(hillas.psi))
            ellipse.setPen(QPen(tel_color, 0.025))
            ellipse.setZValue(10)
            self._scene.addItem(ellipse)
            self._overlay_items.append(ellipse)

            dot = QGraphicsEllipseItem(cx - 0.04, -cy - 0.04, 0.08, 0.08)
            dot.setBrush(QBrush(tel_color))
            dot.setPen(QPen(Qt.PenStyle.NoPen))
            dot.setZValue(11)
            self._scene.addItem(dot)
            self._overlay_items.append(dot)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._scene.sceneRect().isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class SuperimposedStereoGraphicsView(HexCameraGraphicsView):
    """Superimposed camera view showing all telescope Hillas major axes and source markers."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(tel_idx=0, title="Superimposed Stereo View", parent=parent)

    def update_superimposed(
        self,
        combined_charges: np.ndarray,
        hillas_list: List[Optional[HillasParameters]],
        reco_uv: Optional[Tuple[float, float]] = None,
        true_uv: Optional[Tuple[float, float]] = None,
    ) -> None:
        vmax = float(np.max(combined_charges)) if len(combined_charges) else 1.0
        self.update_image(combined_charges, vmax=max(vmax, 1.0), hillas=None, show_hillas=False)

        for t_idx, h in enumerate(hillas_list):
            if h is None:
                continue
            color = QColor(TEL_COLORS[t_idx % len(TEL_COLORS)])
            cx, cy = float(h.centroid_x), float(h.centroid_y)
            axis_len = 2.5
            dx = axis_len * math.cos(h.psi)
            dy = axis_len * math.sin(h.psi)

            line = QGraphicsLineItem(cx - dx, -(cy - dy), cx + dx, -(cy + dy))
            line.setPen(QPen(color, 0.025, Qt.PenStyle.DashLine))
            line.setZValue(10)
            self._scene.addItem(line)
            self._overlay_items.append(line)

            dot = QGraphicsEllipseItem(cx - 0.045, -cy - 0.045, 0.09, 0.09)
            dot.setBrush(QBrush(color))
            dot.setPen(QPen(QColor("#0f172a"), 0.01))
            dot.setZValue(11)
            self._scene.addItem(dot)
            self._overlay_items.append(dot)

        if true_uv is not None:
            tx, ty = true_uv
            for x1, y1, x2, y2 in (
                (tx - 0.09, -ty, tx + 0.09, -ty),
                (tx, -(ty - 0.09), tx, -(ty + 0.09)),
            ):
                marker = QGraphicsLineItem(x1, y1, x2, y2)
                marker.setPen(QPen(QColor("#22d3ee"), 0.035))
                marker.setZValue(15)
                self._scene.addItem(marker)
                self._overlay_items.append(marker)

        if reco_uv is not None:
            rx, ry = reco_uv
            ring = QGraphicsEllipseItem(rx - 0.08, -ry - 0.08, 0.16, 0.16)
            ring.setPen(QPen(QColor("#ffffff"), 0.03))
            ring.setZValue(16)
            self._scene.addItem(ring)
            self._overlay_items.append(ring)


class EventViewerWidget(QWidget):
    """Complete native Cherenkov Event Viewer with time scrubbing, cleaning, and FADC trace inspector."""

    def __init__(self, array_model: DetectorArrayModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.array_model = array_model
        self.events: List[Dict[str, Any]] = []
        self.filtered_indices: List[int] = []
        self.current_pos: int = 0
        self.selected_tel: int = 0
        self.selected_pix: int = 0
        self.dataset_path: str = ""
        self.active_model_path: str = ""
        self.trained_bundle: Optional[Dict[str, Any]] = None
        self.camera_views: List[HexCameraGraphicsView] = []

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._advance_time_slice)

        self._build_ui()
        self.rebuild_camera_grid()
        self._update_provenance_banner()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # --- Controls Bar ---
        ctrl_bar = QHBoxLayout()
        self.btn_load_dir = QPushButton("Load Dataset Dir...")
        self.btn_load_dir.clicked.connect(self._browse_directory)
        self.btn_load_file = QPushButton("Load Raw / Batch File...")
        self.btn_load_file.clicked.connect(self._browse_file)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Events", "Gammas Only", "Hadrons Only"])
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)

        self.btn_prev = QPushButton("Prev")
        self.btn_prev.clicked.connect(self._prev_event)
        self.event_spin = QSpinBox()
        self.event_spin.setRange(0, 0)
        self.event_spin.valueChanged.connect(self._on_event_spin_changed)
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self._next_event)
        self.lbl_count = QLabel("/ 0 events")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                "DoublePass Cleaned (VERITAS)",
                "Tailcut Cleaned",
                "Raw Integrated Charge",
                "Time-Slice Frame",
            ]
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.chk_hillas = QCheckBox("Hillas Overlay")
        self.chk_hillas.setChecked(True)
        self.chk_hillas.toggled.connect(self.render_current_event)

        ctrl_bar.addWidget(self.btn_load_dir)
        ctrl_bar.addWidget(self.btn_load_file)
        ctrl_bar.addWidget(QLabel("Filter:"))
        ctrl_bar.addWidget(self.filter_combo)
        ctrl_bar.addWidget(self.btn_prev)
        ctrl_bar.addWidget(self.event_spin)
        ctrl_bar.addWidget(self.lbl_count)
        ctrl_bar.addWidget(self.btn_next)
        ctrl_bar.addWidget(QLabel("View:"))
        ctrl_bar.addWidget(self.mode_combo)
        ctrl_bar.addWidget(self.chk_hillas)
        ctrl_bar.addStretch(1)
        root.addLayout(ctrl_bar)

        self.lbl_provenance = QLabel("")
        self.lbl_provenance.setWordWrap(True)
        self.lbl_provenance.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #1e293b; border-radius: 5px; "
            "padding: 4px 8px; color: #38bdf8; font-weight: 600; font-size: 11px;"
        )
        root.addWidget(self.lbl_provenance)

        # --- Time-Slice Scrubber Bar ---
        time_bar = QHBoxLayout()
        time_bar.addWidget(QLabel("FADC Time Slice (ns):"))
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, max(0, self.array_model.fadc_config.n_time_bins - 1))
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self._on_time_slider_changed)
        self.lbl_time_bin = QLabel("Bin 0 (0.0 ns)")
        self.btn_play_time = QPushButton("Play Pulse")
        self.btn_play_time.setCheckable(True)
        self.btn_play_time.toggled.connect(self._toggle_time_animation)
        time_bar.addWidget(self.time_slider, stretch=1)
        time_bar.addWidget(self.lbl_time_bin)
        time_bar.addWidget(self.btn_play_time)
        root.addLayout(time_bar)

        # --- Main Splitter: Left Cameras + Superimposed | Right Metadata + FADC Trace ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_layout = QHBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.cam_grid_host = QWidget()
        self.cam_grid_layout = QGridLayout(self.cam_grid_host)
        self.cam_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.cam_grid_layout.setSpacing(6)
        left_layout.addWidget(self.cam_grid_host, stretch=3)

        stereo_box = QWidget()
        stereo_v = QVBoxLayout(stereo_box)
        stereo_v.setContentsMargins(0, 0, 0, 0)
        stereo_v.setSpacing(4)
        self.lbl_stereo_title = QLabel("Superimposed Array & SDP Axes (Cyan=+ True, White=O Reco)")
        self.lbl_stereo_title.setStyleSheet("font-weight: 600; color: #cbd5e1;")
        self.superimposed_view = SuperimposedStereoGraphicsView()
        stereo_v.addWidget(self.lbl_stereo_title)
        stereo_v.addWidget(self.superimposed_view, stretch=1)
        left_layout.addWidget(stereo_box, stretch=2)

        splitter.addWidget(left_widget)

        # Right side: Metadata + Hillas Table + FADC Trace Matplotlib Canvas
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.lbl_meta = QLabel("Load an event dataset (.pt, .h5, .root, .vbf) to inspect showers.")
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 8px; color: #e2e8f0;"
        )
        self.lbl_meta.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.lbl_meta)

        self.hillas_table = QTableWidget(0, 6)
        self.hillas_table.setHorizontalHeaderLabels(["Tel", "Size (p.e.)", "Length", "Width", "Loss1", "t_grad"])
        self.hillas_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.hillas_table.setMaximumHeight(165)
        right_layout.addWidget(self.hillas_table)

        self.trace_fig = Figure(figsize=(4.5, 3.0), facecolor="#0b1120")
        self.trace_canvas = FigureCanvas(self.trace_fig)
        self.trace_ax = self.trace_fig.add_subplot(111)
        self._style_trace_axes()
        right_layout.addWidget(self.trace_canvas, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setSizes([860, 420])
        root.addWidget(splitter, stretch=1)

    def _style_trace_axes(self) -> None:
        ax = self.trace_ax
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.set_xlabel("Time Sample (2 ns/bin)", color="#cbd5e1", fontsize=9)
        ax.set_ylabel("Signal (p.e.)", color="#cbd5e1", fontsize=9)
        ax.set_title("Click any camera pixel to inspect FADC trace", color="#e2e8f0", fontsize=10)
        ax.grid(True, alpha=0.2, color="#64748b")

    def _update_provenance_banner(self) -> None:
        prov = describe_model_and_dataset(
            model_path=self.active_model_path,
            data_path=self.dataset_path,
            bundle=self.trained_bundle,
            array_preset=self.array_model.preset_name,
        )
        self.lbl_provenance.setText(
            f"Event Viewer Reconstruction:  Dataset: {prov['dataset_title']}  |  Array: {prov['array_preset']}  |  "
            f"Dir/Core: Geometric Hillas Stereo (SDP)  |  Energy & G/H: {prov['model_title']}"
        )

    def set_array_model(self, array_model: DetectorArrayModel) -> None:
        self.array_model = array_model
        self.time_slider.setRange(0, max(0, self.array_model.fadc_config.n_time_bins - 1))
        self.rebuild_camera_grid()
        self._update_provenance_banner()
        if self.events:
            self.render_current_event()

    def set_active_model(self, model_path: str, bundle: Optional[Dict[str, Any]] = None) -> None:
        self.active_model_path = str(model_path or "").strip()
        self.trained_bundle = bundle if isinstance(bundle, dict) else None
        self._update_provenance_banner()
        if self.events:
            self.render_current_event()

    def set_trained_bundle(self, bundle: Optional[Dict[str, Any]]) -> None:
        m_path = str(bundle.get("model_path", "") or "") if isinstance(bundle, dict) else ""
        self.set_active_model(m_path or self.active_model_path, bundle)

    def rebuild_camera_grid(self) -> None:
        while self.cam_grid_layout.count():
            item = self.cam_grid_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.camera_views.clear()

        n_tel = max(1, len(self.array_model.telescopes_spec))
        camera = self.array_model.primary_camera()
        cols = 2 if n_tel > 1 else 1
        for idx in range(n_tel):
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)
            lbl = QLabel(f"Telescope {idx + 1}")
            lbl.setStyleSheet(f"font-weight: 600; color: {TEL_COLORS[idx % len(TEL_COLORS)]};")
            view = HexCameraGraphicsView(idx, f"Tel {idx + 1}")
            view.set_camera_geometry(camera)
            view.pixel_clicked.connect(self.on_pixel_clicked)
            v.addWidget(lbl)
            v.addWidget(view, stretch=1)
            self.cam_grid_layout.addWidget(box, idx // cols, idx % cols)
            self.camera_views.append(view)

        self.superimposed_view.set_camera_geometry(camera)

    def load_path(self, path: str) -> int:
        self.dataset_path = str(path or "").replace("\\", "/").strip()
        self._update_provenance_banner()
        loaded = load_events_universal(path, array_preset=self.array_model.preset_name, max_events=5000)
        self.set_events(loaded)
        return len(loaded)

    def set_events(self, events: List[Dict[str, Any]]) -> None:
        self.events = events
        self._apply_filter()

    @Slot()
    def _browse_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Event Dataset Directory", "data")
        if folder:
            self.load_path(folder)

    @Slot()
    def _browse_file(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Event / Raw Telescope File",
            "data",
            "All Supported (*.pt *.h5 *.hdf5 *.root *.vbf *.npz);;PyTorch Batches (*.pt);;HDF5 (*.h5 *.hdf5);;ROOT (*.root);;VBF (*.vbf);;NPZ (*.npz)",
        )
        if fpath:
            self.load_path(fpath)

    @Slot()
    def _apply_filter(self) -> None:
        mode = self.filter_combo.currentText()
        if mode == "Gammas Only":
            self.filtered_indices = [i for i, e in enumerate(self.events) if e.get("label", -1) == 1]
        elif mode == "Hadrons Only":
            self.filtered_indices = [i for i, e in enumerate(self.events) if e.get("label", -1) == 0]
        else:
            self.filtered_indices = list(range(len(self.events)))

        n = len(self.filtered_indices)
        self.event_spin.blockSignals(True)
        self.event_spin.setRange(0, max(0, n - 1))
        self.event_spin.setValue(0)
        self.event_spin.blockSignals(False)
        self.lbl_count.setText(f"/ {n} events")
        self.current_pos = 0
        if n > 0:
            self.render_current_event()

    @Slot(int)
    def _on_event_spin_changed(self, value: int) -> None:
        self.current_pos = int(value)
        self.render_current_event()

    @Slot()
    def _prev_event(self) -> None:
        if self.current_pos > 0:
            self.event_spin.setValue(self.current_pos - 1)

    @Slot()
    def _next_event(self) -> None:
        if self.current_pos + 1 < len(self.filtered_indices):
            self.event_spin.setValue(self.current_pos + 1)

    @Slot()
    def _on_mode_changed(self) -> None:
        self.render_current_event()

    @Slot(int)
    def _on_time_slider_changed(self, bin_idx: int) -> None:
        bw = float(self.array_model.fadc_config.bin_width_ns)
        self.lbl_time_bin.setText(f"Bin {bin_idx} ({bin_idx * bw:.1f} ns)")
        if self.mode_combo.currentText() == "Time-Slice Frame":
            self.render_current_event()

    @Slot(bool)
    def _toggle_time_animation(self, checked: bool) -> None:
        if checked:
            self.mode_combo.setCurrentText("Time-Slice Frame")
            self.btn_play_time.setText("Pause Pulse")
            self._anim_timer.start(220)
        else:
            self.btn_play_time.setText("Play Pulse")
            self._anim_timer.stop()

    @Slot()
    def _advance_time_slice(self) -> None:
        nxt = self.time_slider.value() + 1
        if nxt > self.time_slider.maximum():
            nxt = 0
        self.time_slider.setValue(nxt)

    @Slot(int, int)
    def on_pixel_clicked(self, tel_idx: int, pix_idx: int) -> None:
        self.selected_tel = int(tel_idx)
        self.selected_pix = int(pix_idx)
        self._update_trace_plot()

    @Slot()
    def render_current_event(self) -> None:
        if not self.filtered_indices or self.current_pos >= len(self.filtered_indices):
            return

        ev = self.events[self.filtered_indices[self.current_pos]]
        traces_all = np.asarray(ev["fadc_traces"], dtype=np.float32)
        gains_all = np.asarray(ev.get("gain_flags", np.zeros(traces_all.shape[:2])), dtype=np.float32)

        n_bins = traces_all.shape[2]
        if self.time_slider.maximum() != max(0, n_bins - 1):
            self.time_slider.blockSignals(True)
            self.time_slider.setRange(0, max(0, n_bins - 1))
            self.time_slider.blockSignals(False)

        n_tel_ev = traces_all.shape[0]
        n_pix_ev = traces_all.shape[1]

        # Adapt camera view if event comes from a different preset/geometry
        camera = self.array_model.primary_camera()
        if camera.n_pixels != n_pix_ev:
            for preset in ("VERITAS", "HESS", "MAGIC"):
                from gui.array_models import get_preset_model

                cand = get_preset_model(preset)
                if cand.primary_camera().n_pixels == n_pix_ev:
                    camera = cand.primary_camera()
                    break
        if len(self.camera_views) != n_tel_ev:
            # Ensure view count matches event telescope count
            temp_specs = self.array_model.telescopes_spec[:n_tel_ev]
            while len(temp_specs) < n_tel_ev:
                temp_specs.append(dict(self.array_model.telescopes_spec[0]))
            for v in self.camera_views:
                v.set_camera_geometry(camera)
            self.superimposed_view.set_camera_geometry(camera)

        array = self.array_model.build_telescope_array()
        low_gain_factor = float(self.array_model.fadc_config.low_gain_factor)
        mode = self.mode_combo.currentText()
        time_bin = min(self.time_slider.value(), n_bins - 1)
        show_hillas = self.chk_hillas.isChecked()

        per_tel_disp: List[np.ndarray] = []
        hillas_list: List[Optional[HillasParameters]] = []
        brightest_tel, brightest_pix, brightest_val = 0, 0, -1.0

        self.hillas_table.setRowCount(0)
        for t_idx in range(min(n_tel_ev, len(self.camera_views))):
            restored = restore_low_gain_traces(traces_all[t_idx], gains_all[t_idx], low_gain_factor=low_gain_factor)
            dp = double_pass_fadc_clean(restored, camera)
            h_dp = compute_hillas(camera, dp.charges, dp.mask) if np.count_nonzero(dp.mask) >= 3 else None

            if mode == "DoublePass Cleaned (VERITAS)":
                disp = np.where(dp.mask, dp.charges, 0.0)
                h_use = h_dp
            elif mode == "Tailcut Cleaned":
                c_raw = np.sum(restored, axis=1)
                c_tc, mask_tc = tailcut_clean(c_raw, camera)
                disp = np.where(mask_tc, c_tc, 0.0)
                h_use = compute_hillas(camera, c_tc, mask_tc) if np.count_nonzero(mask_tc) >= 3 else None
            elif mode == "Time-Slice Frame":
                disp = np.maximum(restored[:, time_bin], 0.0)
                h_use = h_dp
            else:  # Raw Integrated Charge
                disp = np.maximum(np.sum(restored, axis=1), 0.0)
                h_use = h_dp

            if len(disp) > 0:
                p_max = int(np.argmax(disp))
                if float(disp[p_max]) > brightest_val:
                    brightest_val = float(disp[p_max])
                    brightest_tel = t_idx
                    brightest_pix = p_max

            per_tel_disp.append(disp)
            hillas_list.append(h_use)

            if h_dp is not None:
                row = self.hillas_table.rowCount()
                self.hillas_table.insertRow(row)
                for col_i, txt in enumerate(
                    [
                        f"T{t_idx + 1}",
                        f"{h_dp.size:.1f}",
                        f"{h_dp.length:.3f}°",
                        f"{h_dp.width:.3f}°",
                        f"{dp.loss1:.2f}",
                        f"{dp.time_gradient:+.2f}",
                    ]
                ):
                    self.hillas_table.setItem(row, col_i, QTableWidgetItem(txt))

        global_vmax = max([float(np.max(c)) for c in per_tel_disp] + [1.0])
        for t_idx, disp in enumerate(per_tel_disp):
            self.camera_views[t_idx].set_camera_geometry(camera)
            self.camera_views[t_idx].update_image(
                disp,
                vmax=global_vmax,
                hillas=hillas_list[t_idx],
                show_hillas=show_hillas,
                gain_flags=gains_all[t_idx],
            )

        zen = float(ev.get("zenith_deg", 30.0))
        azi = float(ev.get("azimuth_deg", 0.0))
        t_zen = float(ev.get("true_zenith_deg", zen))
        t_azi = float(ev.get("true_azimuth_deg", azi))

        d_true = np.array(
            [
                -math.sin(math.radians(t_zen)) * math.cos(math.radians(t_azi)),
                -math.sin(math.radians(t_zen)) * math.sin(math.radians(t_azi)),
                math.cos(math.radians(t_zen)),
            ],
            dtype=np.float64,
        )
        true_uv = _sky_vector_to_cam_deg(d_true, zen, azi)

        geom_res = reconstruct_event_geometric(
            traces_all[: len(array.telescopes)],
            array.telescopes[:n_tel_ev],
            camera,
            pointing_zenith_deg=zen,
            pointing_azimuth_deg=azi,
            z_obs=float(self.array_model.observatory_altitude_m),
            gain_flags=gains_all[: len(array.telescopes)],
        )
        reco_uv = (
            _sky_vector_to_cam_deg(geom_res.reco_direction_vector, zen, azi)
            if geom_res is not None
            else None
        )

        combined = np.sum(np.vstack(per_tel_disp), axis=0) if per_tel_disp else np.zeros(n_pix_ev)
        self.superimposed_view.set_camera_geometry(camera)
        self.superimposed_view.update_superimposed(combined, hillas_list, reco_uv=reco_uv, true_uv=true_uv)

        prov = describe_model_and_dataset(
            model_path=self.active_model_path,
            data_path=self.dataset_path,
            bundle=self.trained_bundle,
            array_preset=self.array_model.preset_name,
        )
        from sim.provenance import clean_float, format_event_provenance

        lbl_val = ev.get("label", -1)
        lbl_txt = "Gamma-ray" if lbl_val == 1 else ("Proton / Hadron" if lbl_val == 0 else "Real / Unlabeled")
        e_true = clean_float(ev.get("energy"))
        e_str = f"{e_true:.1f} GeV ({e_true / 1000.0:.2f} TeV)" if (e_true is not None and e_true > 0) else "Not recorded"
        ix_val = clean_float(ev.get("impact_x"))
        iy_val = clean_float(ev.get("impact_y"))
        core_str = f"({ix_val:.1f}, {iy_val:.1f}) m" if (ix_val is not None and iy_val is not None) else "Not recorded"
        zen_disp = clean_float(zen)
        azi_disp = clean_float(azi)
        pt_str = (
            f"Zen={zen_disp:.1f}°, Azi={azi_disp:.1f}°"
            if (zen_disp is not None and azi_disp is not None)
            else "Not recorded"
        )
        sim_ver_str = format_event_provenance(ev)

        ml_str = "Not loaded (train or load Stereo ML / GNN model in Tab 3)"
        if self.trained_bundle is not None and "classifier" in self.trained_bundle and "energy_regressor" in self.trained_bundle:
            try:
                feat = extract_stereo_features_any_array(
                    traces_all,
                    array.telescopes,
                    camera,
                    pointing_zenith_deg=zen,
                    pointing_azimuth_deg=azi,
                    z_obs=float(self.array_model.observatory_altitude_m),
                    gain_flags=gains_all,
                    include_extended=True,
                )[None, :]
                prob_g = float(self.trained_bundle["classifier"].predict_proba(feat[:, self.trained_bundle["gh_mask"]])[0])
                pred_loge = float(self.trained_bundle["energy_regressor"].predict(feat)[0])
                cal = self.trained_bundle.get("calibrator")
                if cal is not None and cal.is_fitted:
                    pred_loge = float(cal.predict(np.array([pred_loge]))[0])
                pred_e_gev = 10.0 ** pred_loge
                ml_str = f"E_reco = {pred_e_gev:.1f} GeV ({pred_e_gev / 1000.0:.2f} TeV) | P(gamma) = {prob_g:.1%}"
            except Exception as exc:
                ml_str = f"Unavailable ({exc})"
        elif self.active_model_path or self.trained_bundle is not None:
            ml_str = "Dataset-level E_reco & P(gamma) in Tabs 3-5 (live single-trace inference via Stereo ML bundle)"

        geom_str = (
            f"Zen={geom_res.reco_zenith_deg:.2f}°, Azi={geom_res.reco_azimuth_deg:.2f}° | "
            f"Core=({geom_res.reco_impact_x:.1f}, {geom_res.reco_impact_y:.1f}) m "
            f"({geom_res.n_telescopes_used} tels)"
            if geom_res is not None
            else "Failed (< 2 valid stereo images)"
        )
        self.lbl_meta.setText(
            f"Event #{self.filtered_indices[self.current_pos]} [{ev.get('source_format', 'pt')}]\n"
            f"Dataset:     {prov['dataset_title']}\n"
            f"Sim Version: {sim_ver_str}\n"
            f"Model:       {prov['model_title']}\n"
            f"Particle:    {lbl_txt}   |   True Energy: {e_str}\n"
            f"True Core:   {core_str}   |   Pointing: {pt_str}\n"
            f"SDP Reco (Dir/Core): {geom_str}\n"
            f"ML/GNN Reco (E, G/H): {ml_str}"
        )

        self.selected_tel = brightest_tel
        self.selected_pix = brightest_pix
        if 0 <= self.selected_tel < len(self.camera_views):
            self.camera_views[self.selected_tel].set_selected_pixel(self.selected_pix)
        self._update_trace_plot()

    def _update_trace_plot(self) -> None:
        if not self.filtered_indices or self.current_pos >= len(self.filtered_indices):
            return
        ev = self.events[self.filtered_indices[self.current_pos]]
        traces_all = np.asarray(ev["fadc_traces"], dtype=np.float32)
        gains_all = np.asarray(ev.get("gain_flags", np.zeros(traces_all.shape[:2])), dtype=np.float32)
        if self.selected_tel >= traces_all.shape[0] or self.selected_pix >= traces_all.shape[1]:
            return

        raw_trace = traces_all[self.selected_tel, self.selected_pix]
        is_lg = bool(gains_all[self.selected_tel, self.selected_pix] > 0.5)
        lg_factor = float(self.array_model.fadc_config.low_gain_factor)
        restored_trace = raw_trace * lg_factor if is_lg else raw_trace

        ax = self.trace_ax
        ax.clear()
        self._style_trace_axes()
        x_bins = np.arange(len(restored_trace))
        ax.plot(x_bins, restored_trace, "o-", color="#38bdf8", lw=2, markersize=4, label="Restored FADC (p.e.)")
        if is_lg:
            ax.plot(x_bins, raw_trace, "--", color="#f59e0b", lw=1.3, label=f"Raw Low-Gain (÷{lg_factor:.0f})")
        ax.fill_between(x_bins, 0, restored_trace, color="#38bdf8", alpha=0.18)

        if self.mode_combo.currentText() == "Time-Slice Frame":
            t_cur = min(self.time_slider.value(), len(restored_trace) - 1)
            ax.axvline(t_cur, color="#f43f5e", ls="--", lw=1.5, label=f"Slice t={t_cur}")

        ax.set_title(
            f"Tel {self.selected_tel + 1} | Pixel #{self.selected_pix}"
            + (" [LOW GAIN]" if is_lg else " [HIGH GAIN]")
            + f" | Sum={np.sum(restored_trace):.1f} p.e.",
            color="#e2e8f0",
            fontsize=9.5,
        )
        ax.legend(loc="upper right", fontsize=8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
        self.trace_fig.tight_layout()
        self.trace_canvas.draw_idle()
