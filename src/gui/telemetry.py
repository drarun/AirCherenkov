"""Real-time hardware telemetry (GPU, VRAM, CPU, RAM) and pipeline status bars."""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)
import torch

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


class HardwareTelemetry:
    """Queries NVIDIA GPU utilization/VRAM via NVML with PyTorch & psutil fallback."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._nvml_ok = False
        self._handle = None
        self._gpu_name = "CPU Mode"

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                if count > device_index:
                    self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                    raw_name = pynvml.nvmlDeviceGetName(self._handle)
                    self._gpu_name = (
                        raw_name.decode("utf-8", errors="ignore")
                        if isinstance(raw_name, bytes)
                        else str(raw_name)
                    )
                    self._nvml_ok = True
            except Exception:
                self._nvml_ok = False

        if not self._nvml_ok and torch.cuda.is_available():
            try:
                self._gpu_name = torch.cuda.get_device_name(device_index)
            except Exception:
                self._gpu_name = "CUDA GPU"

        if psutil is not None:
            try:
                # Prime non-blocking CPU percent measurement
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def poll(self) -> Dict[str, Any]:
        """Return current GPU, VRAM, temperature, CPU, and system RAM metrics."""
        gpu_util = 0.0
        vram_used_mb = 0.0
        vram_total_mb = 0.0
        vram_pct = 0.0
        gpu_temp: Optional[float] = None
        cuda_alloc_mb = 0.0

        if self._nvml_ok and self._handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                gpu_util = float(util.gpu)
            except Exception:
                pass
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                vram_used_mb = float(mem.used) / (1024.0 ** 2)
                vram_total_mb = float(mem.total) / (1024.0 ** 2)
                if vram_total_mb > 0:
                    vram_pct = 100.0 * vram_used_mb / vram_total_mb
            except Exception:
                pass
            try:
                gpu_temp = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass
        elif torch.cuda.is_available():
            try:
                free_b, total_b = torch.cuda.mem_get_info(self.device_index)
                used_b = total_b - free_b
                vram_used_mb = float(used_b) / (1024.0 ** 2)
                vram_total_mb = float(total_b) / (1024.0 ** 2)
                if vram_total_mb > 0:
                    vram_pct = 100.0 * vram_used_mb / vram_total_mb
            except Exception:
                pass

        if torch.cuda.is_available():
            try:
                cuda_alloc_mb = float(torch.cuda.memory_allocated(self.device_index)) / (1024.0 ** 2)
            except Exception:
                pass

        cpu_pct = 0.0
        ram_pct = 0.0
        if psutil is not None:
            try:
                cpu_pct = float(psutil.cpu_percent(interval=None))
                ram_pct = float(psutil.virtual_memory().percent)
            except Exception:
                pass

        return {
            "gpu_name": self._gpu_name,
            "gpu_util_pct": gpu_util,
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
            "vram_pct": vram_pct,
            "gpu_temp_c": gpu_temp,
            "cuda_allocated_mb": cuda_alloc_mb,
            "cpu_pct": cpu_pct,
            "ram_pct": ram_pct,
        }

    def close(self) -> None:
        if self._nvml_ok and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_ok = False


def _make_bar(accent: str, height: int = 12) -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setFixedHeight(height)
    bar.setStyleSheet(
        f"""
        QProgressBar {{
            border: 1px solid #334155;
            border-radius: 4px;
            background-color: #0f172a;
            color: #e2e8f0;
            font-size: 10px;
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {accent};
            border-radius: 3px;
        }}
        """
    )
    return bar


class SystemAndPipelineStatusBar(QFrame):
    """Persistent status strip tracking Simulation, Training, Analysis, and GPU usage."""

    def __init__(self, parent: Optional[QWidget] = None, poll_interval_ms: int = 750):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("PipelineStatusBar")
        self.setStyleSheet(
            """
            #PipelineStatusBar {
                background-color: #0b1120;
                border-top: 1px solid #1e293b;
                padding: 4px 8px;
            }
            QLabel.statusTitle {
                color: #94a3b8;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel.statusDetail {
                color: #cbd5e1;
                font-size: 10px;
            }
            """
        )

        self.telemetry = HardwareTelemetry()

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(2)

        # 1. Simulation status
        self.sim_title = QLabel("SIMULATION")
        self.sim_title.setProperty("class", "statusTitle")
        self.sim_bar = _make_bar("#38bdf8")
        self.sim_detail = QLabel("Idle")
        self.sim_detail.setProperty("class", "statusDetail")
        layout.addWidget(self._build_cell(self.sim_title, self.sim_bar, self.sim_detail), 0, 0)

        # 2. Training status
        self.train_title = QLabel("MODEL TRAINING")
        self.train_title.setProperty("class", "statusTitle")
        self.train_bar = _make_bar("#34d399")
        self.train_detail = QLabel("Idle")
        self.train_detail.setProperty("class", "statusDetail")
        layout.addWidget(self._build_cell(self.train_title, self.train_bar, self.train_detail), 0, 1)

        # 3. Data analysis (IRF + Spectral) status
        self.analysis_title = QLabel("IRF & SPECTRAL ANALYSIS")
        self.analysis_title.setProperty("class", "statusTitle")
        self.analysis_bar = _make_bar("#a78bfa")
        self.analysis_detail = QLabel("Idle")
        self.analysis_detail.setProperty("class", "statusDetail")
        layout.addWidget(self._build_cell(self.analysis_title, self.analysis_bar, self.analysis_detail), 0, 2)

        # 4. GPU & System usage
        self.gpu_title = QLabel("GPU / HARDWARE")
        self.gpu_title.setProperty("class", "statusTitle")
        self.gpu_util_bar = _make_bar("#f59e0b")
        self.gpu_util_bar.setFormat("GPU %p%")
        self.vram_bar = _make_bar("#f43f5e")
        self.vram_bar.setFormat("VRAM %p%")
        self.gpu_detail = QLabel("Initializing telemetry...")
        self.gpu_detail.setProperty("class", "statusDetail")

        gpu_box = QWidget()
        gpu_v = QVBoxLayout(gpu_box)
        gpu_v.setContentsMargins(0, 0, 0, 0)
        gpu_v.setSpacing(2)
        gpu_v.addWidget(self.gpu_title)
        bars_row = QHBoxLayout()
        bars_row.setSpacing(6)
        bars_row.addWidget(self.gpu_util_bar)
        bars_row.addWidget(self.vram_bar)
        gpu_v.addLayout(bars_row)
        gpu_v.addWidget(self.gpu_detail)
        layout.addWidget(gpu_box, 0, 3)

        for col in range(4):
            layout.setColumnStretch(col, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_hardware)
        self._timer.start(poll_interval_ms)
        self.refresh_hardware()

    @staticmethod
    def _build_cell(title: QLabel, bar: QProgressBar, detail: QLabel) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        v.addWidget(title)
        v.addWidget(bar)
        v.addWidget(detail)
        return box

    @Slot(int, str)
    def set_sim_status(self, pct: int, detail: str) -> None:
        self.sim_bar.setValue(max(0, min(100, int(pct))))
        self.sim_detail.setText(detail)

    @Slot(int, str)
    def set_train_status(self, pct: int, detail: str) -> None:
        self.train_bar.setValue(max(0, min(100, int(pct))))
        self.train_detail.setText(detail)

    @Slot(int, str)
    def set_analysis_status(self, pct: int, detail: str) -> None:
        self.analysis_bar.setValue(max(0, min(100, int(pct))))
        self.analysis_detail.setText(detail)

    @Slot()
    def refresh_hardware(self) -> Dict[str, Any]:
        stats = self.telemetry.poll()
        gpu_pct = int(round(stats["gpu_util_pct"]))
        vram_pct = int(round(stats["vram_pct"]))
        self.gpu_util_bar.setValue(max(0, min(100, gpu_pct)))
        self.vram_bar.setValue(max(0, min(100, vram_pct)))

        vram_used_gb = stats["vram_used_mb"] / 1024.0
        vram_tot_gb = stats["vram_total_mb"] / 1024.0
        temp_str = f" | {stats['gpu_temp_c']:.0f}°C" if stats["gpu_temp_c"] is not None else ""
        if vram_tot_gb > 0:
            gpu_str = f"{stats['gpu_name']}{temp_str} | VRAM {vram_used_gb:.1f}/{vram_tot_gb:.1f} GB"
        else:
            gpu_str = f"{stats['gpu_name']}"
        self.gpu_detail.setText(
            f"{gpu_str} | CPU {stats['cpu_pct']:.0f}% | RAM {stats['ram_pct']:.0f}%"
        )
        return stats

    def shutdown(self) -> None:
        self._timer.stop()
        self.telemetry.close()
