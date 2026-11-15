"""Application entry point for AirCherenkov Studio."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence


def _ensure_src_on_path() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(
        prog="aircherenkov-gui",
        description="Launch AirCherenkov Studio standalone native desktop application.",
    )
    parser.add_argument(
        "--preset",
        default="VERITAS",
        choices=["VERITAS", "HESS", "MAGIC"],
        help="Initial telescope array preset (default: %(default)s)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional event dataset directory or file to open in the Event Viewer",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from PySide6.QtWidgets import QApplication
    from gui.main_window import AirCherenkovMainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AirCherenkov Studio")
    window = AirCherenkovMainWindow(initial_preset=args.preset)
    if args.data:
        window.viewer_tab.load_path(args.data)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
