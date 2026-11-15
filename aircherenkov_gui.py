"""Root-level launcher for AirCherenkov Studio standalone desktop GUI."""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from gui.app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
