"""Execute frozen Beat-Paper recipe map.

Loads the verified Python 3.12 bytecode module (source was wiped Aug 15) and applies
run patches. Full human-readable reconstruction is tracked as a follow-up; the
bytecode + patches path is the production runner for dual_lift scoring.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _load_bytecode import reexport_into  # noqa: E402

reexport_into(globals(), "beat_paper_run", "beat_paper_run")

if __name__ == "__main__":
    main()  # noqa: F821
