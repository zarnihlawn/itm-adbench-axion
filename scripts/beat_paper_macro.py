"""Beat-Paper macro / compare helpers (bytecode restore)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _load_bytecode import reexport_into  # noqa: E402

reexport_into(globals(), "beat_paper_macro", "beat_paper_macro")

if __name__ == "__main__":
    main()  # noqa: F821
