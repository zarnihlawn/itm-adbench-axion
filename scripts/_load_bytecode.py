"""Load a wiped .py module from scripts/_bytecode_bak/*.pyc."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BAK = Path(__file__).resolve().parent / "_bytecode_bak"
_SCRIPTS = Path(__file__).resolve().parent
_AXION_ROOT = _SCRIPTS.parent


def load_bytecode_module(module_name: str, pyc_stem: str | None = None) -> ModuleType:
    stem = pyc_stem or module_name
    pyc = _BAK / f"{stem}.cpython-312.pyc"
    if not pyc.exists():
        raise ImportError(f"Bytecode backup missing: {pyc}")
    existing = sys.modules.get(module_name)
    if existing is not None and getattr(existing, "__axion_bytecode__", False):
        return existing
    spec = importlib.util.spec_from_file_location(module_name, pyc)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {pyc}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    # pyc __file__ is under _bytecode_bak, so Path(__file__).parents[1] is scripts/
    if hasattr(mod, "ROOT"):
        mod.ROOT = _AXION_ROOT
        if module_name == "beat_paper_select":
            mod.LOCAL_MAP_PATH = (
                _AXION_ROOT / "results" / "axion_beat_paper_local" / "thesis" / "recipe_map.json"
            )
            mod.SMOKE_ROOT = _AXION_ROOT / "results" / "axion_beat_paper_smoke"
            mod.SMOKE_MAP_PATH = mod.SMOKE_ROOT / "thesis" / "recipe_map.json"
        if module_name == "beat_paper_run":
            # re-bind paths imported from select if present
            try:
                import beat_paper_select as _sel

                mod.LOCAL_MAP_PATH = _sel.LOCAL_MAP_PATH
                mod.SMOKE_MAP_PATH = _sel.SMOKE_MAP_PATH
            except Exception:
                pass
    mod.__axion_bytecode__ = True
    if module_name == "beat_paper_select":
        try:
            from beat_paper_select_patches import apply_select_patches

            apply_select_patches(mod)
        except Exception as exc:  # noqa: BLE001
            mod.__semi_heavy_patch_error__ = str(exc)
    if module_name == "beat_paper_run":
        try:
            from beat_paper_run_patches import apply_run_patches

            apply_run_patches(mod)
        except Exception as exc:  # noqa: BLE001
            mod.__semi_heavy_run_patch_error__ = str(exc)
    return mod


def reexport_into(globals_dict: dict, module_name: str, pyc_stem: str | None = None) -> ModuleType:
    mod = load_bytecode_module(module_name, pyc_stem=pyc_stem)
    for k, v in vars(mod).items():
        if k.startswith("__") and k not in ("__doc__",):
            continue
        globals_dict[k] = v
    return mod
