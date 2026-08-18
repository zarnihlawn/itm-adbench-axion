#!/usr/bin/env bash
# Assert campus ceiling run ingredients match frozen dual_lift claim.
# Usage (repo root):
#   bash scripts/verify_ceiling_parity.sh              # config + embeds + optional seed111
#   bash scripts/verify_ceiling_parity.sh --config-only
#   bash scripts/verify_ceiling_parity.sh --seed111-only
# Env: RUN_ID, CFG, SEED111_SRC, TOL (default 0.01 for per-slot PR/ROC)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
RUN_ID="${RUN_ID:-axion_beat_paper_dual_lift_gpu}"
CEILING="results/axion_beat_paper_dual_lift"
MAP="${MAP:-$CEILING/thesis/recipe_map_57_beat.json}"
TOL="${TOL:-0.01}"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
export PYTHONPATH=src:scripts
PY="${PY:-python3}"

_pick_cfg() {
  if [[ -n "${CFG:-}" ]]; then
    echo "$CFG"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local vram_mb
    vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -n "$vram_mb" && "$vram_mb" -ge 20000 ]]; then
      echo "configs/gpu_beat_paper_3090.yaml"
      return
    fi
  fi
  echo "configs/gpu_beat_paper.yaml"
}

CFG="$(_pick_cfg)"
export CFG

echo "== ceiling parity =="
echo "CEILING=$CEILING"
echo "RUN_ID=$RUN_ID"
echo "CFG=$CFG"
echo "MAP=$MAP"

for f in \
  "$CEILING/dual_lift_report.json" \
  "$CEILING/compare_to_ddae.json" \
  "$CEILING/STATUS.md" \
  "$MAP" \
  configs/gpu_beat_paper.yaml \
  configs/gpu_beat_paper_3090.yaml \
  configs/gpu_final.yaml; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

echo ""
echo "== frozen claim targets (seed-111 macro, n=57) =="
"$PY" - <<'PY'
import json
from pathlib import Path

ceiling = Path("results/axion_beat_paper_dual_lift")
report = json.loads((ceiling / "dual_lift_report.json").read_text())
cmp_ = json.loads((ceiling / "compare_to_ddae.json").read_text())
paper = cmp_["paper"]
for setting in ("semi-supervised", "unsupervised"):
    m = report["settings"][setting]
    p = paper[setting]
    print(
        f"{setting}: PR={m['PR-AUC']:.2f} ROC={m['ROC-AUC']:.2f} "
        f"(paper {p['PR-AUC']:.2f}/{p['ROC-AUC']:.2f})"
    )
gates = report.get("gates") or {}
print("gates:", ", ".join(f"{k}={gates[k]}" for k in sorted(gates)))
print("claim_margin_all_four:", report.get("claim_margin_all_four"))
PY

if [[ "$MODE" != "--seed111-only" ]]; then
  echo ""
  echo "== config recipe parity (gpu_beat_paper vs gpu_beat_paper_3090) =="
  "$PY" - <<'PY'
import sys
from pathlib import Path
import yaml

THROUGHPUT_KEYS = {
    "score_batch_size", "score_batch_size_oom_fallback", "dataloader_num_workers",
    "tiny_n_safe_hw", "tiny_n_safe_threshold", "tiny_n_score_batch_size", "tiny_n_dataloader_num_workers",
    "mid_n_safe_hw", "mid_n_score_batch_size", "mid_n_dataloader_num_workers",
    "huge_n_safe_hw", "huge_n_safe_threshold", "huge_n_score_batch_size", "huge_n_dataloader_num_workers",
    "mcs_pack_masks", "vectorized_masks",
}

def recipe_axion(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ax = dict((cfg.get("axion") or {}))
    for k in THROUGHPUT_KEYS:
        ax.pop(k, None)
    return ax

base = Path("configs/gpu_beat_paper.yaml")
fast = Path("configs/gpu_beat_paper_3090.yaml")
a, b = recipe_axion(base), recipe_axion(fast)
if a != b:
    only_a = {k: a[k] for k in a if a.get(k) != b.get(k)}
    only_b = {k: b[k] for k in b if a.get(k) != b.get(k)}
    print("ERROR: recipe axion block mismatch", file=sys.stderr)
    print("  gpu_beat_paper only:", only_a, file=sys.stderr)
    print("  gpu_beat_paper_3090 only:", only_b, file=sys.stderr)
    raise SystemExit(1)
print("recipe_axion_ok (throughput keys excluded)")

for path in (base, fast):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if cfg.get("beat_paper", {}).get("axion_config") != path.name:
        print(f"WARN: beat_paper.axion_config != {path.name} in {path}")
print("beat_paper_axion_config_ok")
PY

  if [[ "$MODE" == "all" ]]; then
    echo ""
    echo "== embed + map sanity =="
    bash scripts/vast_ceiling_setup.sh
  fi
fi

if [[ "$MODE" != "--config-only" ]]; then
  echo ""
  echo "== seed111 slot parity (ceiling vs run dir) =="
  SEED111_SRC="${SEED111_SRC:-results/axion_beat_paper_dual_lift/metrics}"
  export RUN_ID SEED111_SRC TOL
  "$PY" - <<'PY'
import json
import os
import sys
from pathlib import Path

tol = float(os.environ.get("TOL", "0.01"))
ceiling = Path("results/axion_beat_paper_dual_lift/metrics")
run_dir = Path("results") / os.environ["RUN_ID"] / "metrics"
src = Path(os.environ.get("SEED111_SRC", "results/axion_beat_paper_dual_lift/metrics"))

def pr_roc(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d.get("metrics") or d
    return m.get("PR-AUC"), m.get("ROC-AUC")

bad = []
checked = 0
for cp in sorted(ceiling.glob("*__111__*.json")):
    key = cp.name
    checked += 1
    for label, dp in (("run", run_dir / key), ("src", src / key)):
        if not dp.exists():
            continue
        cpr, croc = pr_roc(cp)
        dpr, droc = pr_roc(dp)
        if cpr is None or dpr is None:
            continue
        if abs(float(cpr) - float(dpr)) > tol or abs(float(croc) - float(droc)) > tol:
            bad.append((key, label, cpr, croc, dpr, droc))

print(f"ceiling_seed111_slots={checked} run_dir={run_dir} ({len(list(run_dir.glob('*__111__*.json')))} files)")
if bad:
    print("ERROR: seed111 parity failures (tol={}):".format(tol), file=sys.stderr)
    for row in bad[:20]:
        print(" ", row, file=sys.stderr)
    raise SystemExit(1)
if len(list(run_dir.glob("*__111__*.json"))) >= 114:
    print("seed111_parity_ok (114 slots within tol)")
else:
    print("seed111_parity_ok (no mismatches among present files; full 114 not in run dir yet)")
PY
fi

echo ""
echo "VERIFY_CEILING_PARITY_OK"
