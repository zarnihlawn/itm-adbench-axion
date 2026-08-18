#!/usr/bin/env bash
# Smoke claim-matching scores on the four dual_lift gap datasets (zoo only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src:scripts"
export AXION_VERBOSE=1 AXION_FORCE_PROGRESS=1
PY="${PY:-python3}"

MAP="results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json"
RUN_ID="${RUN_ID:-axion_beat_paper_parity_smoke}"
CFG="${CFG:-configs/gpu_beat_paper.yaml}"

"$PY" -u scripts/beat_paper_run.py \
  --config "$CFG" \
  --map "$MAP" \
  --seeds 111 \
  --settings unsupervised,semi-supervised \
  --datasets CIFAR10 FashionMNIST MNIST-C Amazon \
  --run-id "$RUN_ID" \
  --skip-axion

"$PY" - <<PY
import json
from pathlib import Path

claim = Path("results/axion_beat_paper_dual_lift/metrics")
smoke = Path("results/$RUN_ID/metrics")
pairs = [
    ("CIFAR10", "unsupervised"),
    ("FashionMNIST", "semi-supervised"),
    ("FashionMNIST", "unsupervised"),
    ("MNIST-C", "semi-supervised"),
    ("MNIST-C", "unsupervised"),
    ("Amazon", "semi-supervised"),
    ("Amazon", "unsupervised"),
]
print(f"{'dataset':14} {'setting':16} {'claimPR':8} {'smokePR':8} {'dPR':7} {'claimROC':8} {'smokeROC':8} {'nc':4}")
ok = True
for ds, st in pairs:
    cp = claim / f"{ds}__{st}__111__beat_paper.json"
    sp = smoke / f"{ds}__{st}__111__beat_paper.json"
    c = json.loads(cp.read_text())
    s = json.loads(sp.read_text())
    cm, sm = c["metrics"], s["metrics"]
    dpr = sm["PR-AUC"] - cm["PR-AUC"]
    nc = (s.get("extra") or {}).get("n_components")
    print(
        f"{ds:14} {st:16} {cm['PR-AUC']:8.2f} {sm['PR-AUC']:8.2f} {dpr:7.2f} "
        f"{cm['ROC-AUC']:8.2f} {sm['ROC-AUC']:8.2f} {str(nc):4}"
    )
    # Within a few PR of claim on the big gaps (allow ~5 PR noise on filled_from).
    if abs(dpr) > 8.0:
        ok = False
print("PASS" if ok else "CHECK (some |dPR|>8; GPU full run may still close AXION locks)")
Path(f"results/$RUN_ID/thesis").mkdir(parents=True, exist_ok=True)
Path(f"results/$RUN_ID/thesis/PARITY_SMOKE.md").write_text(
    "See smoke_parity_gap4.sh stdout for claim vs smoke table.\\n", encoding="utf-8"
)
PY
