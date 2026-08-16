#!/usr/bin/env bash
# Local leap smoke: catalog assert, unit tests, tiny leap probes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(pwd)/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
RUN_ID="${RUN_ID:-axion_beat_paper_leap_smoke}"
mkdir -p "results/${RUN_ID}"

echo "== catalog assert =="
"$PY" scripts/beat_paper_catalog.py --assert-catalog

echo "== unit tests =="
"$PY" -m pytest tests/test_beat_paper_leap.py -q --tb=short

echo "== smoke leap campaign (wine/Wilt) =="
"$PY" -u scripts/beat_paper_leap_campaign.py --smoke

echo "== smoke probe new recipes =="
"$PY" -u - <<'PY'
import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from beat_paper_leap_campaign import run_probe
for ds, rec in (
    ("wine", "fusion_ecod_knn_native"),
    ("wine", "elliptical_native"),
    ("Wilt", "pca_residual_knn_native"),
):
    hit = run_probe(ds, "unsupervised", rec)
    assert not hit.get("error"), hit
    print(f"ok {ds}/{rec} PR={hit['PR-AUC']:.2f} ROC={hit['ROC-AUC']:.2f}")
print("smoke probes ok")
PY

echo "DONE leap smoke → results/${RUN_ID}"
