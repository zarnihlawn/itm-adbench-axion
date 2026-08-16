# AXION on Vast.ai

Lives inside the git repo at `ITM/project/axion/` (push/pull with `project`).

Layout on the instance:

```
/workspace/ITM/          # or /data/ITM/
  ADBench/adbench/datasets/
  project/               # git clone / pull
    axion/               # this package
```

## One-time setup

```bash
cd /workspace/ITM/project
git pull
cd axion
bash scripts/vast_setup.sh
```

## Ultimate final (ALL seeds, Table-1 fair)

Canonical sheet: [`FINAL_RUN.md`](FINAL_RUN.md). Frozen config: `configs/gpu_final.yaml` (edge_rare recipe).

Hardware for current rental (instance **47448156**): RTX **A4000 16GB** / ~8 CPU / ~64 GiB (`202.122.49.242`).
`run_final.sh` auto-sets `OMP=nproc-2` (expect ~6); YAML uses `score_batch=12288`, `workers=2`.

```bash
cd /data/ITM/project/axion
source .venv/bin/activate
python scripts/assert_final_config.py --config configs/gpu_final.yaml
bash scripts/run_final.sh all    # 12-probe ×5 → soft gate → full57 ×5 (570 jobs)
```

Do **not** point final at smoke / gap-close YAMLs.

## Runs (GPU) — Phase 5 (dual-track / legacy design)

```bash
cd /workspace/ITM/project/axion
source .venv/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1

# Preferred: embed → classical → 12-ds → Track C gate
bash scripts/vast_probe.sh g5-auto configs/gpu.yaml

# Step-by-step:
bash scripts/vast_probe.sh g5-embed configs/gpu.yaml
bash scripts/vast_probe.sh g5-classical configs/gpu.yaml
bash scripts/vast_probe.sh g5-seed111 configs/gpu.yaml   # fast confirm
bash scripts/vast_probe.sh g5 configs/gpu.yaml            # 3 seeds
bash scripts/vast_probe.sh full57 configs/gpu.yaml        # needs Track A+B+C
bash scripts/vast_probe.sh final configs/gpu_final.yaml  # alias → run_final.sh all
```

## Gates (dual-track)

| Track | Script | Pass rule |
|-------|--------|-----------|
| **A classical** | `check_classical_gate.py` | classical-8 semi PR ≥ 63.36 + unsup `pass_probe_margin` |
| **B embed** | `check_embed_gate.py` | CV/NLP-4 semi mean PR ≥ 27.5 + Imdb semi ROC ≥ 50.5 |
| **C probe** | `check_probe_gate.py` | 12-ds **unsup** `pass_probe_margin` + semi **ROC** ≥ paper+1 |

- `AXION_EMBED_GATE_STRICT=1` → aspirational embed 45 / 60  
- `AXION_PROBE_SEMI_STRICT=1` → also require 12-ds semi PR margin (unreachable on ADBench BERT/ResNet; research only)

**Why dual-track C:** 12-ds semi PR ≥63.36 needs CV/NLP mean ~60; geometry+AdaDDAE+AXION ceiling is ~29. See `results/axion_g5_embed/g5_diagnosis.md`.

Paper macros (reporting): unsup **32.77 / 74.08**, semi **61.36 / 83.17**. Ship margins = paper+2 PR / paper+1 ROC where applicable.

## Phase 4 alternate embeds (research only)

ADBench also ships **CV_by_ViT** + **NLP_by_RoBERTa**. Wire them without claiming official Table-1 parity:

```bash
# On laptop or Vast (needs ViT + RoBERTa NPZs under ADBench/datasets)
python scripts/extract_alt_embeds.py --mode adbench \
  --adbench-root /data/ITM/ADBench/adbench/datasets \
  --config-out configs/gpu_alt_embeds.yaml

bash scripts/vast_probe.sh g5-embed-alt configs/gpu_alt_embeds.yaml
bash scripts/vast_probe.sh g5-seed111-alt configs/gpu_alt_embeds.yaml
```

Optional Track-B re-extract (ViT-B/16 + MiniLM or E5):

```bash
python scripts/extract_alt_embeds.py --mode extract-track-b \
  --nlp-model minilm --use-dest-as-root \
  --adbench-root /data/ITM/ADBench/adbench/datasets \
  --dest data/embeds_alt --config-out configs/gpu_alt_minilm.yaml
# --nlp-model e5 → configs/gpu_alt_e5.yaml
```

Official ship path remains `configs/gpu.yaml` (ResNet18 + BERT).

## All-out classical push (Table-1 fair method)

After syncing latest `configs/gpu.yaml` (true ensemble, learned bank weights, cover/fraud latch, **SPEAR**):

```bash
bash scripts/vast_probe.sh g5-classical configs/gpu.yaml
bash scripts/vast_probe.sh g5-embed configs/gpu.yaml
bash scripts/vast_probe.sh g5-seed111 configs/gpu.yaml
```

Targeted smoke first (SPEAR classical holes):

```bash
python scripts/run_probe.py --config configs/gpu.yaml --model axion \
  --datasets glass cover fraud cardio breastw \
  --settings semi-supervised --seeds 111 \
  --run-id axion_spear_p1 --loop-log
```

Disable SPEAR for A/B control: set `spear: false` in config or pass model without the flag.
See `results/axion_spear/diagnosis.md`.

## Sync results back (laptop)

```bash
rsync -avz -e "ssh -p PORT" \
  root@HOST:/data/ITM/project/axion/results/ \
  /home/zarnihlawn/Desktop/ITM/project/axion/results/
```

## Local note

- Do not mix with archived DDAE-PAR under `ITM/archive/` or old AdaDDAE `results/adadae_*`
- ADBench path resolves to `ITM/ADBench/...` automatically from `project/axion`
- Gap-fix / dual-track: `results/axion_g5/g5_diagnosis.md`
