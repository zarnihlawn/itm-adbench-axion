# Local data (not in git)

Beat-Paper runs need **official ADBench NPZs** and **whitened alt embeds** (ViT/E5; not in upstream ADBench).

## Code vs dataset paths

AXION **code** lives in the repo root (`itm-adbench-axion/`). Datasets are not committed.

| What | Path from repo root | YAML key |
|------|---------------------|----------|
| AXION code | `.` (`src/axion/`, `scripts/`, `configs/`) | n/a |
| Official ADBench NPZs | `data/adbench/datasets` (b) or `../ADBench/adbench/datasets` (a) | `paths.adbench_root` |
| Unwhitened alt embeds | `data/embeds_alt` | `paths.embeds_alt_root` |
| Whitened ViT + E5 | `data/embeds_alt_whitened` | `paths.embeds_alt_whitened_root` |

Configs `configs/gpu_beat_paper.yaml` and `configs/gpu_beat_paper_3090.yaml` set:

```text
adbench_root: data/adbench/datasets
embeds_alt_root: data/embeds_alt
embeds_alt_whitened_root: data/embeds_alt_whitened
```

```text
ITM/
├── itm-adbench-axion/                 # code
│   ├── src/axion/paths.py             # DEFAULT_ADBENCH_DATASETS
│   ├── configs/gpu_beat_paper*.yaml   # paths.* above
│   └── data/
│       ├── adbench/datasets/          # YAML target: official NPZs
│       │   ├── Classical/
│       │   ├── CV_by_ResNet18/
│       │   ├── CV_by_ViT/
│       │   ├── NLP_by_BERT/
│       │   └── NLP_by_RoBERTa/
│       ├── embeds_alt/
│       └── embeds_alt_whitened/
└── ADBench/                           # official clone (a)
    └── adbench/datasets/              # same five folders
```

- **(a)** sibling: `LINK_IN_REPO=1 bash scripts/clone_adbench.sh` creates `data/adbench/datasets` -> `../ADBench/adbench/datasets`
- **(b)** in-repo: copy, or `ADBENCH_CLONE_DIR=./ADBench LINK_IN_REPO=1 bash scripts/clone_adbench.sh`

Fallback order in `src/axion/paths.py`: env `ADBENCH_DATASETS` / `ADBENCH_ROOT`, then `data/adbench/datasets`, `./ADBench/adbench/datasets`, `../ADBench/adbench/datasets`.

## Fair path (recommended for RTX 3090 / fresh hosts)

Clone official [Minqi824/ADBench](https://github.com/Minqi824/ADBench) and download NPZs from the upstream manifest. Do **not** rsync a laptop ADBench tree.

```bash
# Option A: sibling ../ADBench (classic ITM layout)
LINK_IN_REPO=1 bash scripts/clone_adbench.sh

# Option B: clone inside this repo
ADBENCH_CLONE_DIR=./ADBench LINK_IN_REPO=1 bash scripts/clone_adbench.sh

# Whitened embeds only (from project/axion or rsync embeds from laptop)
SKIP_ADBENCH=1 EMBEDS_SRC=/path/to/axion/data bash scripts/vendor_data.sh
```

Or one shot: `bash scripts/setup_rtx3090.sh` (clone + embeds vendor by default).

| Path | Size (approx) | Source |
|------|---------------|--------|
| `data/adbench/datasets/` (symlink ok) | ~2 GB | Official ADBench download via `clone_adbench.sh` |
| `data/embeds_alt/` | ~300 MB | `project/axion/data` or extract from official ViT/RoBERTa NPZs |
| `data/embeds_alt_whitened/` | ~1.6 GB | `project/axion/data` (not in official ADBench) |

Atlas CSVs in this folder **are** tracked in git.

## Legacy: rsync laptop copy

```bash
VENDOR_ADBENCH=1 bash scripts/vendor_data.sh
# or rsync whole data/ tree:
rsync -aP laptop:~/ITM/itm-adbench-axion/data/ ./data/
```

## Override sources

```bash
ADBENCH_DATASETS=/path/to/adbench/datasets \
EMBEDS_SRC=/path/to/axion/data \
SKIP_ADBENCH=1 bash scripts/vendor_data.sh
```

## ADBench layout options

| Layout | `ADBENCH_CLONE_DIR` | Config `adbench_root` |
|--------|---------------------|------------------------|
| Sibling (ITM) | `../ADBench` | `data/adbench/datasets` symlink or `../ADBench/adbench/datasets` |
| In-repo clone | `./ADBench` | `data/adbench/datasets` symlink (`LINK_IN_REPO=1`) |
| Vendored copy | n/a | `data/adbench/datasets` (real dir from `VENDOR_ADBENCH=1`) |

Outside mainland China use default `ADBENCH_DL_REPO=github`. On campus CN hosts try `ADBENCH_DL_REPO=jihulab`. `clone_adbench.sh` uses HTTP/1.1 for git clone/pull **for that command only** (not global git config), retries clone 3 times, and retries each NPZ 3 times.

If clone already failed mid-way (`curl 92 HTTP/2 ... CANCEL`):

```bash
rm -rf ../ADBench
git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/Minqi824/ADBench.git ../ADBench
ADBENCH_DL_REPO=jihulab LINK_IN_REPO=1 bash scripts/clone_adbench.sh
```
