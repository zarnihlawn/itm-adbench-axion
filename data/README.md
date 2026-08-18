# Local data (not in git)

Beat-Paper runs need ADBench NPZs and whitened alt embeds inside this repo.
Copy once with:

```bash
bash scripts/vendor_data.sh
```

## Expected layout after vendor

| Path | Size (approx) | Source |
|------|---------------|--------|
| `data/adbench/datasets/` | ~2 GB | `../ADBench/adbench/datasets` |
| `data/embeds_alt/` | ~300 MB | `../project/axion/data/embeds_alt` |
| `data/embeds_alt_whitened/` | ~1.6 GB | `../project/axion/data/embeds_alt_whitened` |

Atlas CSVs in this folder **are** tracked in git.

## Override sources

```bash
ADBENCH_SRC=/path/to/adbench/datasets \
EMBEDS_SRC=/path/to/axion/data \
bash scripts/vendor_data.sh
```

## RTX 3090 box

After `git clone` / `git pull`, run `bash scripts/setup_rtx3090.sh` (includes vendor + sanity check).
Or rsync the whole `data/` tree from your laptop:

```bash
rsync -aP laptop:~/ITM/itm-adbench-axion/data/ ./data/
```
