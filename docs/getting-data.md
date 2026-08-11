# Getting the Data

None of the datasets are committed — they total ~2.5 GB and carry their own licences. This document reproduces them from scratch.

See [datasets.md](datasets.md) for what each dataset contains and why it was chosen.

## Prerequisites

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Verify — MPS matters for training speed on Apple Silicon:

```bash
.venv/bin/python -c "import torch, timm; print(torch.__version__, timm.__version__, torch.backends.mps.is_available())"
```

### ⚠️ If you are behind a TLS-inspecting proxy

On a corporate machine (Zscaler, Netskope, etc.), Python's HTTPS fails with `CERTIFICATE_VERIFY_FAILED` because it uses its own certificate bundle rather than the OS trust store. `curl` works because it uses the system keychain.

[`src/fetch_gbif.py`](../src/fetch_gbif.py) already routes all HTTPS through `curl` for this reason. If you hit the same error elsewhere, ask IT for the sanctioned `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` setup — **do not** modify the system certificate store yourself.

## 1. WaterHyacinth (Mendeley) — manual download

No API; download through the browser.

1. Open <https://data.mendeley.com/datasets/vz6z64nwby/1>
2. Download the archive and unzip to `data/mendeley/`

Expected layout:

```
data/mendeley/
└── Original Images/
    ├── Common Water Hyacinth (Eichornia crassipes)/   470 images
    ├── Water Lettuce (Pistia stratiotes)/             480 images
    ├── Common Duckweeds (Lemna minor)/                390 images
    └── Heartleaf False Pickerelweed (Monochoria korsakowii)/  450 images
```

Verify:

```bash
find data/mendeley -name '*.jpg' | wc -l    # expect 1790
```

**Use "Original Images" only.** The dataset also ships ~4,050 pre-augmented copies. Static augmentation is worse than on-the-fly, and augmented copies of a training image landing in the test split invalidate the metrics. The training script filters them by filename, but not downloading them is cleaner.

**⚠️ The Monochoria class is not used.** *M. korsakowii* is temperate East Asian with zero GBIF records in India or Sri Lanka, and its 450 images come from only 2 capture days. It is replaced by *Salvinia molesta*. See [datasets.md](datasets.md).

### Known limitation

1,790 images sounds like plenty. Measured by filename timestamp, they come from **10 capture days**:

| Date | Images | Classes present |
|---|---|---|
| 2023-08-03 → 08-05 | 203 | hyacinth only |
| 2023-08-11 | 362 | duckweed, hyacinth, lettuce |
| 2023-08-12 → 08-14 | 240 | duckweed, hyacinth, lettuce |
| 2023-08-18 | 195 | lettuce only |
| 2023-09-23 | 532 | duckweed, Monochoria, lettuce |
| 2023-09-26 | 129 | Monochoria only |

This is why a random train/test split reports ~98% and means nothing — it measures whether the model recognises a particular afternoon's pond. See [the split strategy](#how-the-data-is-split) below.

## 2. GBIF — scripted

```bash
.venv/bin/python src/fetch_gbif.py --dry-run                    # counts only
.venv/bin/python src/fetch_gbif.py --out data/gbif --per-species 900
```

Downloads citizen-science observation photos, prioritising India and Sri Lanka (the deployment region) and topping up globally. Roughly 900 MB and 20–40 minutes.

Writes `data/gbif/manifest.csv` with licence, attribution, country, coordinates and GBIF key for every image. **Keep it** — CC BY requires attribution, so the manifest is a compliance record, not a convenience.

### Filters applied

| Filter | Why |
|---|---|
| `license=CC0_1_0,CC_BY_4_0` | Safe for a distributed product. NC blocks commercial use; ND blocks derivatives. |
| `mediaType=StillImage` | Occurrence records without photos are useless here |
| `basisOfRecord=HUMAN_OBSERVATION` | Excludes `PRESERVED_SPECIMEN` — herbarium sheets, which look nothing like a field photo |
| `hasCoordinate=true` | Needed to identify regional records |

`--any-licence` widens to NC/ND. **That is a legal decision, not a technical one** — do not use it for anything shipped.

### Realistic yields

Counts after all filters (checked 2026-08-11):

| Class | Global | Regional (IN + LK) |
|---|---|---|
| water_hyacinth | 3,538 | 527 |
| duckweed | 2,335 | 11 |
| water_lettuce | 1,660 | 196 |
| salvinia | 428 | 24 |

Note how much the filters cost: *Pontederia crassipes* has ~22,500 imaged occurrences, but only 3,538 survive the licence and basis-of-record filters. `--dry-run` reports the honest number.

**Duckweed's 11 regional images are a known weakness.** Its test metrics will be unreliable, and the training script warns about it.

## 3. PPCC water quality (satellite track)

Unrelated to the classifier — in-situ measurements for the Sentinel-2 pipeline.

```bash
.venv/bin/python scripts/fetch_ppcc.py --years 2023 2024 2025 2026
```

Scrapes monthly PDFs from the Puducherry PCC into `data/ppcc_surface_water.csv`, keyed by `(station, year, month)` to join against the Earth Engine export in [`gee/pondy_water.js`](../gee/pondy_water.js). The source site is slow and drops connections; downloads are cached and retried, so re-running only fetches what is missing.

The CSV **is** committed (it is small and is a modelling input); the source PDFs are not.

## How the data is split

Not a random split, and not a conventional group split:

```
train  ├─ Mendeley (all 10 capture days)
       └─ GBIF global records

val    └─ group-held-out slice of the above (checkpoint selection only)

test   └─ GBIF regional records (India + Sri Lanka) — held out entirely
```

**The test set is regional GBIF records only.** Any split of the Mendeley data measures recognition of ten specific Bangladeshi afternoons. Holding out regional records instead asks the question that matters: does a model trained on Bangladeshi and global photos work on South Indian and Sri Lankan water bodies?

It is a much harder test. **Expect a substantially lower number than a random split would report — that gap is the finding, not a problem to tune away.**

Temperature calibration is fitted on validation, never on test — fitting it on the reported set would leak.

## Training

```bash
.venv/bin/python src/train_mobile_classifier.py \
    --data-root data/mendeley data/gbif \
    --export-tflite
```

`--data-root` takes multiple roots and merges them into one label space. Mendeley's long directory names (`Common Water Hyacinth (Eichornia crassipes)` — note the misspelling in the original) map to the same canonical classes as GBIF's slugs.

Outputs to `runs/mobile-classifier/`:

| File | Purpose |
|---|---|
| `mobilenetv3_small_plants.pt` | Checkpoint |
| `metrics.json` | Out-of-domain accuracy, ECE, abstain threshold |
| `mobilenetv3_small_plants.onnx` | For browser / mobile deployment |
| `model_config.json` | **Preprocessing + temperature + abstain threshold** |
| `labels.txt` | Class names |

**`model_config.json` is required by the client.** Without the temperature and threshold, the app ships raw overconfident softmax and can never say "not sure" — the exact failure the abstain path exists to prevent.

## Reproducing from nothing

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 1. Manual: download Mendeley -> data/mendeley/
# 2. Scripted:
.venv/bin/python src/fetch_gbif.py --out data/gbif --per-species 900
.venv/bin/python scripts/fetch_ppcc.py --years 2023 2024 2025 2026

.venv/bin/python src/train_mobile_classifier.py \
    --data-root data/mendeley data/gbif --export-tflite
```

## Licences

| Dataset | Licence | Obligation |
|---|---|---|
| WaterHyacinth (Mendeley) | CC BY 4.0 | Cite [Data in Brief, 2023](https://www.sciencedirect.com/science/article/pii/S2352340923009320) |
| GBIF occurrences | CC0 / CC BY 4.0 | Attribute per-image via `manifest.csv` |
| AqUavplant (not yet used) | CC BY 4.0 | Cite [Scientific Data, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11661991/) |
| PPCC | Indian government publication | Cite the source |

If a trained model is redistributed, ship `manifest.csv` with it. A model trained on CC-BY images is plausibly a derivative work, and attribution is the condition of use.
