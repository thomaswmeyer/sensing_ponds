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
| `license=CC0_1_0,CC_BY_4_0,CC_BY_NC_4_0` | **This is a non-commercial, open-source project, so NC is usable.** ND is still excluded — a model trained on an image is plausibly a derivative work. |
| `mediaType=StillImage` | Occurrence records without photos are useless here |
| `basisOfRecord=HUMAN_OBSERVATION` | Excludes `PRESERVED_SPECIMEN` — herbarium sheets, which look nothing like a field photo |
| `hasCoordinate=true` | Needed to identify regional records |

**Including NC is worth roughly 5× the data**, and regional counts triple — which matters more, since regional records are the held-out test set:

| Class | Global (CC0+BY) | Global (+NC) | Regional (CC0+BY) | Regional (+NC) |
|---|---|---|---|---|
| water_hyacinth | 3,538 | **19,352** | 527 | **1,584** |
| duckweed | 2,335 | **11,070** | 11 | **23** |
| water_lettuce | 1,660 | **9,190** | 196 | **569** |
| salvinia | 428 | **2,041** | 24 | **93** |

`--commercial-safe` drops back to CC0 + CC-BY, costing ~80% of the data. **Use it if this project ever takes on a commercial dimension** — retraining from a narrower pool is far cheaper than discovering the constraint after release.

`--any-licence` removes the filter entirely, including ND. Not recommended.

### Known weaknesses

**Duckweed has only 23 regional images** even with NC included. Its test metrics will be unreliable, and the training script warns about it.

**The `basisOfRecord` filter is expensive in South Asia.** *Monochoria vaginalis* drops from 1,281 records to 6 — nearly all its regional records are herbarium specimens. Licensing is not the constraint there; the filter is. This is why *Monochoria* was dropped as a class entirely rather than substituted.

### Re-running is incremental

Images already on disk are not re-downloaded, and existing `manifest.csv` rows are carried forward as long as their file still exists. A class already at its target is skipped.

⚠️ **Do not delete `manifest.csv` while keeping the images.** Attribution data cannot be reconstructed from filenames — the images become unusable for redistribution. If it is ever lost, delete the images too and re-fetch.

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
.venv/bin/python src/fetch_gbif.py --out data/gbif --per-species 1200

.venv/bin/python src/train_mobile_classifier.py \
    --data-root data/mendeley data/gbif --export-tflite
```

## Licences

| Dataset | Licence | Obligation |
|---|---|---|
| WaterHyacinth (Mendeley) | CC BY 4.0 | Cite [Data in Brief, 2023](https://www.sciencedirect.com/science/article/pii/S2352340923009320) |
| GBIF occurrences | CC0 / CC BY / **CC BY-NC** | Attribute per-image via `manifest.csv` |
| AqUavplant (not yet used) | CC BY 4.0 | Cite [Scientific Data, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11661991/) |

**The training data includes CC-BY-NC images, so the resulting model is non-commercial.** That is a deliberate choice for this open-source, crowdsourced project — it buys ~5× the data. If the project ever needs a commercially usable model, re-fetch with `--commercial-safe` and retrain.

If a trained model is redistributed, ship `manifest.csv` with it. A model trained on CC-BY images is plausibly a derivative work, and attribution is the condition of use.
