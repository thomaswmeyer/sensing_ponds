# sensing_pondy

Remote sensing and computer vision for water hyacinth coverage and water quality monitoring.

## What this is

A Tamil-first mobile web app that identifies floating aquatic plants on-device from a phone photo, tells the user what the plant can be used for, and contributes each observation — photo, GPS, timestamp, prediction — to a growing ground-truth dataset for later human validation.

Classifier training is implemented in [src/train_mobile_classifier.py](src/train_mobile_classifier.py). The capture app is designed in [docs/architecture.md](docs/architecture.md) but not yet built.

**Classes:** water hyacinth, water lettuce, duckweed, *Salvinia molesta* — plus an explicit *abstain* output for anything else.

> **Scope note.** This project previously included a Sentinel-2 satellite segmentation track (coverage mapping, water-quality regression). That work has been removed to focus on the mobile classifier and may be reintegrated later. The capture app still records GPS accuracy and mat extent, which exist only to make observations usable as satellite labels. Removed code is recoverable from git history at `219c80e`.

## Documentation

| Document | Contents |
|---|---|
| [docs/getting-data.md](docs/getting-data.md) | **Start here.** Environment setup, dataset download, training |
| [docs/classifier-options.md](docs/classifier-options.md) | Model choice, augmentation, evaluation discipline |
| [docs/datasets.md](docs/datasets.md) | Public datasets: WaterHyacinth, AqUavplant, iNaturalist/GBIF |
| [docs/architecture.md](docs/architecture.md) | Field capture PWA + Node.js API + human validation loop |

## Quick start

Full instructions in [docs/getting-data.md](docs/getting-data.md).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 1. Manually download WaterHyacinth -> data/mendeley/
#    https://data.mendeley.com/datasets/vz6z64nwby/1
# 2. Fetch geographic diversity from GBIF:
.venv/bin/python src/fetch_gbif.py --out data/gbif --per-species 1200

.venv/bin/python src/train_mobile_classifier.py \
    --data-root data/mendeley data/gbif --export-tflite
```

Outputs to `runs/mobile-classifier/`: checkpoint, `metrics.json`, ONNX, `labels.txt`, and `model_config.json` (preprocessing + calibration + abstain threshold — the client needs all three).

### Two things that will mislead you

**The test set is GBIF regional records, held out entirely.** The Mendeley dataset has 1,790 images but only **10 capture days**, so any split of it measures whether the model recognises ten specific Bangladeshi afternoons. Training uses Mendeley plus global GBIF; testing uses India/Sri Lanka records only. This reports a much lower number than a random split — that gap is the finding, not a bug.

**Abstain is a real output.** A four-way softmax always sums to 1 and cannot say "none of these". Temperature scaling is fitted on validation, a confidence threshold is chosen for target precision, and both ship in `model_config.json`. If no threshold reaches target precision, the script says so — don't ship an auto-ID path on that model.

## Status

**Not yet run.** The environment is set up (torch 2.13, timm 1.0.28, MPS available) and the Mendeley data is in place, but training has not been executed and the GBIF download is incomplete. The reworked split and discovery code is syntax-checked, not tested.

The web client is scaffolding only — `web/package.json` and nothing else.

Everything in `docs/` is a design document. No accuracy figure in this repo has been measured.
