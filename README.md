# Sensing Ponds

Remote sensing and computer vision for water hyacinth coverage and water quality monitoring.

## What this is

A Tamil-first mobile web app that identifies floating aquatic plants on-device from a phone photo, tells the user what the plant can be used for, and contributes each observation — photo, GPS, timestamp, prediction — to a growing ground-truth dataset for later human validation.

Classifier training is implemented in [src/train_mobile_classifier.py](src/train_mobile_classifier.py). The capture app is built and deployed; the upload API and validation loop in [docs/architecture.md](docs/architecture.md) are still design documents.

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

**Trained and deployed** to <https://sensing-ponds.pages.dev>.

88.3% accuracy / 87.8% balanced on the 920-image held-out test set, per-class F1 0.85–0.91, ECE 0.018 after temperature scaling. The abstain threshold of 0.72 reaches the 95% precision target on 85.2% of images; the rest return "not sure". Ships as fp16 at 2.94 MB — INT8 was measured and abandoned, see [docs/model-size.md](docs/model-size.md).

What is *not* done:

- **No field testing.** Every number above comes from citizen-science photographs. None comes from a phone held over a real pond, which is the only test that decides whether this works. The regional subset scores 94.7% — *above* the headline — which says those GBIF images are easier, not that the model transfers.
- **No Tamil audio.** `RECORDED_AUDIO` is empty. The voice path is the primary interface for non-literate users, not a fallback — see [web/ASSETS-NEEDED.md](web/ASSETS-NEEDED.md).
- **No backend.** Observations queue in IndexedDB via the outbox and are never uploaded, because there is nowhere to upload them. `COLLECT_POSITION` is `false` for the same reason: the app does not ask for a location it cannot yet do anything with.
