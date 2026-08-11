# sensing_pondy

Remote sensing and computer vision for water hyacinth coverage and water quality monitoring.

## Two tracks

**1. Satellite coverage mapping** — Sentinel-2 semantic segmentation for hyacinth coverage area and change over time. Design only so far; no imagery source confirmed. See [docs/classifier-options.md](docs/classifier-options.md).

**2. Mobile field identification** — on-device species ID from a phone photo, for ground validation. Classifier training implemented in [src/train_mobile_classifier.py](src/train_mobile_classifier.py); the capture app is designed in [docs/architecture.md](docs/architecture.md) but not built.

The tracks connect through geocoded ground-truth photos: field observations with GPS become training labels for the satellite model. **Capture GPS accuracy, timestamp, and rough mat extent from day one** — retrofitting location onto existing photos is impossible.

## Documentation

| Document | Contents |
|---|---|
| [docs/classifier-options.md](docs/classifier-options.md) | Model choices for both tracks, validated against published literature |
| [docs/datasets.md](docs/datasets.md) | Public datasets: WaterHyacinth, AqUavplant, iNaturalist/GBIF |
| [docs/architecture.md](docs/architecture.md) | Field capture PWA + Node.js API + human validation loop |

## Mobile classifier

Fine-tunes MobileNetV3-Small (2.5M params, ~2 MB INT8) on four floating aquatic plant classes: water hyacinth, water lettuce, duckweed, *Monochoria*.

```bash
pip install -r requirements.txt

# Download WaterHyacinth from https://data.mendeley.com/datasets/vz6z64nwby/1
# Expected layout: data/WaterHyacinth/<class-name>/*.jpg

python src/train_mobile_classifier.py --data-root data/WaterHyacinth
python src/train_mobile_classifier.py --data-root data/WaterHyacinth --export-tflite
```

Outputs to `runs/mobile-classifier/`: checkpoint, `metrics.json`, and (with `--export-tflite`) ONNX + `labels.txt` for TFLite conversion.

### Two things that will mislead you

**Group-aware splitting is on by default.** The dataset is two Bangladeshi districts over three months. A random split puts near-duplicate same-pond photos in both train and test, and reports ~98% accuracy that measures memorised ponds. `--group-mode infer` (default) keeps capture sessions whole and reports an honest, lower number. `--group-mode none` gives the flattering one and prints a warning.

Grouping is inferred from filename patterns since the dataset ships no session metadata — imperfect, but better than random. If you have real provenance, use it.

**The script reports calibration and an abstain threshold, not just accuracy.** A four-class softmax will confidently label a photo of a rock. If the model can't reach the target precision at any confidence threshold, it says so — don't ship an auto-ID path on that.

## Status

Nothing here has been run. The training script is syntactically valid but untested — no dependencies are installed in this environment and the WaterHyacinth dataset has not been downloaded. Expect to debug the first run, particularly `infer_group()`, whose filename heuristics are guesses about a dataset layout not yet inspected.

Everything in `docs/` is a design document, not a validated result.
