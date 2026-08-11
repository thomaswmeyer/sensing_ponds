# Assets not yet present

The app builds and runs without these, but degrades in specific ways. Listed in
the order they block real use.

## `model/plants.onnx` + `model/model_config.json`

**Blocks: all identification.** Produced by training:

```bash
.venv/bin/python src/train_mobile_classifier.py \
    --data-root data/mendeley data/gbif --export-tflite
cp runs/mobile-classifier/mobilenetv3_small_plants.onnx web/public/model/plants.onnx
cp runs/mobile-classifier/model_config.json             web/public/model/
```

`model_config.json` carries the preprocessing parameters, the fitted temperature
and the abstain threshold. The client reads all three from it rather than
hardcoding them, so retraining cannot silently desynchronise the two — but it
also means the app cannot run without it.

Until then `loadModel()` rejects and the UI shows `error.model`.

## `fonts/NotoSansTamil-Regular.woff2`, `fonts/NotoSansTamil-Bold.woff2`

**Blocks: reliable Tamil rendering.** Without them the app falls back to whatever
the device has. Android coverage for Tamil is inconsistent and the failure mode
is tofu — empty boxes where the text should be.

Download from [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+Tamil)
and subset **by the full Tamil Unicode block**, not by observed character
frequency: aggressive subsetting strips combining marks in Indic scripts and
produces broken clusters.

## `audio/ta/*.opus`

**Blocks: use by non-literate users.** Roughly 60–100 fixed strings, recorded
with a native Tamil speaker.

This is the primary voice path, not a fallback. Chrome on Android silently
substitutes an English voice when the Tamil voice pack is absent, which would
read Tamil aloud in English phonetics to someone who cannot see that it is
wrong.

File names match string IDs: `audio/ta/species.water_hyacinth.opus`. Register
each recorded ID in `RECORDED_AUDIO` in `src/i18n/strings.js` — an unregistered
file is never played, and a registered file that does not exist is a silent 404
in the field.

## `icons/icon-192.png`, `icons/icon-512.png`, `icons/icon-512-maskable.png`

**Blocks: installability.** The PWA manifest references them; without them the
install prompt will not appear on Android.

The maskable variant needs its content inside the safe zone (centre 80%) or
Android will crop it.
