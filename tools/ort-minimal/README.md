# Custom minimal ONNX Runtime Web build

Compiles ORT to WASM with only the operator kernels this model needs. Target: **~3 MB raw / ~1 MB gzipped**, against 12.86 MB / 3.28 MB for the stock build — roughly 2.3 MB off every first load.

Now that the model is fp16 at 2.69 MB gzipped, the runtime is the larger half of the payload again, which is what makes this worth doing. See [docs/model-size.md](../../docs/model-size.md) for the full accounting.

## ⚠️ Read before rebuilding

**The operator config is derived from the model architecture and precision.** `required_operators_and_types.config` lists exactly the kernels the current model uses. It is invalidated by:

- changing the backbone (`mobilenetv3_small_100` → `efficientnet_lite0`)
- **changing precision** (fp16 → fp32, or either → INT8, which adds `ConvInteger`, `DynamicQuantizeLinear`, `MatMulInteger`)
- changing the number of classes only if that alters the graph shape (it does not, in practice)

It is **not** invalidated by retraining with different data or different weights. Weights do not change the graph.

**If you load a model needing an excluded operator, inference fails at session creation** with a "kernel not found" error — not a graceful fallback.

## The fp16 trap

**`--enable_runtime_optimizations` is required, and it is not a tuning knob.** Without it, this build throws away the entire fp16 saving.

ORT's CPU execution provider has no fp16 kernels, so it upcasts every fp16 initialiser to float32 during graph optimisation. On the normal `.onnx` path that happens in memory at session creation — the download stays fp16 and only RAM holds float32. But **`.ort` format serialises the post-optimisation graph**, baking the upcast into the file:

| | Raw | Gzipped |
|---|---|---|
| fp16 `.onnx` (what ships today) | 2.94 MB | **2.69 MB** |
| `.ort`, fixed optimisations | 5.99 MB | **3.45 MB** ← *worse than the fp32 ONNX we replaced* |
| `.ort`, runtime optimisations | 3.11 MB | **2.73 MB** ✅ |

Runtime optimisations defer fusion and the upcast to session creation, keeping fp16 weights on disk. Both variants produce byte-identical logits — this is purely a question of what gets serialised.

Two things must therefore stay in step, or the saving silently evaporates:

1. `--enable_runtime_optimizations` in the Dockerfile.
2. The op config must be the **`.with_runtime_opt.config`** variant, not the plain one. The plain config bakes in `FusedConv` and a float32-only `Conv`; the runtime-opt config keeps `Cast` accepting `MLFloat16`.

## Build

> ### 🚨 This build does not currently run
>
> `--enable_runtime_optimizations` in the Dockerfile **is not a real `build.py` flag** in ORT v1.27.0. The build fails in seconds:
>
> ```
> build.py: error: unrecognized arguments: --enable_runtime_optimizations
> ```
>
> The requirement is real and measured — see [the fp16 trap](#the-fp16-trap-ort-conversion-can-undo-the-model-saving) — but the correct spelling is unknown. **Next step:** read `tools/ci_build/build.py` in the v1.27.0 tree and find the real flag (likely a `--minimal_build` sub-option, or something naming "saving runtime optimizations"), then fix the Dockerfile and rerun.
>
> Everything else here — the op config, the version pin, the `.ort` conversion — is correct and verified. No custom WASM exists yet, so the app still loads the stock 12.86 MB runtime.

```bash
docker build --platform linux/amd64 -t ort-minimal:v1.27.0 tools/ort-minimal
```

**1–3 hours on first run** once the flag is fixed. It compiles emsdk, protobuf and ORT itself. Docker layer caching makes subsequent builds fast unless the config changes — the ORT clone alone takes ~5 minutes and is already cached.

Extract the artefacts:

```bash
id=$(docker create ort-minimal:v1.27.0)
docker cp "$id:/out" ./ort-artifacts
docker rm "$id"
ls -la ort-artifacts/
```

## Using the output

Minimal builds **cannot load `.onnx`** — they need ORT format. Three changes are required together:

**1. Convert the shipping model to `.ort`:**

```bash
.venv/bin/python -m onnxruntime.tools.convert_onnx_models_to_ort \
    runs/mobile-classifier/ --enable_type_reduction --output_dir runs/ort
```

Ship the **`.with_runtime_opt.ort`** file, for the reason above.

**2. Point the web app at the custom WASM.** The stock npm `onnxruntime-web` JS glue expects the stock binary; the custom build ships its own `.mjs` loader. Both must come from the same build — mixing versions fails at runtime with unhelpful errors. Keep `ORT_VERSION` here and `onnxruntime-web` in `web/package.json` on the same version.

**3. Update the service worker precache.** `web/vite.config.js` globs `**/*.wasm`; a custom binary committed outside `dist/assets/` needs its path covered, and `maximumFileSizeToCacheInBytes` must clear its uncompressed size. Workbox drops oversized files with only a build-log warning — see the git history of that file for how this broke offline use once already.

## Regenerating the operator config

After any architecture or precision change:

```bash
.venv/bin/python -m onnxruntime.tools.convert_onnx_models_to_ort \
    <dir-with-model.onnx> --enable_type_reduction --output_dir /tmp/ort_cfg
cp /tmp/ort_cfg/required_operators_and_types.with_runtime_opt.config \
   tools/ort-minimal/required_operators_and_types.config
docker build --platform linux/amd64 -t ort-minimal:v1.27.0 tools/ort-minimal
```

Note the `.with_runtime_opt` variant. Copying the plain config is the single easiest way to silently lose the fp16 saving.

## Current config

Generated from the fp16 graph of `mobilenetv3_small_100`, 4 classes:

```
ai.onnx;1;GlobalAveragePool
ai.onnx;6;HardSigmoid
ai.onnx;11;Conv
ai.onnx;13;Cast{MLFloat16, float},Flatten,Gemm,ReduceMean
ai.onnx;14;Add,Mul,Relu
```

Five operator groups with per-operator type restrictions, against several hundred kernels across all supported types in the stock build. `HardSwish` does not appear because ORT decomposes it into `HardSigmoid` + `Mul`.

## Is this worth it?

**It buys ~2.3 MB gzipped per first load**, and the model is now settled at fp16, which was the condition for doing it:

- **Worth it** if users are on metered or slow rural connections — this project's situation.
- **Not worth it** while the architecture is in flux. Every experiment needs a 1–3 hour rebuild, and a stale config produces a runtime that silently cannot load the new model.

The one-line `onnxruntime-web/wasm` import already captured the larger share of the available saving. This is the second-order optimisation.
