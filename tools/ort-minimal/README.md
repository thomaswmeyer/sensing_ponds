# Custom minimal ONNX Runtime Web build

Compiles ORT to WASM with only the operator kernels this model needs. Target: **~3 MB raw / ~1 MB gzipped**, against 13.5 MB / 3.5 MB for the stock build — roughly 2.5 MB off every first load.

See [docs/model-size.md](../../docs/model-size.md) for the full size accounting.

## ⚠️ Read before rebuilding

**The operator config is derived from the model architecture.** `required_operators_and_types.config` lists exactly the kernels the current model uses. It is invalidated by:

- changing the backbone (`mobilenetv3_small_100` → `efficientnet_lite0`)
- changing quantisation (the INT8 op set differs from fp32 — it adds `ConvInteger`, `DynamicQuantizeLinear`, `MatMulInteger`)
- changing the number of classes only if that alters the graph shape (it does not, in practice)

It is **not** invalidated by retraining with different data or different weights. Weights do not change the graph.

**If you load a model needing an excluded operator, inference fails at session creation** with a "kernel not found" error — not a graceful fallback.

## Build

```bash
docker build -t ort-minimal:v1.20.1 tools/ort-minimal
```

**1–3 hours on first run.** It compiles emsdk, protobuf and ORT itself. Docker layer caching makes subsequent builds fast unless the config changes.

Extract the artefacts:

```bash
id=$(docker create ort-minimal:v1.20.1)
docker cp "$id:/out" ./ort-artifacts
docker rm "$id"
ls -la ort-artifacts/
```

## Using the output

Minimal builds **cannot load `.onnx`** — they need ORT format. Two changes are required together:

**1. Emit `.ort` from training:**

```bash
.venv/bin/python -m onnxruntime.tools.convert_onnx_models_to_ort \
    runs/mobile-classifier/ --enable_type_reduction --output_dir runs/ort
```

**2. Point the web app at the custom WASM.** The stock npm `onnxruntime-web` JS glue expects the stock binary; the custom build ships its own `.mjs` loader. Both must come from the same build — mixing versions fails at runtime with unhelpful errors.

## Regenerating the operator config

After any architecture change:

```bash
.venv/bin/python -m onnxruntime.tools.convert_onnx_models_to_ort \
    <dir-with-model.onnx> --enable_type_reduction --output_dir /tmp/ort_cfg
cp /tmp/ort_cfg/required_operators_and_types.config tools/ort-minimal/
docker build -t ort-minimal:v1.20.1 tools/ort-minimal   # full rebuild
```

## Current config

Generated from the INT8 graph of `mobilenetv3_small_100`, 4 classes:

```
ai.onnx;1;GlobalAveragePool
ai.onnx;6;HardSigmoid
ai.onnx;10;ConvInteger
ai.onnx;11;DynamicQuantizeLinear{"outputs": {"0": ["uint8_t"]}}
ai.onnx;13;Cast{...},Flatten,ReduceMean{...}
ai.onnx;14;Add{...},Mul{...},Relu{...}
com.microsoft;1;DynamicQuantizeMatMul
```

Seven operator groups, with per-operator type restrictions. The stock build carries several hundred kernels across all supported types.

## Is this worth it?

**It buys ~2.5 MB gzipped per first load.** Whether that justifies the pipeline depends on the deployment:

- **Worth it** if the model architecture is settled and users are on metered or slow rural connections — which is this project's situation once the model stops changing.
- **Not worth it** while the architecture is in flux. Every experiment needs a 1–3 hour rebuild, and a stale config produces a runtime that silently cannot load the new model.

The one-line `onnxruntime-web/wasm` import already captured the larger share of the available saving. This is the second-order optimisation.
