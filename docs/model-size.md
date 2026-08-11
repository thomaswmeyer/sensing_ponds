# Model and Runtime Size

How small the on-device classifier can get, and what each reduction costs.

All figures measured on this project's actual model (MobileNetV3-Small, 4 classes, 224×224) on 2026-08-11. Gzip is what matters — it is what crosses the wire.

## The headline

**The runtime dominates, not the model.** The model is ~1.3 MB gzipped; the ONNX Runtime WASM binary is 3.5 MB gzipped even after the easy fixes. Effort spent shrinking the model past INT8 is effort spent on the smaller half of the problem.

| | Raw | Gzipped |
|---|---|---|
| Model (INT8) | 1.58 MB | **1.27 MB** |
| ORT WASM runtime (stock) | 13.48 MB | **3.46 MB** |
| App JS + CSS | 292 KB | 94 KB |
| **Total first load** | ~15.4 MB | **~4.8 MB** |

## Already applied

### 1. WASM-only entry point — saved 12.1 MB raw / 2.9 MB gzipped

`import * as ort from 'onnxruntime-web'` pulls the WebGPU and WebGL backends and the JSEP WASM build. Importing `onnxruntime-web/wasm` instead drops both.

| | JS raw | JS gzip | WASM raw | WASM gzip |
|---|---|---|---|---|
| Default entry | 590 KB | 179 KB | 25.58 MB | 6.33 MB |
| `/wasm` entry | **286 KB** | **92 KB** | **13.48 MB** | **3.46 MB** |

WebGPU is not worth its weight here. This model is 1.5M parameters at 224×224 — **1.2 ms per inference on CPU**. GPU acceleration solves a problem this app does not have.

### 2. Threaded WASM excluded from precache

The threaded build needs cross-origin isolation (COOP/COEP headers) to use `SharedArrayBuffer`. Without those headers it downloads and then silently falls back to single-threaded anyway. `ort.env.wasm.numThreads = 1` and a `globIgnores` entry keep it out.

### 3. ❌ INT8 quantisation — abandoned, ships fp32

**Every INT8 strategy destroyed the model.** Measured on the trained model against the 920-image held-out test set:

| Variant | Accuracy | Size |
|---|---|---|
| **fp32 (shipping)** | **0.8826** | 5.81 MB |
| dynamic, per-tensor QUInt8 | 0.2620 | 1.61 MB |
| dynamic, per-channel QUInt8 | 0.2652 | 1.61 MB |
| dynamic, per-channel QInt8 | 0.2207 | 1.61 MB |
| dynamic, 13 worst Conv layers excluded | 0.2880 | 1.83 MB |
| static QDQ, MinMax calibration | 0.2837 | 1.75 MB |
| static QDQ, Entropy calibration | 0.2837 | 1.75 MB |
| static QDQ, **Percentile** calibration | 0.5467 | 1.75 MB |

Four classes, so chance is 0.25. Most variants are **at chance** — total failure, not degradation.

Reproduce with [`src/compare_quantization.py`](../src/compare_quantization.py).

#### Why: activation outliers, not weights

The first hypothesis was weight range. 13 of 53 Conv layers have per-channel weight ranges spanning more than 10 orders of magnitude — some squeeze-excite projection channels max out at `1e-16` beside others at `5e-1`:

```
     ratio  shape                     min|w|    max|w|
 5.3e+11    (64, 240, 1, 1)         1.18e-16  5.32e-01
 2.6e+11    (32, 120, 1, 1)         1.32e-13  2.65e-01
 2.3e+11    (24, 96, 1, 1)          1.35e-12  3.16e-01
```

**That hypothesis was wrong.** Excluding all 13 layers from quantisation left accuracy at 0.2880. Those near-zero channels are dead units the model learned to ignore — harmless in fp32 and not the cause. Per-channel quantisation, which exists precisely to handle weight-range spread, also did not help.

The actual mechanism shows up when you scale the input:

```
input ×0.1  →  int8 logits ≈ [-0.6, -3.7, -0.5, -3.0]     fp32 ≈ [0.6, -0.1, -2.4, -1.3]
input ×5.0  →  int8 logits ≈ [-10304, -23582, 13344, 3348]
```

Logits explode by four orders of magnitude while fp32 stays bounded. **Hard-swish activations produce large outliers**, and a per-tensor activation scale derived from those outliers crushes the real signal into a few quantisation levels.

This is consistent with the one partial success: Percentile calibration clips activation outliers at the 99.999th percentile instead of taking the true max, and recovers 0.2837 → 0.5467. Still 34 points short.

#### Why regularisation would not fix it

Stronger weight decay or dropout shrinks weights *toward* zero, compressing dynamic range rather than lifting the floor — it would create more `1e-16` channels, not fewer. And since excluding the extreme-range layers changed nothing, weight distribution is not the binding constraint. The problem is on the activation side, which weight regularisation does not control.

#### Decision: ship fp32

**5.81 MB raw / 5.33 MB gzipped.** Roughly 4 MB more than INT8 would have cost, on a first load that is cached forever afterwards. A 62-point accuracy loss is not a trade worth making.

Paths that could recover INT8, in order of cost:

1. **`efficientnet_lite0`** — designed for quantisation, with hard-swish and squeeze-excite deliberately removed. ~4.7M params, so somewhat larger fp32, but should quantise cleanly. Requires retraining (~1 hour) and re-verifying.
2. **Quantisation-aware training** — the model learns to tolerate quantisation noise. Most reliable, most work.
3. **fp16** — halves size to ~2.9 MB with far less accuracy risk than INT8. Untested here; the obvious next thing to try.

## Possible: custom minimal ORT build

**The model uses 11 distinct operators.** The stock runtime carries hundreds.

```
fp32: Conv, Identity, HardSwish, Relu, ReduceMean,
      HardSigmoid, Mul, Add, GlobalAveragePool, Flatten, Gemm

int8: adds DynamicQuantizeLinear, Cast, Reshape,
      ConvInteger, MatMulInteger
```

[ONNX Runtime supports building with only required kernels](https://onnxruntime.ai/docs/build/custom.html):

```bash
# 1. Config listing only the operators our model needs
python tools/python/create_reduced_build_config.py \
    --format ORT model.onnx > required_ops.config

# 2. Convert to ORT format (required by minimal builds)
python -m onnxruntime.tools.convert_onnx_models_to_ort \
    --enable_type_reduction model.onnx

# 3. Build
./build.sh --build_wasm --minimal_build extended \
    --include_ops_by_config required_ops.config \
    --enable_reduced_operator_type_support \
    --config MinSizeRel --disable_ml_ops --disable_rtti
```

**Expected: ~3 MB raw, roughly 1 MB gzipped** — a further ~2.5 MB off the wire.

### What it costs

- **Build infrastructure.** Requires Emscripten and a full ORT source build — hours on first run, and a Docker image to keep it reproducible.
- **Rebuild on every architecture change.** Swapping to `efficientnet_lite0` changes the operator set and invalidates the build. The INT8 op set differs from fp32, so the quantisation decision must be final first.
- **ORT format models.** Minimal builds cannot load `.onnx`; the training script's export step would need to emit `.ort` too.
- **A binary in the repo.** The custom WASM cannot be fetched from npm — it has to be committed or built in CI.

### Recommendation

**Not yet.** Do it when the model architecture is settled and 2.5 MB is worth a reproducible build pipeline. Until then it would need rebuilding on every experiment, and the model is not final — the four-class set may change, and INT8 accuracy is unverified.

The `/wasm` entry point captured most of the available saving for a one-line change.

## Other levers

| Lever | Saving | Cost |
|---|---|---|
| **Brotli instead of gzip** | ~15–20% over gzip | Server config only. Free win — do this at deploy time. |
| **Lazy-load the model** | 4.8 MB off *first paint* | Camera UI appears immediately, model loads behind it. Does not reduce total bytes. |
| Smaller input (160×160) | ~50% fewer FLOPs, model size unchanged | Retraining; accuracy loss. Not size-motivated — we are not compute-bound. |
| `mobilenetv3_small_075` | ~30% fewer params (~1.1 MB INT8) | Accuracy loss on an already-small model. |

**Brotli is the cheapest remaining win.** Render serves it automatically for static sites.

## What the user actually downloads

First visit, gzipped, with the current build:

```
app shell (JS + CSS + HTML)     94 KB
ORT WASM runtime             3,460 KB
model (INT8)                 1,270 KB
Noto Sans Tamil (subset)     ~100 KB
Tamil audio (~100 strings)  1,000-3,000 KB
                            ─────────
                            ~6-8 MB
```

Cached in the service worker after that, so the field cost is zero — but the first load matters on a rural connection.

**Recommendation: make the language pack a deliberate first-run download over Wi-Fi** rather than something fetched in the field. The audio is comparable in size to everything else combined.

## Reproducing these numbers

```bash
.venv/bin/python -m pip install onnx onnxruntime
.venv/bin/python - <<'PY'
import torch, timm, gzip
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.preprocess import quant_pre_process

m = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=4).eval()
torch.onnx.export(m, torch.randn(1,3,224,224), "/tmp/m.onnx", opset_version=17, dynamo=False)
quant_pre_process("/tmp/m.onnx", "/tmp/m_prep.onnx", skip_symbolic_shape=True)
quantize_dynamic("/tmp/m_prep.onnx", "/tmp/m_int8.onnx", weight_type=QuantType.QUInt8)

for tag, p in [("fp32","/tmp/m.onnx"), ("int8","/tmp/m_int8.onnx")]:
    raw = open(p,"rb").read()
    print(f"{tag:<6} {len(raw)/1048576:5.2f} MB  gzip {len(gzip.compress(raw))/1048576:5.2f} MB")
PY
```
