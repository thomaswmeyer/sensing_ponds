# Model and Runtime Size

How small the on-device classifier can get, and what each reduction costs.

All figures measured on this project's actual model (MobileNetV3-Small, 4 classes, 224×224) on 2026-08-11. Gzip is what matters — it is what crosses the wire.

## The headline

**The runtime dominates again, now that the model is fp16.** Measured on the actual build, 2026-08-12:

| | Raw | Gzipped |
|---|---|---|
| Model (fp16) | 2.94 MB | **2.69 MB** |
| ORT WASM runtime (stock) | 12.86 MB | **3.28 MB** |
| App JS + CSS | 292 KB | 94 KB |
| **Total first load** | ~16.1 MB | **~6.1 MB** |

### History of this table, because it inverted twice

The ordering here has flipped with each precision decision, and stale versions of this table have twice pointed the work in the wrong direction:

1. **INT8 shipping (1.27 MB gz):** runtime dominated ~3:1. Correct conclusion at the time: shrink the runtime, not the model.
2. **INT8 abandoned, fp32 shipping (5.39 MB gz):** the model became the *larger* half — and fp32 barely compresses, since its weights are high-entropy (5.81 → 5.39 MB, only 7%). This document was not updated and kept advising against model work for a day, exactly when model work had become the highest-value thing available.
3. **fp16 shipping (2.69 MB gz):** runtime is the larger half again, but only 1.2:1 — much closer than in (1).

**If you change precision, update this table in the same commit.** It is the document people consult before deciding where to spend effort, and a stale version actively misdirects that decision.

## Already applied

### 1. WASM-only entry point — saved 12.1 MB raw / 2.9 MB gzipped

`import * as ort from 'onnxruntime-web'` pulls the WebGPU and WebGL backends and the JSEP WASM build. Importing `onnxruntime-web/wasm` instead drops both.

| | JS raw | JS gzip | WASM raw | WASM gzip |
|---|---|---|---|---|
| Default entry | 590 KB | 179 KB | 25.58 MB | 6.33 MB |
| `/wasm` entry | **286 KB** | **92 KB** | **13.48 MB** | **3.46 MB** |

WebGPU is not worth its weight here. This model is 1.5M parameters at 224×224 — **1.2 ms per inference on CPU**. GPU acceleration solves a problem this app does not have.

### 2. Single-threaded execution — no size saving, and the precache bug it caused

`ort.env.wasm.numThreads = 1` avoids the `SharedArrayBuffer` thread pool, which needs cross-origin isolation (COOP/COEP) we do not set. Without those headers ORT falls back to one thread anyway, so this only makes the real behaviour explicit.

**It saves no bytes.** There is no separate single-threaded binary to select. Since ~1.19, `onnxruntime-web` ships only `-threaded` builds:

```
ort-wasm-simd-threaded.wasm           12.86 MB   ← the one we load
ort-wasm-simd-threaded.jsep.wasm      26 MB      (WebGPU)
ort-wasm-simd-threaded.asyncify.wasm  23 MB
ort-wasm-simd-threaded.jspi.wasm      14 MB
```

An earlier `globIgnores: ['**/ort-wasm-simd-threaded*.wasm', '**/ort-*.mjs']` was written believing it excluded a fat threaded build in favour of a lean single-threaded one. **No such build exists** — that pattern matched the only runtime the app has, plus its loader glue, and dropped both from the service-worker precache while `plants.onnx` stayed in it. The app kept working online, where the runtime is fetched on demand, so the breakage was invisible outside a genuinely offline install — the one scenario this app is built for.

A second, independent cause pointed the same way: `maximumFileSizeToCacheInBytes` was 12 MB and the runtime is 12.86 MB. Workbox compares against the **uncompressed** size and silently drops anything larger with only a build-log warning, so the cap alone would have excluded the runtime even with the globs fixed. Both are fixed in [`web/vite.config.js`](../web/vite.config.js); the precache went from 12 entries to 13.

**Verify offline behaviour on a real airplane-mode install after touching any of this.** Both failure modes are silent, and neither shows up in an online smoke test.

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

#### Decision: fp32 over INT8

A 62-point accuracy loss is not a trade worth making for 4 MB. fp32 shipped until fp16 replaced it below.

### 4. ✅ fp16 — shipping, and it costs nothing

**Identical accuracy to fp32, half the size.** Measured on the same 920-image held-out test set:

| | Accuracy | Balanced | ECE | Raw | Gzipped |
|---|---|---|---|---|---|
| fp32 | 0.8826 | 0.8782 | 0.0184 | 5.81 MB | 5.39 MB |
| **fp16 (shipping)** | **0.8826** | **0.8782** | **0.0181** | **2.94 MB** | **2.69 MB** |

Per-class F1 is identical to four decimal places for all four classes. Prediction agreement is **1.0000** — not one of the 920 images changes its predicted label. **2.70 MB off the wire.**

Reproduce with [`src/compare_fp16.py`](../src/compare_fp16.py).

#### Why fp16 survives where INT8 did not

They fail differently. INT8 gives you 256 uniformly-spaced levels across the whole tensor range, so one hard-swish outlier at ±23000 stretches the scale until real signal collapses into a handful of levels. **fp16 keeps 5 exponent bits**, so it represents both 23000 and 0.001 to roughly 3 decimal digits each — the outliers stay representable and stop crushing everything else. The INT8 diagnosis was about *dynamic range*, and fp16 is precisely the format that preserves it.

26 subnormal weights flush to fp16's floor during conversion. These are the same dead squeeze-excite channels the INT8 investigation identified (fp32 magnitudes down to `1e-16`) — the model already ignores them, which is why zero predictions change.

#### What to watch: abstain, not accuracy

Abstain agreement is **0.9978** — 2 of 920 images cross the 0.72 threshold. Probabilities shift by up to 0.0277, and the client thresholds on a probability rather than an argmax, so a variant can preserve every prediction and still change what the user sees. This is noise rather than degradation (coverage rose slightly, 0.8522 → 0.8543; precision held at 0.9504, above the 0.95 target), but **it is the metric to check on any future precision change** — an accuracy-only comparison would have shown nothing at all.

`compare_fp16_accuracy()` in the training script checks both, and export falls back to fp32 if either fails.

#### Implementation

`keep_io_types=True` keeps the graph's input and output fp32, casting just inside the boundary, so the client still sends a `Float32Array` and needs no change. `GlobalAveragePool` and `ReduceMean` stay in fp32 — both accumulate over many elements where half-precision rounding compounds, and both hold a negligible share of the weights.

Requires `onnxconverter-common`.

#### If you need to go smaller still

1. **`efficientnet_lite0`** — designed for quantisation, with hard-swish and squeeze-excite deliberately removed. ~4.7M params, so larger in fp32, but should quantise cleanly to INT8 (~1.2 MB). Requires retraining and full revalidation.
2. **Quantisation-aware training** — the model learns to tolerate quantisation noise. Most reliable, most work.
3. **`mobilenetv3_small_075` / smaller input** — cheap to try, costs accuracy on an already-small model.

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
| **Brotli instead of gzip** | ~15–20% over gzip | Already automatic on Cloudflare Pages. |
| **Lazy-load the model** | 4.8 MB off *first paint* | Camera UI appears immediately, model loads behind it. Does not reduce total bytes. |
| Smaller input (160×160) | ~50% fewer FLOPs, model size unchanged | Retraining; accuracy loss. Not size-motivated — we are not compute-bound. |
| `mobilenetv3_small_075` | ~30% fewer params (~1.1 MB INT8) | Accuracy loss on an already-small model. |

**Brotli is already in hand.** The deploy target is Cloudflare Pages, which brotli-compresses static assets in transit automatically — see the note in [`web/public/_headers`](../web/public/_headers). The gzip figures in this document are therefore an upper bound on what actually crosses the wire; real transfer is roughly 15–20% below them.

## What the user actually downloads

First visit, gzipped, with the current build:

```
app shell (JS + CSS + HTML)     94 KB
ORT WASM runtime             3,280 KB
model (fp16)                 2,690 KB
Noto Sans Tamil (subset)     ~100 KB
Tamil audio (~100 strings)  1,000-3,000 KB
                            ─────────
                            ~7-9 MB   (gzip; ~15-20% less over brotli)
```

All of it is precached by the service worker on first load, so the field cost afterwards is zero — but the first load matters on a rural connection. That precache is only trustworthy because both bugs in §2 are fixed; before that, the runtime was excluded and the app could not classify offline at all.

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
