/**
 * On-device plant classification with ONNX Runtime Web.
 *
 * PREPROCESSING MUST MATCH TRAINING EXACTLY. This is the single most common
 * source of silent accuracy loss in browser deployment: a mismatched resize
 * filter or a forgotten normalisation costs accuracy without raising any error.
 * The pipeline below mirrors eval_transform() in src/train_mobile_classifier.py:
 *
 *     Resize(255, 255) -> CenterCrop(224) -> Normalize(ImageNet mean/std)
 *
 * Note the resize is to a SQUARE 255x255, not a shortest-side resize -- that is
 * what albumentations A.Resize does, and it distorts aspect ratio. Matching the
 * distortion matters more than avoiding it.
 *
 * Parameters are read from model_config.json (written by the training script)
 * rather than hardcoded here, so retraining cannot silently desynchronise the
 * two. See docs/getting-data.md#training.
 */

// The `/wasm` entry point, not the default. The default bundle pulls in WebGPU
// and WebGL backends plus the JSEP WASM build: ~590 KB of JS and a 25.6 MB
// binary. This app needs neither -- a 2 MB model at 224x224 runs in single-digit
// milliseconds on the CPU backend. See docs/model-size.md.
import * as ort from 'onnxruntime-web/wasm'

const MODEL_URL = '/model/plants.onnx'
const CONFIG_URL = '/model/model_config.json'

let sessionPromise = null
let config = null

// Run the WASM runtime on a single thread. Note this does NOT select a smaller
// binary: onnxruntime-web ships only `-threaded` builds, so
// ort-wasm-simd-threaded.wasm (12.86 MB raw, ~3.3 MB brotli) is what loads
// either way. What this avoids is the SharedArrayBuffer thread pool, which
// needs cross-origin isolation (COOP/COEP) we do not set -- without those
// headers ORT falls back to one thread regardless, so this just makes the
// actual behaviour explicit. The model is small enough that single-threaded
// inference is a few milliseconds.
ort.env.wasm.numThreads = 1
ort.env.wasm.simd = true

export class ModelUnavailableError extends Error {}

async function loadConfig() {
  const res = await fetch(CONFIG_URL)
  if (!res.ok) throw new ModelUnavailableError(`model_config.json: ${res.status}`)
  const cfg = await res.json()

  // The abstain threshold is not optional. Without it the app would ship raw
  // softmax, which is systematically overconfident and would confidently label
  // a rock -- exactly the failure the abstain path exists to prevent.
  if (cfg.abstain_supported && typeof cfg.abstain_threshold !== 'number') {
    throw new ModelUnavailableError('model_config.json claims abstain support but has no threshold')
  }
  return cfg
}

export async function loadModel() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      config = await loadConfig()
      return ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      })
    })().catch((err) => {
      sessionPromise = null // allow retry
      throw err
    })
  }
  return sessionPromise
}

export function getConfig() {
  return config
}

/**
 * Source image -> normalised NCHW Float32Array.
 *
 * Exported for the parity test: web/src/lib/inference.test.js compares this
 * against tensors produced by the Python pipeline on the same image. Any change
 * here must keep that test passing.
 */
export function preprocess(source, cfg) {
  const { resize, center_crop: crop, mean, std } = cfg.preprocessing

  const canvas = document.createElement('canvas')
  canvas.width = resize
  canvas.height = resize
  const ctx = canvas.getContext('2d', { willReadFrequently: true })

  // Square resize, matching albumentations A.Resize -- deliberately not
  // preserving aspect ratio, because training did not.
  ctx.drawImage(source, 0, 0, resize, resize)

  const offset = Math.floor((resize - crop) / 2)
  const { data } = ctx.getImageData(offset, offset, crop, crop)

  // HWC uint8 RGBA -> CHW float32 RGB, normalised.
  const out = new Float32Array(3 * crop * crop)
  const plane = crop * crop
  for (let i = 0; i < plane; i++) {
    const p = i * 4
    out[i] = (data[p] / 255 - mean[0]) / std[0]
    out[plane + i] = (data[p + 1] / 255 - mean[1]) / std[1]
    out[2 * plane + i] = (data[p + 2] / 255 - mean[2]) / std[2]
  }
  return out
}

function softmax(logits, temperature) {
  const scaled = logits.map((v) => v / temperature)
  const max = Math.max(...scaled)
  const exp = scaled.map((v) => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map((v) => v / sum)
}

/**
 * Classify an image source (video frame, canvas, or image element).
 *
 * Returns { abstain, label, confidence, probabilities }.
 *
 * `abstain` is a first-class outcome, not an error. A four-way softmax always
 * sums to 1, so the model cannot express "none of these" -- the threshold is
 * what gives it that vocabulary. Callers must handle abstain as a normal result
 * and still record the observation: an uncertain capture is the most valuable
 * one for the human review queue.
 */
export async function classify(source) {
  const session = await loadModel()
  const cfg = config

  const input = preprocess(source, cfg)
  const size = cfg.preprocessing.center_crop
  const tensor = new ort.Tensor('float32', input, [1, 3, size, size])

  const output = await session.run({ [session.inputNames[0]]: tensor })
  const logits = Array.from(output[session.outputNames[0]].data)

  // Temperature scaling was fitted on held-out validation data during training.
  // Applying it here is what makes the confidence value mean something.
  const probs = softmax(logits, cfg.temperature ?? 1)

  let best = 0
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i

  const threshold = cfg.abstain_supported ? cfg.abstain_threshold : 0
  const abstain = probs[best] < threshold

  return {
    abstain,
    label: cfg.classes[best],
    confidence: probs[best],
    probabilities: Object.fromEntries(cfg.classes.map((c, i) => [c, probs[i]])),
  }
}
