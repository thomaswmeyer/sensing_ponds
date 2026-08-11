/**
 * Preprocessing parity guard.
 *
 * The browser pipeline must produce the same tensor as eval_transform() in
 * src/train_mobile_classifier.py. A mismatched resize or a forgotten
 * normalisation loses accuracy silently -- no error, just worse predictions in
 * the field. These tests check the arithmetic that is easy to get wrong.
 *
 * They do NOT prove parity with PyTorch. That needs a fixture: run the Python
 * pipeline on a known image, dump the tensor, and compare here to ~1e-3. Write
 * that once a trained model exists -- see the pending test at the bottom.
 */

import { describe, expect, it, vi } from 'vitest'

vi.mock('onnxruntime-web', () => ({
  env: { wasm: {} },
  InferenceSession: { create: vi.fn() },
  Tensor: class {},
}))

const { preprocess } = await import('./inference.js')

const CFG = {
  preprocessing: {
    resize: 255,
    center_crop: 224,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  },
}

/** A canvas-like source filled with one solid colour. */
function solidSource(r, g, b) {
  const calls = []
  globalThis.document = {
    createElement: () => ({
      width: 0,
      height: 0,
      getContext: () => ({
        drawImage: (...args) => calls.push(args),
        getImageData: (x, y, w, h) => ({
          data: new Uint8ClampedArray(w * h * 4).map((_, i) => {
            const channel = i % 4
            return channel === 0 ? r : channel === 1 ? g : channel === 2 ? b : 255
          }),
        }),
      }),
    }),
  }
  return { calls }
}

describe('preprocess', () => {
  it('produces a CHW tensor of the cropped size', () => {
    solidSource(0, 0, 0)
    const out = preprocess({}, CFG)
    expect(out).toBeInstanceOf(Float32Array)
    expect(out.length).toBe(3 * 224 * 224)
  })

  it('normalises with ImageNet statistics', () => {
    // Mid-grey: (0.5 - mean) / std, per channel.
    solidSource(128, 128, 128)
    const out = preprocess({}, CFG)
    const plane = 224 * 224
    const v = 128 / 255

    expect(out[0]).toBeCloseTo((v - 0.485) / 0.229, 5)
    expect(out[plane]).toBeCloseTo((v - 0.456) / 0.224, 5)
    expect(out[2 * plane]).toBeCloseTo((v - 0.406) / 0.225, 5)
  })

  it('separates channels rather than interleaving them', () => {
    // Pure red must land entirely in the first plane. Interleaved RGB is the
    // classic mistake here and produces plausible-looking garbage.
    solidSource(255, 0, 0)
    const out = preprocess({}, CFG)
    const plane = 224 * 224

    expect(out[0]).toBeCloseTo((1 - 0.485) / 0.229, 5)
    expect(out[plane]).toBeCloseTo((0 - 0.456) / 0.224, 5)
    expect(out[2 * plane]).toBeCloseTo((0 - 0.406) / 0.225, 5)
  })

  it('resizes to a square before cropping, matching albumentations', () => {
    // A.Resize(255, 255) distorts aspect ratio. Training did that, so inference
    // must too -- matching the distortion matters more than avoiding it.
    const { calls } = solidSource(0, 0, 0)
    preprocess({}, CFG)
    expect(calls[0].slice(1)).toEqual([0, 0, 255, 255])
  })
})

describe.todo(
  'matches the PyTorch pipeline on a fixture image to 1e-3 ' +
    '(needs a tensor dumped from eval_transform once a model is trained)',
)
