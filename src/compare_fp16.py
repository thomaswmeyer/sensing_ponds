"""Compare fp16 against the fp32 baseline on the held-out test set.

Written after INT8 was abandoned (docs/model-size.md): every INT8 strategy
collapsed to near the 25% chance level because hard-swish activation outliers
dominate the per-tensor activation scale. fp16 is a different failure mode
entirely -- it keeps 5 exponent bits, so those same outliers stay representable
where INT8's 256 uniform levels could not hold them. Halving the weight bytes
should therefore cost approximately nothing.

"Should cost approximately nothing" is exactly the assumption INT8 taught us to
measure, so this script measures it: accuracy, per-class F1, and -- because the
client thresholds on a calibrated probability rather than an argmax -- the
abstain behaviour and calibration error too. A variant that keeps argmax intact
but shifts probabilities across the 0.72 threshold would change what the user
sees while showing no accuracy delta at all.

Usage:
    .venv/bin/python src/compare_fp16.py
"""

from __future__ import annotations

import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as rt
from torch.utils.data import DataLoader

from train_mobile_classifier import PlantDataset, discover, eval_transform, group_split

RUN_DIR = Path("runs/mobile-classifier")
FP32 = RUN_DIR / "mobilenetv3_small_plants.onnx"
FP16 = RUN_DIR / "mobilenetv3_small_plants_fp16.onnx"
CONFIG = RUN_DIR / "model_config.json"

# Ops kept in fp32 inside an otherwise-fp16 graph. Both accumulate over many
# elements, where half-precision rounding compounds; they are also a negligible
# share of the weights, so keeping them fp32 costs bytes we do not care about
# and buys numerical stability we do.
KEEP_FP32 = ["GlobalAveragePool", "ReduceMean"]


def convert_fp16(src: Path, dst: Path) -> None:
    """fp32 -> fp16 weights, with fp32 kept at the graph boundary.

    keep_io_types=True is what makes this a drop-in swap: the graph still takes a
    float32 NCHW tensor and still emits float32 logits, with casts inserted just
    inside the boundary. web/src/lib/inference.js therefore needs no change at
    all -- it keeps building the same Float32Array it always did.
    """
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    # The converter warns once per subnormal weight it flushes to fp16's floor.
    # Those are the dead squeeze-excite channels the INT8 investigation already
    # identified as contributing nothing (docs/model-size.md) -- dozens of
    # warnings about weights the model learned to ignore, so they are summarised
    # below rather than printed individually.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fp16_model = float16.convert_float_to_float16(
            model, keep_io_types=True, op_block_list=KEEP_FP32
        )
    onnx.checker.check_model(fp16_model)
    onnx.save(fp16_model, str(dst))
    n_trunc = sum("truncated" in str(w.message) for w in caught)
    print(f"  {n_trunc} subnormal weights flushed to the fp16 floor (dead SE channels)")


def sizes(p: Path) -> tuple[float, float]:
    raw = p.read_bytes()
    return len(raw) / 1048576, len(gzip.compress(raw)) / 1048576


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run(path: Path, batches: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    sess = rt.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: xb})[0] for xb, _ in batches])


def ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error -- mean gap between confidence and accuracy."""
    conf = probs.max(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def per_class_f1(pred: np.ndarray, y: np.ndarray, n_classes: int) -> list[float]:
    out = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        out.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return out


def report(
    tag: str, logits: np.ndarray, y: np.ndarray, classes: list[str], temp: float, thr: float
) -> dict:
    probs = softmax(logits, temp)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    confident = probs.max(axis=1) >= thr

    stats = {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(
            np.mean([correct[y == c].mean() for c in range(len(classes)) if (y == c).any()])
        ),
        "ece": ece(probs, correct),
        # Coverage: the share of images answered rather than returned "not sure".
        "abstain_coverage": float(confident.mean()),
        # Precision among answered images -- the number the 0.72 threshold was
        # chosen to hold at 0.95. A size optimisation must not erode it.
        "precision_when_confident": float(correct[confident].mean()) if confident.any() else 0.0,
        "f1": per_class_f1(pred, y, len(classes)),
    }
    print(
        f"{tag:<6} acc {stats['accuracy']:.4f}  bal {stats['balanced_accuracy']:.4f}"
        f"  ECE {stats['ece']:.4f}  coverage {stats['abstain_coverage']:.4f}"
        f"  precision {stats['precision_when_confident']:.4f}"
    )
    return stats


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    temp = cfg["temperature"]
    thr = cfg["abstain_threshold"]

    samples, classes = discover([Path("data/mendeley"), Path("data/gbif")])
    _, _, test = group_split(samples, 0.15, 42)
    loader = DataLoader(PlantDataset(test, eval_transform()), batch_size=64, num_workers=0)
    batches = [(x.numpy(), y.numpy()) for x, y in loader]
    y = np.concatenate([yb for _, yb in batches])
    print(f"{len(y)} held-out images, classes={classes}")
    print(f"temperature={temp:.4f}  abstain_threshold={thr:.4f}\n")

    print("Converting fp32 -> fp16:")
    convert_fp16(FP32, FP16)
    raw32, gz32 = sizes(FP32)
    raw16, gz16 = sizes(FP16)
    print(f"  fp32 {raw32:5.2f} MB raw / {gz32:5.2f} MB gz")
    print(f"  fp16 {raw16:5.2f} MB raw / {gz16:5.2f} MB gz")
    print(f"  saving {gz32 - gz16:.2f} MB gzipped ({(1 - gz16 / gz32) * 100:.0f}% smaller)\n")

    lo32, lo16 = run(FP32, batches), run(FP16, batches)
    s32 = report("fp32", lo32, y, classes, temp, thr)
    s16 = report("fp16", lo16, y, classes, temp, thr)

    p32, p16 = softmax(lo32, temp), softmax(lo16, temp)
    agreement = float((p32.argmax(1) == p16.argmax(1)).mean())
    # Whether the two models make the SAME abstain decision, image by image. A
    # model can match on accuracy and still flip which images it declines to
    # answer, and that flip is what a field user would actually notice.
    abstain_agreement = float(((p32.max(1) >= thr) == (p16.max(1) >= thr)).mean())

    print(f"\nprediction agreement {agreement:.4f}   abstain agreement {abstain_agreement:.4f}")
    print(f"max |logit delta|    {np.abs(lo32 - lo16).max():.4f}")
    print(f"max |prob delta|     {np.abs(p32 - p16).max():.4f}")

    print("\nper-class F1:")
    for i, c in enumerate(classes):
        print(f"  {c:<16} fp32 {s32['f1'][i]:.4f}   fp16 {s16['f1'][i]:.4f}")

    delta = s16["accuracy"] - s32["accuracy"]
    print(f"\naccuracy delta {delta:+.4f}")
    print(
        "\nVERDICT: fp16 is shippable."
        if delta >= -0.005 and abstain_agreement >= 0.99
        else "\nVERDICT: fp16 changed behaviour materially -- investigate before shipping."
    )


if __name__ == "__main__":
    main()
