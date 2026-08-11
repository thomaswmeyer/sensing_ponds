"""Compare INT8 quantisation strategies against the fp32 baseline.

Written because default dynamic quantisation destroyed this model: 88.3% -> 28.3%,
barely above the 25% four-class chance level. MobileNetV3's depthwise convolutions
are the known culprit -- per-tensor weight scales cannot cover the very different
value ranges across depthwise channels, so most channels quantise to near-zero.

Per-channel quantisation gives each output channel its own scale and normally
recovers almost all of the loss. This script measures rather than assumes.

Usage:
    .venv/bin/python src/compare_quantization.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as rt
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.preprocess import quant_pre_process
from torch.utils.data import DataLoader

from train_mobile_classifier import PlantDataset, discover, eval_transform, group_split

RUN_DIR = Path("runs/mobile-classifier")
FP32 = RUN_DIR / "mobilenetv3_small_plants.onnx"

VARIANTS = {
    "per-tensor QUInt8 (default)": dict(weight_type=QuantType.QUInt8, per_channel=False),
    "per-channel QUInt8": dict(weight_type=QuantType.QUInt8, per_channel=True),
    "per-channel QInt8": dict(weight_type=QuantType.QInt8, per_channel=True),
}


def accuracy(path: Path | str, batches: list[tuple[np.ndarray, np.ndarray]]) -> float:
    sess = rt.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    ok = n = 0
    for xb, yb in batches:
        pred = sess.run(None, {name: xb})[0].argmax(1)
        ok += int((pred == yb).sum())
        n += len(yb)
    return ok / n


def main() -> None:
    samples, classes = discover([Path("data/mendeley"), Path("data/gbif")])
    _, _, test = group_split(samples, 0.15, 42)

    # num_workers=0: the transform pipeline does not survive fork in every
    # context, and this is a one-off measurement where speed does not matter.
    loader = DataLoader(PlantDataset(test, eval_transform()), batch_size=64, num_workers=0)
    batches = [(x.numpy(), y.numpy()) for x, y in loader]
    print(f"{len(batches)} batches, {sum(len(y) for _, y in batches)} images, classes={classes}\n")

    base = accuracy(FP32, batches)
    base_mb = FP32.stat().st_size / 1048576
    print(f"{'fp32 baseline':<30} {base:.4f}   {base_mb:5.2f} MB")

    prep = Path("/tmp/quant_prep.onnx")
    quant_pre_process(str(FP32), str(prep), skip_symbolic_shape=True)

    results = {}
    for i, (name, kwargs) in enumerate(VARIANTS.items()):
        out = Path(f"/tmp/quant_variant_{i}.onnx")
        quantize_dynamic(str(prep), str(out), **kwargs)
        acc = accuracy(out, batches)
        mb = out.stat().st_size / 1048576
        results[name] = (acc, mb, out)
        print(f"{name:<30} {acc:.4f}   {mb:5.2f} MB   delta {acc - base:+.4f}")

    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f"\nBest: {best[0]}  ({best[1][0]:.4f}, {best[1][1]:.2f} MB)")
    if best[1][0] < base - 0.02:
        print(
            "\nNo INT8 variant is within 2 points of fp32. Options:\n"
            "  - ship fp32 (5.8 MB raw, 5.3 MB gzipped)\n"
            "  - retrain with efficientnet_lite0, built without the hard-swish and\n"
            "    squeeze-excite blocks that quantise badly\n"
            "  - quantisation-aware training"
        )
    else:
        print(f"\nShip this variant: {best[1][2]}")


if __name__ == "__main__":
    main()
