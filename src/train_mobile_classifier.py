"""Fine-tune MobileNetV3-Small for floating aquatic plant identification.

Trains a four-class classifier (water hyacinth / water lettuce / duckweed /
Monochoria) on the WaterHyacinth dataset for on-device field identification.

See docs/classifier-options.md#mobile-field-identification-track for rationale.

Key design decisions, all deliberate:

  * MobileNetV3-Small, not Ultralytics yolo*-cls. Comparable size, but AGPL-3.0
    is a live risk for a distributed app. timm is Apache-2.0.
  * Group-aware splitting. The dataset is two districts over three months, so a
    random split leaks near-duplicate same-pond shots across train/test and
    reports inflated accuracy. See group_split().
  * Augmentation targets the domain gap (water colour, glare, phone cameras),
    not raw volume.
  * Uses only the 1,790 originals. The published 4,050 pre-augmented images are
    static rotations/flips -- worse than on-the-fly, and they leak across splits.

Usage:
    python src/train_mobile_classifier.py --data-root data/WaterHyacinth
    python src/train_mobile_classifier.py --data-root data/... --export-tflite
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "albumentations is required: pip install albumentations opencv-python-headless"
    ) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class Sample:
    path: Path
    label: int
    group: str
    source: str  # "mendeley" | "gbif"
    regional: bool = False  # GBIF only: recorded in the deployment region


def infer_group(path: Path) -> str:
    """Capture-session identifier, used to keep related photos out of both splits.

    Mendeley filenames carry full timestamps (``20230811_211154.jpg``,
    ``IMG_20230923_105301.jpg``), so date+hour is a real session key rather than a
    guess. Verified against the actual dataset: 1,765 of 1,790 images match, and
    they collapse to just 10 distinct capture dates.

    That scarcity is the whole reason this function exists. A random split puts
    photos of the same pond minutes apart into train and test, and the model
    scores ~98% by recognising the afternoon. Grouping by date+hour is the finest
    honest granularity the filenames support.

    GBIF images are keyed individually -- each is a separate observer, place and
    day, so there is no session to preserve.
    """
    stem = path.stem

    if stem.startswith("gbif_"):
        return f"gbif:{stem}"

    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[_-](\d{2})", stem)
    if m:
        return f"{path.parent.name}:{m.group(1)}{m.group(2)}{m.group(3)}_h{m.group(4)}"

    date = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", stem)
    if date:
        return f"{path.parent.name}:{date.group(1)}{date.group(2)}{date.group(3)}"

    return f"{path.parent.name}:ungrouped"


# Mendeley directory names -> canonical class folders. The Mendeley tree uses
# long descriptive names ("Common Water Hyacinth (Eichornia crassipes)" -- note
# the misspelling in the original); GBIF uses short slugs. Both must map to one
# label space.
MENDELEY_CLASS_MAP = {
    "common water hyacinth": "water_hyacinth",
    "water lettuce": "water_lettuce",
    "common duckweeds": "duckweed",
    # Deliberately absent: "heartleaf false pickerelweed" (Monochoria korsakowii).
    # Dropped as a class -- temperate East Asian, 0 GBIF records in IN/LK, and
    # only 2 capture days here. See docs/datasets.md.
}


def canonical_class(dirname: str) -> str | None:
    """Map a source directory name to a canonical class, or None to skip it."""
    base = dirname.split("(")[0].strip().lower()
    if base in MENDELEY_CLASS_MAP:
        return MENDELEY_CLASS_MAP[base]
    slug = dirname.strip().lower().replace(" ", "_")
    if slug in set(MENDELEY_CLASS_MAP.values()) | {"salvinia"}:
        return slug
    return None


def discover(roots: list[Path]) -> tuple[list[Sample], list[str]]:
    """Walk one or more folder-per-class trees into a single label space.

    Accepts both the Mendeley layout and the GBIF fetcher's output, mapping each
    to canonical class names. Directories that map to nothing (e.g. Monochoria)
    are skipped with a notice rather than silently dropped -- a quietly missing
    class is the kind of thing nobody notices until the model ships.
    """
    found: dict[str, list[tuple[Path, str, bool]]] = defaultdict(list)
    skipped: set[str] = set()

    gbif_regional: set[str] = set()
    for root in roots:
        manifest = root / "manifest.csv"
        if manifest.exists():
            import csv as _csv

            with manifest.open() as f:
                for row in _csv.DictReader(f):
                    if row.get("regional", "").lower() in ("true", "1"):
                        gbif_regional.add(f"gbif_{row['gbif_key']}")

    for root in roots:
        if not root.exists():
            raise SystemExit(f"Data root does not exist: {root}")
        source = "gbif" if (root / "manifest.csv").exists() else "mendeley"

        for cdir in sorted(d for d in root.rglob("*") if d.is_dir()):
            cls = canonical_class(cdir.name)
            images = [p for p in cdir.glob("*") if p.suffix.lower() in IMAGE_EXTS]
            if not images:
                continue
            if cls is None:
                skipped.add(cdir.name)
                continue
            for p in sorted(images):
                # Drop pre-augmented copies if present alongside originals.
                if re.search(r"(_aug|_augmented|_rot\d|_flip)", p.stem, re.I):
                    continue
                found[cls].append((p, source, p.stem in gbif_regional))

    if skipped:
        print(f"Skipped (not a known class): {', '.join(sorted(skipped))}")
    if not found:
        raise SystemExit(f"No images found under {[str(r) for r in roots]}")

    classes = sorted(found)
    samples = [
        Sample(path=p, label=i, group=infer_group(p), source=src, regional=reg)
        for i, cls in enumerate(classes)
        for p, src, reg in found[cls]
    ]
    return samples, classes


def group_split(
    samples: list[Sample],
    val_frac: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Split into train / val / out-of-domain test.

    The test set is **GBIF regional records only** (India + Sri Lanka), held out
    entirely. This is deliberate and is the only measurement here worth trusting.

    Why not a conventional random or group split: the Mendeley images come from
    just 10 capture days in two Bangladeshi districts. Any split of that data
    measures how well the model recognises those particular ponds, not whether it
    identifies plants. Even a strict date-level split leaves train and test one
    afternoon apart at the same site.

    Holding out regional GBIF records instead measures the thing that actually
    matters: does a model trained mostly on Bangladeshi and global photos work on
    South Indian and Sri Lankan water bodies -- the deployment region. It is a
    harder and much more honest test, so expect a materially lower number than a
    random split would report. That gap is the finding, not a problem to tune away.

    Validation is a group-held-out slice of the training pool, used only for
    checkpoint selection.
    """
    test = [s for s in samples if s.regional]
    pool = [s for s in samples if not s.regional]

    if len(test) < 40:
        raise SystemExit(
            f"Only {len(test)} regional images -- too few for an out-of-domain test set.\n"
            "Run: python src/fetch_gbif.py --out data/gbif  (and check the regional counts)"
        )

    by_group: dict[str, list[Sample]] = defaultdict(list)
    for s in pool:
        by_group[s.group].append(s)

    groups = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)

    want_val = int(len(pool) * val_frac)
    val: list[Sample] = []
    train: list[Sample] = []
    for g in groups:
        (val if len(val) < want_val else train).extend(by_group[g])

    for name, split in (("train", train), ("val", val)):
        if not split:
            raise SystemExit(f"{name} split is empty -- too few distinct groups ({len(groups)}).")

    missing = set(range(max(s.label for s in samples) + 1)) - {s.label for s in test}
    if missing:
        print(
            f"NOTE: {len(missing)} class(es) have no regional test images; "
            "their test-set metrics will be absent or unreliable."
        )
    return train, val, test


class PlantDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: A.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = np.array(Image.open(s.path).convert("RGB"))
        return self.transform(image=image)["image"], s.label


def train_transform() -> A.Compose:
    """Augmentations chosen for the field domain gap, not for volume.

    Rationale per docs/classifier-options.md: aggressive crop (users will not
    frame like the dataset), full rotation and both flips (no canonical "up"
    looking down at water), heavy colour jitter (water hue varies with turbidity
    and sky -- force reliance on leaf morphology), glare and blur and JPEG
    artefacts (specular reflection off water is the characteristic failure).

    Deliberately absent: Cutout/CoarseDropout, which can erase the swollen
    petiole that distinguishes Eichhornia; and grayscale, since colour is signal.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), ratio=(0.8, 1.25)
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, border_mode=0, p=0.7),
            A.Affine(scale=(0.9, 1.1), shear=(-8, 8), p=0.3),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            A.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            # Specular glare off water -- the characteristic field failure.
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5), src_radius=110, angle_range=(0.0, 1.0), p=0.15
            ),
            A.RandomShadow(p=0.15),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.Defocus(radius=(1, 3)),
                ],
                p=0.3,
            ),
            # Cheap phone sensors: noise, compression, and effective downscale.
            A.GaussNoise(p=0.2),
            A.ImageCompression(quality_range=(40, 90), p=0.3),
            A.Downscale(scale_range=(0.5, 0.9), p=0.15),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(int(IMG_SIZE * 1.14), int(IMG_SIZE * 1.14)),
            A.CenterCrop(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def mixup(x: torch.Tensor, y: torch.Tensor, n_classes: int, alpha: float):
    """MixUp with soft targets.

    At ~1.8k images this regularises well, and the soft labels damp the
    overconfidence that would otherwise break the abstain threshold.
    """
    y_soft = F.one_hot(y, n_classes).float()
    if alpha <= 0:
        return x, y_soft
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_soft + (1 - lam) * y_soft[perm]


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, temperature: float = 1.0):
    model.eval()
    logits_all, labels_all = [], []
    for x, y in loader:
        logits_all.append(model(x.to(device, non_blocking=True)).cpu())
        labels_all.append(y)

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    probs = (logits / temperature).softmax(dim=-1)
    preds = probs.argmax(dim=-1)

    acc = (preds == labels).float().mean().item()
    # Balanced accuracy: the class counts are uneven, so plain accuracy flatters.
    per_class = [
        (preds[labels == c] == c).float().mean().item()
        for c in labels.unique(sorted=True)
    ]
    return acc, float(np.mean(per_class)), probs, labels, logits


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    """ECE -- how far confidence sits from accuracy. Drives the abstain threshold."""
    conf, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean().item() * abs(
                correct[mask].mean().item() - conf[mask].mean().item()
            )
    return ece


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling: divide logits by T to calibrate confidence.

    A network trained with cross-entropy is systematically overconfident, which
    makes raw softmax a poor basis for an abstain decision. One scalar fitted on
    held-out data fixes most of it without touching accuracy (dividing by a
    positive constant cannot change the argmax).
    """
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def abstain_threshold(probs: torch.Tensor, labels: torch.Tensor, target_precision: float):
    """Lowest confidence threshold whose retained predictions hit target precision.

    Abstain is a real output of this system, not a UI nicety. The model knows
    four plants; a user will point it at a fifth species, a rock, or their own
    shoe, and a four-way softmax always sums to 1 -- it has no way to say "none
    of these" unless we give it one.

    Returns (threshold, coverage). Coverage is the fraction of images the model
    would answer at all; 1 - coverage is how often the user sees "not sure".
    """
    conf, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    for t in np.arange(0.30, 1.00, 0.01):
        keep = conf >= t
        if keep.sum() < max(10, 0.05 * len(labels)):
            break
        if correct[keep].mean().item() >= target_precision:
            return float(t), float(keep.float().mean().item())
    return None, None


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        args.device
        if args.device
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    samples, classes = discover([Path(r) for r in args.data_root])
    train_s, val_s, test_s = group_split(samples, args.val_frac, args.seed)

    src = Counter(s.source for s in samples)
    print(f"Device:  {device}")
    print(f"Classes: {classes}")
    print(f"Sources: {dict(src)}  ({len({s.group for s in samples})} capture groups)")
    print(f"Split:   train={len(train_s)}  val={len(val_s)}  test={len(test_s)} (regional, held out)")
    print(f"         train: {dict(Counter(classes[s.label] for s in train_s))}")
    print(f"         test:  {dict(Counter(classes[s.label] for s in test_s))}")

    loader_kw = dict(
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    train_dl = DataLoader(
        PlantDataset(train_s, train_transform()), shuffle=True, drop_last=True, **loader_kw
    )
    val_dl = DataLoader(PlantDataset(val_s, eval_transform()), shuffle=False, **loader_kw)
    test_dl = DataLoader(PlantDataset(test_s, eval_transform()), shuffle=False, **loader_kw)

    model = timm.create_model(
        args.model, pretrained=True, num_classes=len(classes), drop_rate=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.25
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "mobilenetv3_small_plants.pt"

    best_bal_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y_soft = mixup(x, y, len(classes), args.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            loss = soft_cross_entropy(model(x), y_soft)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * x.size(0)

        acc, bal_acc, _, _, _ = evaluate(model, val_dl, device)
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {running / len(train_dl.dataset):.4f}"
            f"  val_acc {acc:.4f}  val_bal_acc {bal_acc:.4f}"
        )

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            torch.save(
                {"model": model.state_dict(), "classes": classes, "arch": args.model},
                ckpt_path,
            )

    # ---- Calibrate on validation, evaluate out-of-domain ----------------- #
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])

    # Temperature is fitted on val, never on test -- fitting it on the set you
    # report would leak and make the calibration figure meaningless.
    _, _, _, val_labels, val_logits = evaluate(model, val_dl, device)
    temperature = fit_temperature(val_logits, val_labels)

    acc, bal_acc, probs, labels, _ = evaluate(model, test_dl, device, temperature)
    ece = expected_calibration_error(probs, labels)
    thr, coverage = abstain_threshold(probs, labels, args.target_precision)

    print("\n" + "=" * 70)
    print("OUT-OF-DOMAIN TEST -- GBIF regional records (India / Sri Lanka)")
    print(f"accuracy {acc:.4f}   balanced {bal_acc:.4f}   ECE {ece:.4f}   T={temperature:.3f}")
    print("=" * 70)
    present = sorted({int(c) for c in labels.unique()})
    print(
        classification_report(
            labels.numpy(), probs.argmax(-1).numpy(),
            labels=present, target_names=[classes[i] for i in present],
            digits=3, zero_division=0,
        )
    )
    print("Confusion matrix (rows = true):")
    print(confusion_matrix(labels.numpy(), probs.argmax(-1).numpy(), labels=present))

    print("\n--- Abstain behaviour ---")
    if thr is not None:
        print(
            f"Threshold {thr:.2f} reaches {args.target_precision:.0%} precision on "
            f"{coverage:.1%} of images.\n"
            f"The remaining {1 - coverage:.1%} must return 'not sure' rather than a guess."
        )
    else:
        print(
            f"NO threshold reached {args.target_precision:.0%} precision at usable coverage.\n"
            "Do not ship an automatic-ID path on this model: it cannot tell you when\n"
            "it is wrong. Either gather more regional data or lower the target."
        )

    print(
        "\nNOTE: this is an out-of-domain score -- trained mostly on Bangladeshi and\n"
        "global photos, tested on regional ones. It is deliberately harder than a\n"
        "random split and is the number that predicts field behaviour. Deployment\n"
        "adds further shift (season, camera, holding angle) -- expect lower still."
    )

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "test_set": "gbif_regional_held_out",
                "test_accuracy": acc,
                "test_balanced_accuracy": bal_acc,
                "expected_calibration_error": ece,
                "temperature": temperature,
                "abstain_threshold": thr,
                "abstain_coverage": coverage,
                "target_precision": args.target_precision,
                "n_train": len(train_s),
                "n_val": len(val_s),
                "n_test": len(test_s),
                "train_sources": dict(Counter(s.source for s in train_s)),
            },
            indent=2,
        )
    )
    print(f"\nCheckpoint: {ckpt_path}\nMetrics:    {out_dir / 'metrics.json'}")

    if args.export_tflite:
        export_mobile(model, classes, out_dir, device, temperature, thr)


def export_mobile(
    model: nn.Module,
    classes: list[str],
    out_dir: Path,
    device: torch.device,
    temperature: float,
    threshold: float | None,
) -> None:
    """Export ONNX for downstream TFLite/ExecuTorch conversion.

    Direct PyTorch->TFLite has no first-party path; ONNX then onnx2tf (or
    ai-edge-torch) is the practical route.

    Watch INT8 accuracy here: MobileNetV3's hard-swish and squeeze-excite blocks
    quantise poorly. If post-training quantisation degrades noticeably, switch to
    efficientnet_lite0 (built without those ops) or use quantisation-aware
    training. Verify -- do not assume it survived.
    """
    model.eval().to("cpu")
    onnx_path = out_dir / "mobilenetv3_small_plants.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 3, IMG_SIZE, IMG_SIZE),
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    (out_dir / "labels.txt").write_text("\n".join(classes) + "\n")

    # The client cannot abstain without these. Shipping the model without the
    # temperature and threshold means shipping raw overconfident softmax, which
    # is exactly the failure the abstain path exists to prevent.
    (out_dir / "model_config.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "input_size": IMG_SIZE,
                "preprocessing": {
                    "resize": int(IMG_SIZE * 1.14),
                    "center_crop": IMG_SIZE,
                    "mean": list(IMAGENET_MEAN),
                    "std": list(IMAGENET_STD),
                },
                "temperature": temperature,
                "abstain_threshold": threshold,
                "abstain_supported": threshold is not None,
            },
            indent=2,
        )
    )
    print(f"ONNX:       {onnx_path}")
    print(f"Labels:     {out_dir / 'labels.txt'}")
    print(f"Config:     {out_dir / 'model_config.json'}  (preprocessing + abstain params)")
    print("Convert:    onnx2tf -i %s -oiqt   # INT8 TFLite" % onnx_path.name)
    model.to(device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, nargs="+",
                   help="One or more folder-per-class roots (Mendeley and/or GBIF)")
    p.add_argument("--out-dir", default="runs/mobile-classifier")
    p.add_argument("--model", default="mobilenetv3_small_100",
                   help="timm model; try efficientnet_lite0 if INT8 degrades")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--mixup-alpha", type=float, default=0.2, help="0 disables MixUp")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--target-precision", type=float, default=0.95,
                   help="Precision the abstain threshold must reach")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--export-tflite", action="store_true", help="Export ONNX for mobile conversion")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
