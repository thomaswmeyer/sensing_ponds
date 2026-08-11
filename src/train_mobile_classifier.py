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
    """Stratified random split into train / val / test, balanced per class.

    An earlier design held out GBIF regional records (India + Sri Lanka) as the
    whole test set, to measure transfer to the deployment region. That is a more
    honest question, but it produced an unusable split: hyacinth has by far the
    most regional records, so it landed 396 training images against 1,200 test,
    while duckweed got 18 test images. Neither number means anything.

    So: stratified random, which keeps every class properly represented in both
    sets. The cost is real and worth stating plainly -- the Mendeley images come
    from 10 capture days, so photos of the same pond minutes apart will land in
    both train and test. The headline accuracy is therefore optimistic and does
    not predict field performance.

    To keep the honest signal, the regional images are tracked as a *secondary*
    evaluation slice (see `regional` on Sample) and reported alongside the
    headline number rather than removed from training. The gap between the two is
    the domain-shift estimate.

    Grouping is still respected within the stratification: an entire capture
    session goes to one split, so at least same-minute duplicates do not straddle
    the boundary.
    """
    rng = random.Random(seed)
    test_frac = val_frac  # symmetric val/test

    by_class: dict[int, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_class[s.label][s.group].append(s)

    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []

    # Stratify per class so each split sees every class in proportion, then
    # assign whole groups within a class to avoid same-session leakage.
    for label in sorted(by_class):
        groups = sorted(by_class[label])
        rng.shuffle(groups)
        n = sum(len(by_class[label][g]) for g in groups)
        want_test, want_val = int(n * test_frac), int(n * val_frac)

        n_test = n_val = 0
        for g in groups:
            bucket = by_class[label][g]
            if n_test < want_test:
                test.extend(bucket)
                n_test += len(bucket)
            elif n_val < want_val:
                val.extend(bucket)
                n_val += len(bucket)
            else:
                train.extend(bucket)

    for name, split in (("train", train), ("val", val), ("test", test)):
        if not split:
            raise SystemExit(f"{name} split is empty -- too few distinct groups.")

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

    model = build_model(args, len(classes)).to(device)

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
    print("TEST -- stratified random split")
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

    # Secondary slice: the regional (India / Sri Lanka) images inside the test
    # set. These are the closest proxy available for deployment conditions, so
    # the gap between this and the headline is the domain-shift estimate.
    regional_idx = [i for i, s in enumerate(test_s) if s.regional]
    regional = None
    if len(regional_idx) >= 20:
        r_probs = probs[regional_idx]
        r_labels = labels[regional_idx]
        r_preds = r_probs.argmax(-1)
        regional = {
            "n": len(regional_idx),
            "accuracy": (r_preds == r_labels).float().mean().item(),
            "classes_present": sorted({classes[int(c)] for c in r_labels.unique()}),
        }
        print(
            f"\n--- Regional subset ({regional['n']} images from IN/LK) ---\n"
            f"accuracy {regional['accuracy']:.4f}   "
            f"(headline {acc:.4f}, gap {regional['accuracy'] - acc:+.4f})"
        )
        print(f"classes present: {', '.join(regional['classes_present'])}")
    else:
        print(f"\nRegional subset too small to report ({len(regional_idx)} images).")

    print(
        "\nNOTE: the headline number comes from a stratified RANDOM split. The\n"
        "Mendeley images span only 10 capture days, so near-duplicate photos of the\n"
        "same pond appear in both train and test -- this figure is optimistic and\n"
        "does not predict field performance. The regional subset above is the\n"
        "better proxy, and deployment adds further shift beyond even that."
    )

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "test_set": "stratified_random",
                "split_caveat": (
                    "Random split over data spanning 10 Mendeley capture days; "
                    "same-session leakage inflates this number."
                ),
                "test_accuracy": acc,
                "test_balanced_accuracy": bal_acc,
                "regional_subset": regional,
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
        export_mobile(model, classes, out_dir, device, temperature, thr, test_dl)


def build_model(args: argparse.Namespace, n_classes: int) -> nn.Module:
    """Create the model, loading ImageNet weights from a local file if given.

    --pretrained-weights exists because timm fetches weights through
    huggingface_hub, which uses Python's certificate bundle. On a machine behind
    a TLS-inspecting proxy that fails with CERTIFICATE_VERIFY_FAILED, and unlike
    the GBIF fetcher there is no way to route it through curl. Download the file
    separately and pass it in:

        curl -sSL -o weights.safetensors \\
          https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k/resolve/main/model.safetensors

    Training from scratch is not a viable fallback: ~6k images is nowhere near
    enough, and ImageNet initialisation is doing most of the work here.
    """
    if not args.pretrained_weights:
        return timm.create_model(
            args.model, pretrained=True, num_classes=n_classes, drop_rate=args.dropout
        )

    from safetensors.torch import load_file

    model = timm.create_model(
        args.model, pretrained=False, num_classes=n_classes, drop_rate=args.dropout
    )
    state = load_file(args.pretrained_weights)

    # The checkpoint's classifier is 1000-way ImageNet; ours is n_classes. Drop
    # any shape-mismatched tensor and let it stay randomly initialised.
    model_state = model.state_dict()
    usable = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    skipped = sorted(set(state) - set(usable))

    missing, unexpected = model.load_state_dict(usable, strict=False)
    print(
        f"Loaded {len(usable)}/{len(state)} pretrained tensors from "
        f"{args.pretrained_weights}"
    )
    if skipped:
        print(f"  reinitialised (shape mismatch): {', '.join(skipped)}")
    if unexpected:
        print(f"  WARNING unexpected keys: {len(unexpected)}")
    # A near-total mismatch means the wrong checkpoint for this architecture --
    # training would silently proceed from noise.
    if len(usable) < 0.5 * len(model_state):
        raise SystemExit(
            f"Only {len(usable)}/{len(model_state)} tensors matched -- "
            f"'{args.pretrained_weights}' does not look like weights for {args.model}."
        )
    return model


def quantize_int8(onnx_path: Path) -> Path | None:
    """Dynamic INT8 quantisation of the exported graph.

    Dynamic rather than static: static quantisation needs a calibration dataset
    and gets marginally better accuracy, but dynamic needs nothing and the
    weights are what dominate size here. Weights-only INT8 gets the 3.7x.

    Returns None (with a warning) if onnx/onnxruntime are absent -- they are
    optional deps, and training should not fail because the export tooling is
    missing.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.preprocess import quant_pre_process
    except ImportError:
        print(
            "\nNOTE: onnx/onnxruntime not installed -- skipping INT8 quantisation.\n"
            "      pip install onnx onnxruntime  (fp32 ONNX was still written)"
        )
        return None

    prep = onnx_path.with_name(onnx_path.stem + "_prep.onnx")
    int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")

    quant_pre_process(str(onnx_path), str(prep), skip_symbolic_shape=True)
    quantize_dynamic(str(prep), str(int8_path), weight_type=QuantType.QUInt8)
    prep.unlink(missing_ok=True)

    fp32_mb = onnx_path.stat().st_size / 1048576
    int8_mb = int8_path.stat().st_size / 1048576
    print(f"\nINT8:       {int8_path}")
    print(f"            {fp32_mb:.2f} MB -> {int8_mb:.2f} MB ({fp32_mb / int8_mb:.1f}x smaller)")
    return int8_path


@torch.no_grad()
def compare_int8_accuracy(
    fp32_path: Path,
    int8_path: Path,
    loader: DataLoader,
    temperature: float,
) -> dict | None:
    """Re-evaluate the quantised graph on the held-out set.

    This is the check that decides whether INT8 is shippable. MobileNetV3's
    hard-swish and squeeze-excite blocks are known to quantise badly, and the
    failure is silent -- the model keeps working, just worse. Measuring beats
    assuming.
    """
    try:
        import onnxruntime as rt
    except ImportError:
        return None

    sessions = {
        tag: rt.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        for tag, p in (("fp32", fp32_path), ("int8", int8_path))
    }

    correct = {"fp32": 0, "int8": 0}
    agree = 0
    total = 0
    for x, y in loader:
        batch = x.numpy()
        preds = {}
        for tag, sess in sessions.items():
            logits = sess.run(None, {sess.get_inputs()[0].name: batch})[0]
            preds[tag] = logits.argmax(axis=1)
            correct[tag] += int((preds[tag] == y.numpy()).sum())
        agree += int((preds["fp32"] == preds["int8"]).sum())
        total += len(y)

    result = {
        "fp32_accuracy": correct["fp32"] / total,
        "int8_accuracy": correct["int8"] / total,
        "agreement": agree / total,
    }
    delta = result["int8_accuracy"] - result["fp32_accuracy"]
    print(
        f"\nINT8 accuracy check ({total} held-out images):\n"
        f"  fp32 {result['fp32_accuracy']:.4f}   int8 {result['int8_accuracy']:.4f}"
        f"   delta {delta:+.4f}   agreement {result['agreement']:.4f}"
    )
    if delta < -0.02:
        print(
            "  WARNING: INT8 costs >2 points of accuracy. MobileNetV3 hard-swish and\n"
            "  squeeze-excite quantise poorly -- consider efficientnet_lite0 or QAT."
        )
    return result


def export_mobile(
    model: nn.Module,
    classes: list[str],
    out_dir: Path,
    device: torch.device,
    temperature: float,
    threshold: float | None,
    test_loader: DataLoader | None = None,
) -> None:
    """Export ONNX (fp32 and INT8) for browser and mobile deployment.

    INT8 is the shipping artefact: 5.79 MB -> 1.58 MB, and 1.27 MB gzipped, which
    matters on a rural connection. It is measurably *slower* per inference than
    fp32 on a small model like this -- dynamic quantisation adds quantise and
    dequantise nodes that cost more than the arithmetic they save -- but both are
    milliseconds, so size wins. See docs/model-size.md.

    Watch INT8 accuracy: MobileNetV3's hard-swish and squeeze-excite blocks
    quantise poorly. The accuracy delta is reported below; if it is material,
    switch to efficientnet_lite0 (built without those ops) or use
    quantisation-aware training. Verify -- do not assume it survived.
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
        # TorchScript exporter, not dynamo. Torch 2.9+ defaults to dynamo, which
        # pulls in onnxscript and emits a graph the ORT quantiser handles less
        # predictably. This model has no control flow, so the legacy path is both
        # sufficient and one fewer dependency.
        dynamo=False,
    )
    (out_dir / "labels.txt").write_text("\n".join(classes) + "\n")

    int8_path = quantize_int8(onnx_path)

    quant = None
    if int8_path and test_loader is not None:
        quant = compare_int8_accuracy(onnx_path, int8_path, test_loader, temperature)

    # The client cannot abstain without these. Shipping the model without the
    # temperature and threshold means shipping raw overconfident softmax, which
    # is exactly the failure the abstain path exists to prevent.
    (out_dir / "model_config.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "input_size": IMG_SIZE,
                # The INT8 graph is what ships; fp32 is kept for comparison and
                # as the source for further conversion.
                "model_file": (int8_path or onnx_path).name,
                "quantization": "int8_dynamic" if int8_path else "none",
                "preprocessing": {
                    "resize": int(IMG_SIZE * 1.14),
                    "center_crop": IMG_SIZE,
                    "mean": list(IMAGENET_MEAN),
                    "std": list(IMAGENET_STD),
                },
                "temperature": temperature,
                "abstain_threshold": threshold,
                "abstain_supported": threshold is not None,
                **({"int8_vs_fp32": quant} if quant else {}),
            },
            indent=2,
        )
    )
    print(f"\nONNX fp32:  {onnx_path}")
    print(f"Labels:     {out_dir / 'labels.txt'}")
    print(f"Config:     {out_dir / 'model_config.json'}  (preprocessing + abstain params)")
    if int8_path:
        print(
            f"\nDeploy:     cp {int8_path.name} web/public/model/plants.onnx\n"
            f"            cp model_config.json web/public/model/"
        )
    model.to(device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, nargs="+",
                   help="One or more folder-per-class roots (Mendeley and/or GBIF)")
    p.add_argument("--out-dir", default="runs/mobile-classifier")
    p.add_argument("--model", default="mobilenetv3_small_100",
                   help="timm model; try efficientnet_lite0 if INT8 degrades")
    p.add_argument("--pretrained-weights", default=None,
                   help="Local .safetensors of ImageNet weights. Use when timm "
                        "cannot reach huggingface.co (e.g. TLS-inspecting proxy)")
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
