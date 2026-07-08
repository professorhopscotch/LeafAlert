#!/usr/bin/env python3
"""
train_v5.py — direct retraining of the LeafAlert PlantDetector.

WHY v5 (see ML_QUALITY.md):
  * The shipped v4/distilled model has a severe held-out quality gap: toxic-recall
    43.5% @0.65, poison_ivy recall 51%, and — worst of all — a MOTION-BLUR CLIFF
    where 58% of toxic plants flip to "safe" under blur (vs 4.4% clean).
  * ViT distillation is NET-NEGATIVE (teacher 62% vs student 77%). v5 DROPS
    distillation entirely and trains the student directly.
  * The v4 head is over-parameterized: a 2560->512->128->4 bottleneck is ~1.38M
    params for a 4-class problem on ~1,400 images — a memorization engine. v5 uses
    a LIGHT head (Dropout -> Linear(1280, 4), ~5.1k params) on the 1280-d pooled
    EfficientNet-B0 feature (dual pooling and the CBAM attention are removed; they
    doubled head width and added capacity without moving the held-out frontier).
    An optional small bottleneck (--head bottleneck: 1280->256->4, ~0.33M) is
    available but OFF by default.

WHAT v5 CHANGES vs train_model.py:
  1. timm EfficientNet-B0 backbone, num_classes=0 (1280-d pooled feature), light head.
  2. No distillation, no mixup/cutmix by default (they masked, not fixed, the blur
     cliff). Loss = CrossEntropy + label smoothing, inverse-frequency class weights.
  3. STRONG field-failure augmentation: directional MOTION BLUR (the measured
     58%->target failure), gaussian/defocus blur, partial-occlusion RandomErasing,
     plus the existing color / scale / rotation / perspective.
  4. A SOURCE/OBSERVATION-DISJOINT train/val split (NOT a random image split): whole
     source groups (by provenance-manifest observation id if present, else by the
     filename source-token used in audit_dataset.py) go entirely to train OR val, so
     near-duplicate photos of one plant can't leak across the split and inflate val.
  5. Val transform is the PARITY transform: Resize((224,224)) squash (no crop) +
     per-channel ImageNet normalize — matches the on-device .scaleFill / eval path.

PARITY CONTRACT (honored): squash-resize to 224x224, per-channel ImageNet
normalize, and export via scripts/coreml_export.py which bakes the per-channel
normalize + softmax and takes raw 0-255 RGB. Held-out TrainingData/Testing is the
frozen leakage-free set and is NEVER read here.

Usage:
    # smoke test only (validate wiring — 1-2 steps, tiny subset, no full training):
    python3 scripts/train_v5.py --smoke

    # real run (NOT run by the scaffolding workflow):
    python3 scripts/train_v5.py --epochs 40

Output:
    checkpoints/plant_detector_v5.pth  (state_dict, loadable by --checkpoint tooling)
    LeafAlert/Resources/MLModels/PlantDetector.mlpackage  (normalize + softmax baked)
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import timm

# Make the sibling coreml_export importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coreml_export import export_coreml, IMAGENET_MEAN, IMAGENET_STD

# ─── Config / contracts ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Train POOL lives in TrainingData/{class}/ (NOT TrainingData/Testing — frozen held-out).
DEFAULT_DATA_DIR = PROJECT_ROOT / "TrainingData"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "plant_detector_v5.pth"
DEFAULT_COREML = PROJECT_ROOT / "LeafAlert" / "Resources" / "MLModels" / "PlantDetector.mlpackage"

# ImageFolder-alphabetical class order — DO NOT CHANGE. Toxic = first three.
CLASS_LABELS = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]
# The held-out frozen set — must never be pulled into the train pool.
HELD_OUT_DIRNAME = "Testing"

IMAGE_SIZE = 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ─── Reproducibility ─────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    """Seed all RNGs and make cuDNN deterministic for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Field-failure augmentation ──────────────────────────────────────
class RandomMotionBlur:
    """Directional (linear) motion blur — the augmentation that directly targets
    the measured 58% toxic->safe motion-blur cliff.

    torchvision's GaussianBlur is isotropic and does NOT reproduce the streaking
    of phone-shake / walking motion, which is what flips toxic plants to "safe" in
    the field. This applies a length-`k` line kernel at a random angle via PIL, so
    the network sees the exact degradation it fails on today.

    Operates on a PIL RGB image (place BEFORE ToTensor).
    """

    def __init__(self, p: float = 0.35, kernel_sizes=(5, 9, 13, 17)):
        self.p = p
        self.kernel_sizes = kernel_sizes

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        k = random.choice(self.kernel_sizes)
        angle = random.uniform(0.0, 180.0)
        kernel = self._line_kernel(k, angle)
        kernel = kernel / kernel.sum()  # normalize to preserve brightness
        # PIL's ImageFilter.Kernel only supports 3x3/5x5; do the convolution
        # ourselves so arbitrary streak lengths work. Reflect-pad each channel and
        # convolve via an FFT-free sliding sum over the line's nonzero offsets.
        arr = np.asarray(img, dtype=np.float32)  # H x W x 3
        offsets = np.argwhere(kernel > 0)         # (row, col) of line pixels
        c = (k - 1) // 2
        pad = c
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        out = np.zeros_like(arr)
        H, W = arr.shape[:2]
        for (ry, rx) in offsets:
            w = kernel[ry, rx]
            dy, dx = ry - c, rx - c
            out += w * padded[pad + dy: pad + dy + H, pad + dx: pad + dx + W, :]
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")

    @staticmethod
    def _line_kernel(k: int, angle_deg: float) -> np.ndarray:
        """Build a k x k kernel with a 1-px line through the center at angle_deg."""
        kernel = np.zeros((k, k), dtype=np.float32)
        c = (k - 1) / 2.0
        theta = np.deg2rad(angle_deg)
        dx, dy = np.cos(theta), np.sin(theta)
        # March along the line in both directions from the center.
        for t in np.linspace(-c, c, num=k * 2):
            x = int(round(c + dx * t))
            y = int(round(c + dy * t))
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1.0
        if kernel.sum() == 0:  # degenerate guard
            kernel[int(c), int(c)] = 1.0
        return kernel


class RandomDefocusBlur:
    """Circular defocus (out-of-focus) blur via PIL BoxBlur at a random radius.
    Complements the isotropic GaussianBlur and directional MotionBlur so the model
    is robust to the full range of field blur, not just one flavor.
    Operates on a PIL RGB image (place BEFORE ToTensor)."""

    def __init__(self, p: float = 0.20, radii=(1, 2, 3)):
        self.p = p
        self.radii = radii

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.BoxBlur(random.choice(self.radii)))


def build_train_transforms(image_size: int) -> transforms.Compose:
    """Strong augmentation aimed at the MEASURED field failures.

    Order matters: PIL-domain geometric + photometric + blur first, then ToTensor,
    Normalize, then tensor-domain RandomErasing (occlusion). Blur augmentations are
    weighted heavily because the motion-blur cliff is the single worst failure.
    """
    return transforms.Compose([
        # Scale / crop diversity (RandomResizedCrop also squashes to a square, so the
        # trained scale distribution matches the .scaleFill val/device path).
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
        # ── Field-blur block (the fix for the 58% motion-blur cliff) ──
        RandomMotionBlur(p=0.35),
        RandomDefocusBlur(p=0.20),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 2.5))], p=0.25
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Partial occlusion (leaf hidden behind another leaf / hand / branch).
        transforms.RandomErasing(p=0.30, scale=(0.02, 0.20), value="random"),
    ])


def build_val_transforms(image_size: int) -> transforms.Compose:
    """PARITY transform: squash the whole image to image_size x image_size (NO crop)
    then per-channel ImageNet normalize — matches the on-device .scaleFill / eval
    path. Deterministic; no augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Dataset with source-group provenance ────────────────────────────
def _source_token(filename: str) -> str:
    """Group key from the filename, matching scripts/audit_dataset.py's convention:
    strip a trailing _<digits> (and an optional _test) to get the source/sub-taxon
    token (e.g. blackberry_0007.jpg -> blackberry, poison_ivy_0000.jpg -> poison_ivy).
    Photos sharing a token are treated as the same SOURCE so they can't straddle the
    train/val split."""
    stem = Path(filename).stem
    parts = stem.split("_")
    while parts and (parts[-1].isdigit() or parts[-1] == "test"):
        parts.pop()
    return "_".join(parts) if parts else "(none)"


def _load_provenance(data_dir: Path):
    """Optional provenance map for a truly OBSERVATION-disjoint split.

    If the data-expansion pipeline has written a manifest recording the source
    observation id per image, we key the split on that (best: no two photos of the
    same physical plant straddle the split). Supported, in priority order:
      * TrainingData/provenance.json  -> {"relative/path.jpg": {"observation_id": ...}}
        or {"relative/path.jpg": "<obs_id>"}
      * TrainingData/manifest.jsonl   -> one JSON obj/line with keys
        {"path" | "file" | "filename"} and {"observation_id" | "obs_id" | "source"}
    Returns dict[str filename_or_relpath -> str group_id], or {} if none found.
    """
    import json

    mapping: dict[str, str] = {}
    pj = data_dir / "provenance.json"
    if pj.exists():
        try:
            raw = json.loads(pj.read_text())
            for k, v in raw.items():
                gid = v.get("observation_id") if isinstance(v, dict) else v
                if gid is not None:
                    mapping[Path(k).name] = f"obs:{gid}"
        except Exception as e:  # provenance is best-effort; never fatal
            print(f"  [provenance] skipping {pj.name}: {e}")
    ml = data_dir / "manifest.jsonl"
    if ml.exists():
        try:
            for line in ml.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                path = obj.get("path") or obj.get("file") or obj.get("filename")
                gid = (
                    obj.get("observation_id")
                    or obj.get("obs_id")
                    or obj.get("source")
                )
                if path and gid is not None:
                    mapping[Path(path).name] = f"obs:{gid}"
        except Exception as e:
            print(f"  [provenance] skipping {ml.name}: {e}")
    return mapping


def scan_dataset(data_dir: Path):
    """Scan TrainingData/{class}/ (EXCLUDING the frozen held-out Testing dir).
    Returns (records, class_to_idx). Each record: {path, label, group}.
    `group` is obs:<id> if provenance exists for the file, else src:<class>/<token>.
    """
    data_dir = Path(data_dir)
    provenance = _load_provenance(data_dir)
    records = []
    for idx, cls in enumerate(CLASS_LABELS):
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
                continue
            obs = provenance.get(p.name)
            # Fall back to the filename source-token, namespaced by class so the
            # same token in different classes never merges into one group.
            group = obs if obs is not None else f"src:{cls}/{_source_token(p.name)}"
            records.append({"path": str(p), "label": idx, "group": group})
    if not records:
        raise RuntimeError(
            f"No images found under {data_dir}/{{{','.join(CLASS_LABELS)}}}. "
            "Populate the train pool first."
        )
    # The frozen held-out set (TrainingData/Testing) is excluded structurally: we
    # only iterate the CLASS_LABELS subdirs, never HELD_OUT_DIRNAME. Assert it in
    # case a class dir is ever symlinked to include it.
    for r in records[: min(len(records), 50)] + records[-min(len(records), 50):]:
        if HELD_OUT_DIRNAME in Path(r["path"]).parts:
            raise AssertionError(
                f"Held-out '{HELD_OUT_DIRNAME}' leaked into the train pool: {r['path']}"
            )
    return records, {c: i for i, c in enumerate(CLASS_LABELS)}


def _subshard_group(recs, n_shards: int):
    """Deterministically split one over-coarse group into <= n_shards sub-groups by
    a stable hash of each file's basename. Photos of one iNaturalist observation are
    downloaded consecutively (sequential filenames), so a hash of the name keeps a
    plant's frames together far better than a random image split would, while giving
    the splitter enough distinct groups to hit the val fraction. Returns a
    {sub_group_id -> [recs]} dict."""
    import hashlib

    shards = defaultdict(list)
    for r in recs:
        h = hashlib.sha1(Path(r["path"]).name.encode()).hexdigest()
        sub_gid = f"{r['group']}#s{int(h, 16) % n_shards}"
        # Stamp the sub-group id back onto the record so downstream disjointness
        # checks (main()'s overlap assertion) see the ACTUAL split unit, not the
        # coarse original token.
        r["group"] = sub_gid
        shards[sub_gid].append(r)
    return shards


def _grouped_by_class(records, val_frac: float):
    """label -> {group -> [recs]}, re-sharding any class whose real groups are too
    coarse to reach val_frac (e.g. the toxic classes today have ONE filename token
    each, which would dump the whole class onto one side of the split). Re-sharding
    only triggers for the fallback filename-token grouping; once real per-observation
    provenance exists, classes already have many groups and this is a no-op."""
    by_class = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_class[r["label"]][r["group"]].append(r)

    for label, groups in list(by_class.items()):
        n_total = sum(len(v) for v in groups.values())
        # No single group may exceed ~half the val target. The original pool shares
        # ONE filename token per class (hundreds of DISTINCT plants under e.g.
        # "poison_oak"), so an un-split giant group dumps the whole class onto one
        # side and starves the other — the collapse we just debugged. Subshard any
        # oversized group by a stable filename hash (a plant's real frames stay
        # together for small, correctly-grouped pilot observations; the coarse
        # original token is effectively image-split, which is correct — those are
        # not one source). This makes val_frac addressable in fine steps.
        cap = max(1, int(round(n_total * val_frac * 0.5)))
        regrouped = {}
        for gid, recs in groups.items():
            if len(recs) > cap:
                n_shards = -(-len(recs) // cap)  # ceil div -> every shard <= cap
                regrouped.update(_subshard_group(recs, n_shards))
            else:
                regrouped[gid] = recs
        by_class[label] = defaultdict(list, regrouped)
    return by_class


def source_disjoint_split(records, val_frac: float, seed: int):
    """Split records so that WHOLE source groups go to train OR val (never both).

    Greedy per-class bin-packing of groups toward the target val fraction, so each
    class still lands ~val_frac in val while keeping every group intact. This is the
    key defense against the memorization/overfitting the held-out gap exposed: a
    random image split leaks near-duplicate photos of the same plant across the
    boundary and inflates val accuracy. Classes with too-coarse grouping are
    re-sharded first (see _grouped_by_class) so no class is starved on either side.
    """
    rng = random.Random(seed)
    by_class = _grouped_by_class(records, val_frac)

    train_recs, val_recs = [], []
    for label, groups in sorted(by_class.items()):
        group_items = sorted(groups.items())  # deterministic pre-shuffle order
        rng.shuffle(group_items)
        n_total = sum(len(v) for _, v in group_items)
        n_groups = len(group_items)
        target_val = n_total * val_frac

        # Greedily fill VAL toward target_val one whole group at a time, but never
        # take the last remaining train group — every class MUST keep training data.
        # (The old logic over-filled val and could starve a class to 0 train, which
        # then produced pathological inverse-frequency class weights and collapse.)
        val_count = 0
        n_val_groups = 0
        for gi, (_, recs) in enumerate(group_items):
            groups_left = n_groups - gi
            train_groups_so_far = gi - n_val_groups
            must_keep_for_train = groups_left == 1 and train_groups_so_far == 0
            if val_count < target_val and not must_keep_for_train:
                val_recs.extend(recs)
                val_count += len(recs)
                n_val_groups += 1
            else:
                train_recs.extend(recs)
    rng.shuffle(train_recs)
    rng.shuffle(val_recs)
    return train_recs, val_recs


class RecordDataset(Dataset):
    """Loads (image, label) from scanned records with a given transform."""

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.transform(img), r["label"]


# ─── Model: EfficientNet-B0 + LIGHT head ─────────────────────────────
class PlantDetectorV5(nn.Module):
    """timm EfficientNet-B0 (1280-d pooled feature) + a LIGHT classifier head.

    Head choices (justification in the module docstring):
      * "linear"     : Dropout(p) -> Linear(1280, num_classes)  (~5.1k params) [default]
      * "bottleneck" : Dropout(p) -> Linear(1280,256) -> BN -> ReLU -> Dropout ->
                       Linear(256, num_classes)                 (~0.33M params)
    Both are far below the old ~1.38M-param 2560->512->128->4 head that memorized a
    1,400-image set.
    """

    def __init__(self, num_classes: int, head: str = "linear",
                 dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        # num_classes=0 -> backbone returns the 1280-d globally-pooled feature.
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 1280
        if head == "linear":
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes),
            )
        elif head == "bottleneck":
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(f"unknown head '{head}' (expected linear|bottleneck)")
        self.head_kind = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)      # [B, 1280]
        return self.head(feat)       # [B, num_classes] logits


def freeze_backbone(model: PlantDetectorV5, freeze: bool):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


# ─── Class weights ───────────────────────────────────────────────────
def compute_class_weights(records, num_classes: int) -> torch.Tensor:
    counts = Counter(r["label"] for r in records)
    total = sum(counts.values())
    # A class with zero training samples yields a runaway inverse-frequency weight
    # (e.g. 3.9 vs 0.01) that collapses the model onto the starved class. This must
    # never happen; surface it loudly rather than train a broken model.
    empty = [CLASS_LABELS[i] for i in range(num_classes) if counts.get(i, 0) == 0]
    if empty:
        raise ValueError(
            f"No training samples for class(es) {empty}. The train/val split "
            f"starved a class — check source_disjoint_split / --val-frac. "
            f"Per-class train counts: "
            f"{ {CLASS_LABELS[i]: counts.get(i, 0) for i in range(num_classes)} }"
        )
    weights = torch.tensor(
        [total / (num_classes * max(counts.get(i, 0), 1)) for i in range(num_classes)],
        dtype=torch.float32,
    )
    return weights / weights.mean()


# ─── Train / eval loops ──────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              max_steps: int | None = None):
    model.train() if train else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for step, (inputs, labels) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    torch.set_grad_enabled(True)
    n = max(total, 1)
    return running_loss / n, correct / n


def toxic_recall(model, loader, device, num_toxic: int = 3) -> float:
    """Held-out-style safety metric: fraction of TOXIC samples (labels 0..2) whose
    argmax is any toxic class (i.e. NOT called safe_plants). This is the number that
    matters for a poison detector; we surface it during val to steer selection."""
    model.eval()
    toxic_seen, toxic_ok = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            preds = model(inputs).argmax(1).cpu()
            for pred, lab in zip(preds, labels):
                if lab.item() < num_toxic:
                    toxic_seen += 1
                    if pred.item() < num_toxic:
                        toxic_ok += 1
    return toxic_ok / max(toxic_seen, 1)


# ─── Export ──────────────────────────────────────────────────────────
def export(model: PlantDetectorV5, out_path: Path, image_size: int):
    """Export via the shared helper (bakes per-channel normalize + softmax; takes
    raw 0-255 RGB). Prints the resulting path."""
    short_description = (
        "Detects poison ivy, poison oak, and poison sumac from camera images. "
        "EfficientNet-B0 (timm) with a light head, trained directly (v5, no "
        "distillation) with motion-blur / defocus / occlusion augmentation. "
        f"Classes: {', '.join(CLASS_LABELS)}"
    )
    model.eval().cpu()
    export_coreml(
        model, CLASS_LABELS, out_path, image_size,
        short_description=short_description,
        version="5.0.0",
    )
    print(f"\nCore ML model exported to: {out_path.resolve()}")


# ─── Main ────────────────────────────────────────────────────────────
def build_argparser():
    p = argparse.ArgumentParser(
        description="LeafAlert PlantDetector v5 — direct training (no distillation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                   help="Train POOL root containing {class}/ subdirs (NOT Testing).")
    p.add_argument("--epochs", type=int, default=40, help="Total training epochs.")
    p.add_argument("--head-epochs", type=int, default=5,
                   help="Epochs with the backbone frozen (head warmup) before fine-tuning.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="Head LR (backbone gets lr/10).")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--head", choices=["linear", "bottleneck"], default="linear",
                   help="Classifier head: light linear (default) or small bottleneck.")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Fraction of each class (by whole source groups) held for val.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0, help="0 is safe on macOS.")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT),
                   help="Where to write the best state_dict.")
    p.add_argument("--coreml-out", type=str, default=str(DEFAULT_COREML))
    p.add_argument("--no-export", action="store_true",
                   help="Skip Core ML export (e.g. for a pure training run).")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Do not download ImageNet weights (offline smoke).")
    p.add_argument("--smoke", action="store_true",
                   help="Wiring check: tiny subset, 1-2 steps/epoch, no full training.")
    return p


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU (MPS)"
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA GPU"
    return torch.device("cpu"), "CPU"


def main():
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    print("=" * 64)
    print("LeafAlert PlantDetector — Training v5 (direct, no distillation)")
    print("=" * 64)

    device, dev_name = pick_device()
    print(f"Device: {dev_name}")

    # ── Data ──
    data_dir = Path(args.data_dir)
    print(f"\nScanning train pool: {data_dir}  (held-out '{HELD_OUT_DIRNAME}' excluded)")
    records, class_to_idx = scan_dataset(data_dir)
    print(f"Classes (fixed order): {CLASS_LABELS}")
    per_class = Counter(r["label"] for r in records)
    for i, c in enumerate(CLASS_LABELS):
        print(f"  {c:16s}: {per_class.get(i, 0)} images")
    n_groups = len({r["group"] for r in records})
    print(f"Total images: {len(records)} across {n_groups} source groups")

    train_recs, val_recs = source_disjoint_split(records, args.val_frac, args.seed)

    if args.smoke:
        # Tiny, class-balanced subsets so every class is exercised in 1-2 steps.
        def take_per_class(recs, k):
            picked, seen = [], Counter()
            for r in recs:
                if seen[r["label"]] < k:
                    picked.append(r)
                    seen[r["label"]] += 1
            return picked
        train_recs = take_per_class(train_recs, 3)
        # Take val ONLY from the (disjoint) val records. Do NOT fall back to
        # train_recs: that would put an identical group on both sides and the
        # disjointness assertion below would immediately crash. If the tiny
        # split yields no val records, run without val/toxic-recall eval.
        val_recs = take_per_class(val_recs, 2)
        print(f"\n[SMOKE] train={len(train_recs)} val={len(val_recs)} (tiny subset)")
        if not val_recs:
            print("[SMOKE] no disjoint val records available; "
                  "skipping val / toxic-recall eval for this smoke run.")

    # Sanity: no source group may appear on both sides of the split.
    overlap = {r["group"] for r in train_recs} & {r["group"] for r in val_recs}
    if overlap:
        raise AssertionError(f"Split leak — groups in BOTH train and val: {sorted(overlap)[:5]}")
    print(f"Split: train={len(train_recs)}  val={len(val_recs)}  (source-disjoint, no group overlap)")

    train_ds = RecordDataset(train_recs, build_train_transforms(IMAGE_SIZE))
    val_ds = RecordDataset(val_recs, build_val_transforms(IMAGE_SIZE))

    bs = min(args.batch_size, len(train_ds)) if args.smoke else args.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=max(bs, 1), shuffle=True,
        num_workers=args.num_workers, drop_last=(not args.smoke and len(train_ds) > bs),
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(min(args.batch_size, len(val_ds)), 1),
        shuffle=False, num_workers=args.num_workers,
    )

    # ── Model ──
    pretrained = not (args.no_pretrained or args.smoke)  # smoke stays offline
    print(f"\nBuilding EfficientNet-B0 + '{args.head}' head "
          f"(pretrained={pretrained})...")
    model = PlantDetectorV5(
        num_classes=len(CLASS_LABELS), head=args.head,
        dropout=args.dropout, pretrained=pretrained,
    ).to(device)
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"Head params: {head_params:,}  (old v4 head was ~1.38M)")

    class_weights = compute_class_weights(train_recs, len(CLASS_LABELS)).to(device)
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )

    max_steps = 2 if args.smoke else None
    total_epochs = 2 if args.smoke else args.epochs
    head_epochs = 1 if args.smoke else min(args.head_epochs, total_epochs)

    best_metric, best_state = -1.0, None

    def epoch_line(tag, ep, tr_loss, tr_acc, va_loss, va_acc, tr, lr):
        print(f"  [{tag}] Epoch {ep:2d} | "
              f"train {tr_loss:.4f}/{tr_acc:.1%} | "
              f"val {va_loss:.4f}/{va_acc:.1%} | "
              f"toxic-recall {tr:.1%} | lr {lr:.2e}")

    # ── Phase 1: head warmup (backbone frozen) ──
    if head_epochs > 0:
        print(f"\n--- Phase 1: head warmup ({head_epochs} epoch(s), backbone frozen) ---")
        freeze_backbone(model, True)
        opt = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr, weight_decay=args.weight_decay,
        )
        for ep in range(1, head_epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device,
                                        train=True, max_steps=max_steps)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, opt, device,
                                        train=False, max_steps=max_steps)
            tr = toxic_recall(model, val_loader, device)
            epoch_line("warm", ep, tr_loss, tr_acc, va_loss, va_acc, tr,
                       opt.param_groups[0]["lr"])
            # Select the best GENERALIZER by val accuracy. Selecting on toxic-recall
            # alone is degenerate — a model that predicts a toxic class for every
            # image scores 100% toxic-recall, so it would always win. On-device
            # per-class thresholds (ToxicityThresholds) tune the safety/precision
            # tradeoff at inference; here we just want the best-fitting model.
            metric = va_acc
            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ── Phase 2: fine-tune whole network (discriminative LR) ──
    ft_epochs = total_epochs - head_epochs
    if ft_epochs > 0:
        print(f"\n--- Phase 2: fine-tune backbone + head ({ft_epochs} epoch(s)) ---")
        freeze_backbone(model, False)
        opt = optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr / 10},
            {"params": model.head.parameters(), "lr": args.lr},
        ], weight_decay=args.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ft_epochs)
        for ep in range(1, ft_epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device,
                                        train=True, max_steps=max_steps)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, opt, device,
                                        train=False, max_steps=max_steps)
            tr = toxic_recall(model, val_loader, device)
            epoch_line("ft", head_epochs + ep, tr_loss, tr_acc, va_loss, va_acc, tr,
                       opt.param_groups[-1]["lr"])
            sched.step()
            metric = va_acc  # select best generalizer, not the degenerate all-toxic model
            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ── Restore best, save checkpoint ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    final_tr = toxic_recall(model, val_loader, device)
    _, final_acc = run_epoch(model, val_loader, criterion, None, device, train=False,
                             max_steps=max_steps)
    print(f"\nBest val toxic-recall (selection): {final_tr:.1%} | val acc {final_acc:.1%}")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt_path))
    print(f"Saved checkpoint: {ckpt_path.resolve()}")

    # ── Export ──
    if args.no_export or args.smoke:
        print("\n[export skipped]" + ("  (--smoke)" if args.smoke else "  (--no-export)"))
    else:
        print("\n--- Exporting to Core ML (per-channel normalize + softmax baked) ---")
        export(model, Path(args.coreml_out), IMAGE_SIZE)

    print("\n" + "=" * 64)
    print("v5 training complete.")
    print(f"  checkpoint : {ckpt_path.resolve()}")
    if not (args.no_export or args.smoke):
        print(f"  coreml     : {Path(args.coreml_out).resolve()}")
    print("Next: evaluate on the FROZEN held-out set, then RE-DERIVE thresholds.")
    print("  * scripts/evaluate_model.py --coreml <mlpackage> --data TrainingData/Testing")
    print("    (evaluate via the .mlpackage — it is self-contained. NOTE: the")
    print("     --checkpoint torch path in evaluate_model.py loads distill_model's")
    print("     PlantDetectorNet and will NOT accept this v5 state_dict; point that")
    print("     tooling at PlantDetectorV5 or just eval the exported Core ML model.)")
    print("  * scripts/calibration_report.py + robustness_report.py")
    print("  * RE-DERIVE LeafAlert/Models/ToxicityThresholds.swift (model-specific).")
    print("=" * 64)


if __name__ == "__main__":
    main()
