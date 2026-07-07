#!/usr/bin/env python3
"""
Real-world robustness report for the LeafAlert PlantDetector student model.

Estimates the gap between lab accuracy and "phone-in-the-woods" performance by
applying realistic outdoor-hiking image perturbations to the evaluation images
and measuring how toxic-recall and overall accuracy degrade relative to clean
inputs. Perturbations are ranked by how much they hurt TOXIC-RECALL — the
safety-critical (dangerous) direction, since a toxic plant classified as safe
(a false negative) is the failure that gets a hiker into poison ivy.

What it does
------------
1. Loads the distilled student checkpoint (checkpoints/student_distilled.pth)
   into PlantDetectorNet from scripts/distill_model.py.
2. Builds an evaluation image list from a class-labeled ImageFolder-style
   directory (default: TrainingData/, the canonical source images).
3. For "clean" and each perturbation, resizes each image to 224x224 by
   SQUASHING the whole frame (Resize((224,224)), NO center crop) and applies
   per-channel ImageNet normalization — the exact parity contract used
   on-device (Vision .scaleFill). This makes the torch numbers comparable to
   the shipped Core ML model.
4. Reports clean vs perturbed accuracy, toxic-recall, and toxic->safe
   miss-rate, plus the degradation (delta) for each perturbation, ranked by
   toxic-recall drop.

PARITY CONTRACT (canonical): resize by squashing to 224x224 (no crop), then
per-channel ImageNet normalize mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225].
The perturbations operate on the ORIGINAL full-resolution PIL image (as a real
camera frame would be degraded) and the squash-resize + normalize happen
afterward, exactly as the on-device pipeline degrades then feeds Vision.

IMPORTANT CAVEATS (printed in the report too)
---------------------------------------------
* The shipped checkpoint was trained BEFORE deterministic seeding was added,
  so its original train/val split is NOT reconstructable. Any metric on
  TrainingData images the model trained on is OPTIMISTIC (train-set). The
  ABSOLUTE clean numbers here are therefore an upper bound; the RELATIVE
  degradation (clean -> perturbed) is the robust, load-bearing signal.
* This measures robustness to *synthetic* perturbations of existing photos, not
  a true out-of-distribution field test. Real hiking photos differ in ways no
  augmentation captures (novel backgrounds, species, lighting). Treat this as a
  lower bound on the lab-vs-field gap, not the full gap.

Usage
-----
    .venv/bin/python scripts/robustness_report.py
    .venv/bin/python scripts/robustness_report.py --data-dir TrainingData_split/test
    .venv/bin/python scripts/robustness_report.py --threshold 0.5 --max-per-class 100
    .venv/bin/python scripts/robustness_report.py --device cpu --json out.json
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

# Make sibling scripts importable regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_model import PlantDetectorNet  # noqa: E402

# ─── Canonical config ────────────────────────────────────────────────
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ImageFolder-alphabetical canonical class order.
CLASS_NAMES = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]
TOXIC_CLASSES = {"poison_ivy", "poison_oak", "poison_sumac"}
SAFE_CLASS = "safe_plants"

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "student_distilled.pth"
DEFAULT_DATA_DIR = PROJECT_ROOT / "TrainingData"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─── Preprocessing (parity contract) ─────────────────────────────────

def squash_resize_normalize(img: Image.Image) -> np.ndarray:
    """Squash the full PIL image to 224x224 (NO crop), then per-channel
    ImageNet normalize. Returns a CHW float32 numpy array.

    This is the exact on-device parity contract (Vision .scaleFill + baked
    per-channel normalize)."""
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1)).copy()  # CHW


# ─── Perturbations (operate on the ORIGINAL full-res PIL image) ──────
#
# Each function takes a PIL.Image (RGB) and a rng (np.random.Generator for the
# stochastic ones) and returns a perturbed PIL.Image. They are designed to
# mimic real outdoor-hiking capture degradation: hand shake, out-of-focus,
# glare/shade, camera JPEG, thumb/leaf occlusion, zoom framing, tilt.

def _to_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img


def motion_blur(img, rng, kernel=15):
    """Directional (horizontal) motion blur — simulates hand shake / walking."""
    img = _to_rgb(img)
    arr = np.asarray(img, dtype=np.float32)
    k = np.zeros((kernel, kernel), dtype=np.float32)
    k[kernel // 2, :] = 1.0 / kernel
    # Separable-ish horizontal convolution per channel via cumulative box.
    pad = kernel // 2
    padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for off in range(kernel):
        out += padded[:, off:off + arr.shape[1], :]
    out /= kernel
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def gaussian_blur(img, rng, radius=3.0):
    """Out-of-focus / soft-focus blur."""
    return _to_rgb(img).filter(ImageFilter.GaussianBlur(radius=radius))


def brightness_shift(img, rng, factor=1.6):
    """Multiplicative brightness change (bright sun glare if >1, deep shade if <1)."""
    arr = np.asarray(_to_rgb(img), dtype=np.float32) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def contrast_shift(img, rng, factor=0.55):
    """Contrast change around mid-gray (hazy / washed-out lighting)."""
    arr = np.asarray(_to_rgb(img), dtype=np.float32)
    arr = (arr - 128.0) * factor + 128.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg_compression(img, rng, quality=15):
    """Aggressive JPEG re-compression (low-bandwidth upload / camera artifacts)."""
    buf = io.BytesIO()
    _to_rgb(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def partial_occlusion(img, rng, frac=0.30):
    """Occlude a random rectangle covering ~frac of the frame area with a
    neutral gray patch — simulates a thumb over the lens or a leaf/branch in
    front of the target plant."""
    img = _to_rgb(img).copy()
    w, h = img.size
    arr = np.asarray(img).copy()
    # Rectangle with area ~ frac of the image, random aspect within reason.
    aspect = rng.uniform(0.6, 1.6)
    rect_area = frac * w * h
    rw = int(min(w, max(8, (rect_area * aspect) ** 0.5)))
    rh = int(min(h, max(8, rect_area / max(1, rw))))
    x0 = rng.integers(0, max(1, w - rw))
    y0 = rng.integers(0, max(1, h - rh))
    arr[y0:y0 + rh, x0:x0 + rw, :] = 114  # neutral gray
    return Image.fromarray(arr)


def scale_zoom(img, rng, factor=1.5):
    """Center-crop then re-expand to simulate digital zoom / getting closer,
    losing surrounding context (framing tighter than training crops)."""
    img = _to_rgb(img)
    w, h = img.size
    cw, ch = int(w / factor), int(h / factor)
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def rotation(img, rng, degrees=20):
    """In-plane rotation (camera tilt) with edge-replicate fill (no black
    corners, which would themselves be an artifact)."""
    img = _to_rgb(img)
    # expand=False keeps size; fill with replicate by rotating on a padded copy.
    return img.rotate(degrees, resample=Image.BILINEAR, expand=False, fillcolor=(114, 114, 114))


def gaussian_noise(img, rng, sigma=25.0):
    """Additive sensor noise (low-light ISO grain)."""
    arr = np.asarray(_to_rgb(img), dtype=np.float32)
    noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


# Registry: name -> (function, kwargs). Ordered roughly by capture-realism.
PERTURBATIONS = {
    "motion_blur_k15":        (motion_blur,       {"kernel": 15}),
    "gaussian_blur_r3":       (gaussian_blur,     {"radius": 3.0}),
    "brightness_up_1.6x":     (brightness_shift,  {"factor": 1.6}),
    "brightness_down_0.5x":   (brightness_shift,  {"factor": 0.5}),
    "contrast_down_0.55x":    (contrast_shift,    {"factor": 0.55}),
    "jpeg_q15":               (jpeg_compression,  {"quality": 15}),
    "occlusion_30pct":        (partial_occlusion, {"frac": 0.30}),
    "zoom_1.5x":              (scale_zoom,        {"factor": 1.5}),
    "rotation_20deg":         (rotation,          {"degrees": 20}),
    "gaussian_noise_s25":     (gaussian_noise,    {"sigma": 25.0}),
}


# ─── Data loading ────────────────────────────────────────────────────

def collect_images(data_dir: Path, max_per_class: int | None):
    """Return list of (path, class_name) for class subdirs matching CLASS_NAMES.

    Only directories whose basename is in CLASS_NAMES are used (so a stray
    'Testing' dir with a handful of images is ignored)."""
    samples = []
    per_class = defaultdict(int)
    for cls in CLASS_NAMES:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        paths = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        for p in paths:
            if max_per_class is not None and per_class[cls] >= max_per_class:
                break
            samples.append((p, cls))
            per_class[cls] += 1
    return samples, dict(per_class)


# ─── Inference ───────────────────────────────────────────────────────

def run_batch(model, batch_arrays, device):
    """Run a list of CHW numpy arrays through the model, return softmax probs
    as a numpy array [N, num_classes]."""
    x = torch.from_numpy(np.stack(batch_arrays, axis=0)).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()


def evaluate_condition(model, samples, device, perturb, batch_size, rng_seed):
    """Evaluate the model on all samples under a given perturbation function.

    perturb: None for clean, else a callable(img, rng)->img.
    Returns dict of predictions and probabilities aligned to `samples`.
    """
    rng = np.random.default_rng(rng_seed)
    all_probs = []
    batch_arrays = []
    for path, _cls in samples:
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:  # unreadable file — skip with a sentinel
            all_probs.append(None)
            continue
        if perturb is not None:
            img = perturb(img, rng)
        arr = squash_resize_normalize(img)
        batch_arrays.append(arr)
        # Flush full batches.
        if len(batch_arrays) == batch_size:
            probs = run_batch(model, batch_arrays, device)
            all_probs.extend(list(probs))
            batch_arrays = []
    if batch_arrays:
        probs = run_batch(model, batch_arrays, device)
        all_probs.extend(list(probs))
    return all_probs


# ─── Metrics ─────────────────────────────────────────────────────────

def compute_metrics(samples, probs, threshold):
    """Compute accuracy, toxic-recall, toxic->safe miss-rate, and per-class
    recall from aligned samples/probs.

    Definitions (safety-oriented):
      * argmax prediction = class with max softmax prob.
      * accuracy = fraction of argmax predictions equal to the true label.
      * toxic-recall (argmax) = of true-toxic images, fraction whose argmax is
        ANY toxic class (correctly flagged as dangerous, even if the exact
        toxic species is wrong — a hiker just needs "don't touch").
      * toxic-recall (thresholded) = of true-toxic images, fraction where the
        SUMMED toxic probability >= threshold. This mirrors an app that alerts
        when total toxic confidence crosses a bar; it is the operationally
        relevant recall.
      * toxic->safe miss-rate = of true-toxic images, fraction whose argmax is
        safe_plants. This is the outright dangerous false negative.
      * toxic->below-threshold miss-rate = of true-toxic images, fraction where
        summed toxic prob < threshold (no alert fires). The dangerous silence.
    """
    class_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    toxic_idx = [class_idx[c] for c in CLASS_NAMES if c in TOXIC_CLASSES]
    safe_i = class_idx[SAFE_CLASS]

    n = 0
    correct = 0
    toxic_total = 0
    toxic_recall_argmax = 0
    toxic_recall_thresh = 0
    toxic_to_safe = 0
    toxic_below_thresh = 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    for (path, true_cls), p in zip(samples, probs):
        if p is None:
            continue
        n += 1
        pred_i = int(np.argmax(p))
        pred_cls = CLASS_NAMES[pred_i]
        true_i = class_idx[true_cls]
        per_class_total[true_cls] += 1
        if pred_i == true_i:
            correct += 1
            per_class_correct[true_cls] += 1

        if true_cls in TOXIC_CLASSES:
            toxic_total += 1
            toxic_prob_sum = float(sum(p[i] for i in toxic_idx))
            if pred_cls in TOXIC_CLASSES:
                toxic_recall_argmax += 1
            if toxic_prob_sum >= threshold:
                toxic_recall_thresh += 1
            else:
                toxic_below_thresh += 1
            if pred_i == safe_i:
                toxic_to_safe += 1

    def safe_div(a, b):
        return a / b if b else float("nan")

    per_class_recall = {
        c: safe_div(per_class_correct[c], per_class_total[c]) for c in CLASS_NAMES
    }
    return {
        "n": n,
        "accuracy": safe_div(correct, n),
        "toxic_total": toxic_total,
        "toxic_recall_argmax": safe_div(toxic_recall_argmax, toxic_total),
        "toxic_recall_thresh": safe_div(toxic_recall_thresh, toxic_total),
        "toxic_to_safe_miss_rate": safe_div(toxic_to_safe, toxic_total),
        "toxic_below_thresh_miss_rate": safe_div(toxic_below_thresh, toxic_total),
        "per_class_recall": per_class_recall,
    }


# ─── Reporting ───────────────────────────────────────────────────────

def fmt_pct(x):
    return "  nan " if x != x else f"{x:6.1%}"


def main():
    ap = argparse.ArgumentParser(
        description="Real-world robustness report for LeafAlert PlantDetector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="ImageFolder-style dir with poison_ivy/oak/sumac/safe_plants subdirs.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Summed-toxic-prob alert threshold for thresholded recall.")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="Cap images per class (for a fast run). Default: all.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=1234,
                    help="Seed for stochastic perturbations (occlusion, noise).")
    ap.add_argument("--json", type=Path, default=None,
                    help="Optional path to write full results as JSON.")
    args = ap.parse_args()

    # Device.
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Model.
    model = PlantDetectorNet(len(CLASS_NAMES))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)

    # Data.
    samples, per_class = collect_images(args.data_dir, args.max_per_class)
    if not samples:
        print(f"ERROR: no images found under {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 78)
    print("LeafAlert — Real-world Robustness Report")
    print("=" * 78)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Device     : {device}")
    print(f"Threshold  : summed-toxic-prob >= {args.threshold} fires an alert")
    print(f"Images/class: " + ", ".join(f"{c}={per_class.get(c,0)}" for c in CLASS_NAMES))
    print(f"Total eval images: {len(samples)}")
    print()
    print("PARITY: squash-resize to 224x224 (no crop) + per-channel ImageNet")
    print("        normalize — matches Vision .scaleFill + baked normalize on-device.")
    print()
    print("CAVEAT: shipped checkpoint predates deterministic seeding; its original")
    print("        train/val split is NOT reconstructable. TrainingData metrics are")
    print("        TRAIN-SET OPTIMISTIC. The load-bearing signal is the DEGRADATION")
    print("        (clean -> perturbed delta), not the absolute clean numbers.")
    print("=" * 78)

    # Clean baseline.
    clean_probs = evaluate_condition(model, samples, device, None,
                                     args.batch_size, args.seed)
    clean = compute_metrics(samples, clean_probs, args.threshold)

    # Perturbations.
    results = {"clean": clean}
    for i, (name, (fn, kwargs)) in enumerate(PERTURBATIONS.items()):
        def perturb(img, rng, fn=fn, kwargs=kwargs):
            return fn(img, rng, **kwargs)
        probs = evaluate_condition(model, samples, device, perturb,
                                   args.batch_size, args.seed + 1 + i)
        results[name] = compute_metrics(samples, probs, args.threshold)

    # ─── Clean summary ───────────────────────────────────────────────
    print("\nCLEAN BASELINE (train-set optimistic):")
    print(f"  accuracy                 : {fmt_pct(clean['accuracy'])}")
    print(f"  toxic-recall (argmax)    : {fmt_pct(clean['toxic_recall_argmax'])}  "
          f"(true-toxic flagged as some toxic class)")
    print(f"  toxic-recall (thr>={args.threshold})  : {fmt_pct(clean['toxic_recall_thresh'])}  "
          f"(summed toxic prob crosses alert bar)")
    print(f"  toxic->safe miss-rate    : {fmt_pct(clean['toxic_to_safe_miss_rate'])}  "
          f"(DANGEROUS: toxic argmaxed as safe)")
    print(f"  toxic->below-thr silence : {fmt_pct(clean['toxic_below_thresh_miss_rate'])}  "
          f"(DANGEROUS: no alert fires)")
    print("  per-class recall (argmax):")
    for c in CLASS_NAMES:
        print(f"    {c:14s}: {fmt_pct(clean['per_class_recall'][c])}")

    # ─── Per-perturbation table ──────────────────────────────────────
    print("\n" + "=" * 78)
    print("PER-PERTURBATION DEGRADATION (delta vs clean; negative = worse)")
    print("=" * 78)
    header = (f"{'perturbation':22s} {'acc':>7s} {'dAcc':>7s} "
              f"{'txRec_am':>9s} {'dRec_am':>8s} {'txRec_th':>9s} {'dRec_th':>8s} "
              f"{'tx->safe':>9s} {'silence':>8s}")
    print(header)
    print("-" * len(header))

    def row(name, m):
        d_acc = m["accuracy"] - clean["accuracy"]
        d_ram = m["toxic_recall_argmax"] - clean["toxic_recall_argmax"]
        d_rth = m["toxic_recall_thresh"] - clean["toxic_recall_thresh"]
        return (f"{name:22s} {fmt_pct(m['accuracy'])} {d_acc:+7.1%} "
                f"{fmt_pct(m['toxic_recall_argmax'])} {d_ram:+8.1%} "
                f"{fmt_pct(m['toxic_recall_thresh'])} {d_rth:+8.1%} "
                f"{fmt_pct(m['toxic_to_safe_miss_rate'])} "
                f"{fmt_pct(m['toxic_below_thresh_miss_rate'])}")

    print(row("clean", clean))
    print("-" * len(header))
    # Rank by toxic-recall (thresholded) drop — the dangerous direction.
    ranked = sorted(
        ((name, results[name]) for name in PERTURBATIONS),
        key=lambda kv: kv[1]["toxic_recall_thresh"] - clean["toxic_recall_thresh"],
    )
    for name, m in ranked:
        print(row(name, m))

    # ─── Ranking summary (dangerous direction) ───────────────────────
    print("\n" + "=" * 78)
    print("RANK: perturbations that most hurt TOXIC-RECALL (thresholded) — the")
    print("      dangerous direction (a toxic plant that no longer trips an alert)")
    print("=" * 78)
    for i, (name, m) in enumerate(ranked, 1):
        d_rth = m["toxic_recall_thresh"] - clean["toxic_recall_thresh"]
        d_tts = m["toxic_to_safe_miss_rate"] - clean["toxic_to_safe_miss_rate"]
        print(f"  {i:2d}. {name:22s} toxic-recall {fmt_pct(m['toxic_recall_thresh'])} "
              f"(delta {d_rth:+.1%}); toxic->safe {fmt_pct(m['toxic_to_safe_miss_rate'])} "
              f"(delta {d_tts:+.1%})")

    # ─── JSON dump ───────────────────────────────────────────────────
    if args.json:
        payload = {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "threshold": args.threshold,
            "device": str(device),
            "per_class_counts": per_class,
            "total_images": len(samples),
            "class_names": CLASS_NAMES,
            "results": results,
            "ranking_by_thresholded_toxic_recall_drop": [
                {
                    "perturbation": name,
                    "toxic_recall_thresh": results[name]["toxic_recall_thresh"],
                    "delta_toxic_recall_thresh": results[name]["toxic_recall_thresh"]
                    - clean["toxic_recall_thresh"],
                    "toxic_to_safe_miss_rate": results[name]["toxic_to_safe_miss_rate"],
                    "delta_accuracy": results[name]["accuracy"] - clean["accuracy"],
                }
                for name, _ in ranked
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nFull results written to: {args.json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
