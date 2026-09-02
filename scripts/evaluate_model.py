#!/usr/bin/env python3
"""
LeafAlert PlantDetector — evaluation harness.

Loads BOTH the torch student and the shipped Core ML model, runs them over a
dataset with the CANONICAL PARITY preprocessing, and reports:

The torch checkpoint architecture is selected by --arch {auto,distilled,v5}
(default auto): 'distilled' = PlantDetectorNet (distill_model.py), 'v5' =
PlantDetectorV5 (train_v5.py). 'auto' inspects the state_dict keys (head.* -> v5,
classifier./spatial_attn.* -> distilled). The Core ML (.mlpackage) path is
architecture-agnostic and unaffected.

  * full 4x4 confusion matrix (per model)
  * per-class precision / recall / F1 / support
  * overall accuracy + macro-F1
  * SAFETY metrics:
      - toxic-recall     : fraction of actual toxic plants flagged toxic
      - toxic->safe miss : actual toxic predicted safe_plants (false negative)
      - safe->toxic alarm: actual safe predicted toxic (false positive)
    reported at a default 0.65 threshold AND as a threshold sweep
  * torch vs Core ML agreement (argmax agreement + mean/max prob delta)

PARITY CONTRACT (must match on-device Vision path exactly):
  resize the WHOLE image to 224x224 (transforms.Resize((224,224)) == Vision
  .scaleFill, NO center crop), then per-channel ImageNet normalize
  mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225].
  The Core ML model bakes normalize+softmax internally and consumes raw 0-255
  RGB, so we feed it the *resized* PIL image (no normalization) and feed torch
  the normalized tensor. Both see the identical .scaleFill 224x224 pixels.

CANONICAL CLASS ORDER (ImageFolder-alphabetical):
  ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]
  toxic = first three; safe_plants is the negative class.

The "threshold" for safety is applied to the TOXIC decision: a sample is
flagged toxic iff its top prediction is a toxic class AND that class's
probability >= threshold; otherwise it is treated as safe/abstain.

Usage:
  python3 scripts/evaluate_model.py \
      --checkpoint checkpoints/student_distilled.pth \
      --coreml LeafAlert/Resources/MLModels/PlantDetector.mlpackage \
      --data TrainingData/Testing \
      --split heldout

  # Evaluate the v5 torch checkpoint (auto-detected; --arch v5 to force):
  python3 scripts/evaluate_model.py \
      --checkpoint checkpoints/plant_detector_v5.pth \
      --data TrainingData/Testing --split heldout

  --split is a free-text LABEL echoed into the report so callers can annotate
  whether the --data path is train-set (optimistic) or a clean held-out set.
  Use --kfold N to additionally run a stratified k-fold linear-probe honest
  estimate on the torch feature space (does NOT retrain the shipped model).
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Import the student architecture + canonical constants from the trainer so we
# never drift from the shipped definition.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_model import PlantDetectorNet, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

CANONICAL_CLASSES = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]
TOXIC_CLASSES = ["poison_ivy", "poison_oak", "poison_sumac"]
SAFE_CLASS = "safe_plants"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ─── Data loading (canonical parity preprocessing) ──────────────────────

def list_dataset(data_dir: Path):
    """Return [(path, class_index)] for class subdirs matching CANONICAL_CLASSES.

    Only the four canonical class folders are read, in canonical order, so the
    label indexing is identical to ImageFolder-alphabetical regardless of what
    else lives under data_dir (e.g. a nested Testing/ folder)."""
    samples = []
    for idx, cls in enumerate(CANONICAL_CLASSES):
        cdir = data_dir / cls
        if not cdir.is_dir():
            continue
        for f in sorted(cdir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                samples.append((f, idx))
    return samples


def load_resized_rgb(path: Path) -> Image.Image:
    """Load an image and squash it to IMAGE_SIZE x IMAGE_SIZE (== Vision
    .scaleFill). This single resized PIL image is the shared input to BOTH
    models, guaranteeing identical pixels."""
    img = Image.open(path).convert("RGB")
    # BILINEAR matches torchvision.transforms.Resize default and Vision's
    # default scaling well enough for parity; the dominant parity risk is the
    # geometry (.scaleFill vs crop), which we handle by squashing.
    return img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def pil_to_torch_normalized(img: Image.Image) -> torch.Tensor:
    """Resized PIL -> normalized CHW float tensor for the torch student."""
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(chw)


# ─── Model runners ──────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── Architecture detection / construction ──────────────────────────────

def detect_arch(state: dict) -> str:
    """Infer the torch architecture from a state_dict's key namespace.

    PlantDetectorNet (distilled) keys live under backbone./spatial_attn./
    classifier.; PlantDetectorV5 keys live under backbone./head.. Both share a
    `backbone.` prefix, so the unambiguous discriminator is the presence of a
    `head.` namespace (v5) vs `classifier.`/`spatial_attn.` (distilled)."""
    keys = list(state.keys())
    has_head = any(k.startswith("head.") for k in keys)
    has_classifier = any(k.startswith("classifier.") for k in keys)
    has_attn = any(k.startswith("spatial_attn.") for k in keys)
    if has_head and not has_classifier:
        return "v5"
    if has_classifier or has_attn:
        return "distilled"
    raise ValueError(
        "Could not auto-detect architecture from checkpoint keys "
        f"(sample: {keys[:6]}). Pass --arch distilled|v5 explicitly."
    )


def build_torch_model(checkpoint: Path, arch: str = "auto"):
    """Load a torch checkpoint into the correct architecture.

    arch is 'auto' (inspect state_dict keys), 'distilled' (PlantDetectorNet from
    distill_model), or 'v5' (PlantDetectorV5 from train_v5). Returns
    (eval-mode model on CPU, resolved_arch str)."""
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    resolved = detect_arch(state) if arch == "auto" else arch
    if resolved == "distilled":
        model = PlantDetectorNet(len(CANONICAL_CLASSES))
    elif resolved == "v5":
        # Imported lazily: train_v5 -> coreml_export -> coremltools, and we want
        # the torch-only path (--skip-coreml) usable without coremltools.
        # load_v5 rebuilds the backbone/head the checkpoint was trained with
        # (JSON sidecar, else inferred from the keys) — v9 candidates are no
        # longer all efficientnet_b0.
        from train_v5 import load_v5
        return load_v5(checkpoint), resolved
    else:
        raise ValueError(f"unknown --arch '{arch}' (expected auto|distilled|v5)")
    model.load_state_dict(state)
    model.eval()
    return model, resolved


def run_torch(checkpoint: Path, samples, device, arch="auto", batch_size=64):
    """Return ((N, 4) softmax-probability array, resolved_arch) from the torch
    student.

    The shipped checkpoint outputs raw logits; we apply softmax here so the
    probabilities are comparable to the Core ML model (which bakes softmax)."""
    model, resolved = build_torch_model(checkpoint, arch)
    model.to(device)
    print(f"  torch architecture: {resolved}"
          + (" (auto-detected)" if arch == "auto" else ""))

    probs = np.zeros((len(samples), len(CANONICAL_CLASSES)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            batch = torch.stack([
                pil_to_torch_normalized(load_resized_rgb(p)) for p, _ in chunk
            ]).to(device)
            logits = model(batch)
            p = F.softmax(logits, dim=1).cpu().numpy()
            probs[start:start + len(chunk)] = p
    return probs, resolved


def run_coreml(coreml_path: Path, samples):
    """Return (N, 4) probability array from the Core ML classifier, columns in
    CANONICAL_CLASSES order (remapped from the model's dict output)."""
    import coremltools as ct
    model = ct.models.MLModel(str(coreml_path))
    spec = model.get_spec()
    in_name = spec.description.input[0].name
    prob_name = spec.description.predictedProbabilitiesName or "classLabel_probs"

    probs = np.zeros((len(samples), len(CANONICAL_CLASSES)), dtype=np.float32)
    col = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
    for i, (p, _) in enumerate(samples):
        img = load_resized_rgb(p)  # raw 0-255 RGB; model bakes normalize+softmax
        out = model.predict({in_name: img})
        d = out[prob_name]
        for cls, prob in d.items():
            if cls in col:
                probs[i, col[cls]] = prob
    return probs


# ─── Metrics ────────────────────────────────────────────────────────────

def confusion_matrix(y_true, y_pred, n=4):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm):
    """Return dict class_idx -> (precision, recall, f1, support)."""
    out = {}
    n = cm.shape[0]
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[i] = (prec, rec, f1, int(support))
    return out


def safety_metrics(y_true, probs, threshold):
    """Compute toxic-recall, toxic->safe miss-rate, safe->toxic false-alarm.

    Decision rule (matches on-device threshold semantics): a sample is FLAGGED
    TOXIC iff argmax is a toxic class AND that class prob >= threshold. Anything
    else (argmax safe, or a toxic argmax below threshold) is NOT flagged toxic.

    - toxic_recall     = flagged_toxic / actual_toxic
    - toxic_to_safe    = (actual toxic, predicted argmax == safe_plants) / actual_toxic
                         (a hard misclassification into the safe class)
    - toxic_below_thr  = (actual toxic, argmax toxic but < threshold) / actual_toxic
                         (would not alert -> also a dangerous miss operationally)
    - toxic_miss_total = actual toxic NOT flagged toxic  / actual_toxic
                         (= 1 - toxic_recall; the full false-negative rate)
    - safe_false_alarm = (actual safe, flagged toxic)    / actual_safe
    """
    safe_idx = CANONICAL_CLASSES.index(SAFE_CLASS)
    toxic_idx = {CANONICAL_CLASSES.index(c) for c in TOXIC_CLASSES}

    argmax = probs.argmax(axis=1)
    top_prob = probs.max(axis=1)

    actual_toxic = actual_safe = 0
    flagged_toxic_and_toxic = 0
    toxic_pred_safe = 0
    toxic_below_thr = 0
    safe_flagged_toxic = 0

    for t, am, tp in zip(y_true, argmax, top_prob):
        flagged_toxic = (am in toxic_idx) and (tp >= threshold)
        if t in toxic_idx:
            actual_toxic += 1
            if flagged_toxic:
                flagged_toxic_and_toxic += 1
            if am == safe_idx:
                toxic_pred_safe += 1
            elif (am in toxic_idx) and tp < threshold:
                toxic_below_thr += 1
        else:  # actual safe
            actual_safe += 1
            if flagged_toxic:
                safe_flagged_toxic += 1

    def frac(a, b):
        return a / b if b else float("nan")

    return {
        "threshold": threshold,
        "actual_toxic": actual_toxic,
        "actual_safe": actual_safe,
        "toxic_recall": frac(flagged_toxic_and_toxic, actual_toxic),
        "toxic_miss_total": frac(actual_toxic - flagged_toxic_and_toxic, actual_toxic),
        "toxic_to_safe_hard": frac(toxic_pred_safe, actual_toxic),
        "toxic_below_thr": frac(toxic_below_thr, actual_toxic),
        "safe_false_alarm": frac(safe_flagged_toxic, actual_safe),
    }


def format_report(name, y_true, probs, split_label, thresholds):
    argmax = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, argmax)
    prf = per_class_prf(cm)
    acc = np.mean(argmax == np.array(y_true))
    macro_f1 = np.mean([prf[i][2] for i in range(len(CANONICAL_CLASSES))])

    lines = []
    lines.append(f"\n{'=' * 68}")
    lines.append(f"MODEL: {name}   |   split: {split_label}   |   N={len(y_true)}")
    lines.append(f"{'=' * 68}")

    lines.append("\nConfusion matrix (rows=true, cols=pred):")
    hdr = "true\\pred".ljust(14) + "".join(c[:11].rjust(13) for c in CANONICAL_CLASSES)
    lines.append(hdr)
    for i, c in enumerate(CANONICAL_CLASSES):
        row = c[:13].ljust(14) + "".join(str(cm[i, j]).rjust(13) for j in range(4))
        lines.append(row)

    lines.append("\nPer-class precision / recall / F1 / support:")
    lines.append("  class".ljust(20) + "prec".rjust(9) + "recall".rjust(9)
                 + "f1".rjust(9) + "support".rjust(10))
    for i, c in enumerate(CANONICAL_CLASSES):
        p, r, f, s = prf[i]
        lines.append(f"  {c:16s}" + f"{p:9.3f}{r:9.3f}{f:9.3f}{s:10d}")

    lines.append(f"\nOverall accuracy: {acc:.4f}   Macro-F1: {macro_f1:.4f}")

    lines.append("\nSafety metrics (threshold applied to toxic decision):")
    lines.append("  thr   toxic-recall  miss-total  ->safe(hard)  <thr(abstain)  safe->toxic alarm")
    for thr in thresholds:
        s = safety_metrics(y_true, probs, thr)
        lines.append(
            f"  {thr:.2f}"
            f"   {s['toxic_recall']:11.4f}"
            f"  {s['toxic_miss_total']:10.4f}"
            f"  {s['toxic_to_safe_hard']:12.4f}"
            f"  {s['toxic_below_thr']:13.4f}"
            f"  {s['safe_false_alarm']:16.4f}"
        )
    return "\n".join(lines), {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "per_class": {CANONICAL_CLASSES[i]: prf[i] for i in prf},
    }


def agreement_report(pt_probs, cm_probs):
    pt_arg = pt_probs.argmax(axis=1)
    cm_arg = cm_probs.argmax(axis=1)
    agree = np.mean(pt_arg == cm_arg)
    mean_abs = np.mean(np.abs(pt_probs - cm_probs))
    max_abs = np.max(np.abs(pt_probs - cm_probs))
    # top-1 prob correlation
    lines = [
        f"\n{'=' * 68}",
        "TORCH vs CORE ML AGREEMENT",
        f"{'=' * 68}",
        f"  argmax agreement:        {agree:.4f} ({int(agree*len(pt_arg))}/{len(pt_arg)})",
        f"  mean |prob delta|:       {mean_abs:.5f}",
        f"  max  |prob delta|:       {max_abs:.5f}",
    ]
    # show worst disagreements
    disagree = np.where(pt_arg != cm_arg)[0]
    if len(disagree):
        lines.append(f"  {len(disagree)} argmax disagreements (idx: torch->coreml):")
        for idx in disagree[:10]:
            lines.append(
                f"    #{idx}: torch={CANONICAL_CLASSES[pt_arg[idx]]}"
                f"({pt_probs[idx].max():.3f}) "
                f"coreml={CANONICAL_CLASSES[cm_arg[idx]]}({cm_probs[idx].max():.3f})"
            )
    return "\n".join(lines), {
        "argmax_agreement": float(agree),
        "mean_abs_prob_delta": float(mean_abs),
        "max_abs_prob_delta": float(max_abs),
    }


# ─── K-fold honest estimate (linear probe on frozen features) ───────────

def _frozen_features(model, arch: str, batch: torch.Tensor) -> np.ndarray:
    """Frozen penultimate feature for the linear probe, per architecture.

    distilled: reproduce the forward up to the concatenated pooled vector
      (2560-d: spatial-attention + avg/max dual-pool) that feeds the classifier.
    v5: the 1280-d globally-pooled EfficientNet-B0 feature that feeds the head
      (backbone was built with num_classes=0, so it already returns the pooled
      vector). Head-choice-agnostic (linear vs bottleneck)."""
    if arch == "v5":
        return model.backbone(batch).cpu().numpy()
    f = model.backbone(batch)
    f = model.spatial_attn(f)
    avg = model.avg_pool(f).flatten(1)
    mx = model.max_pool(f).flatten(1)
    return torch.cat([avg, mx], dim=1).cpu().numpy()


def kfold_linear_probe(checkpoint, samples, device, arch="auto", k=5, seed=42):
    """Honest generalization estimate that does NOT reuse the shipped model's
    unknown train/val split and does NOT retrain the shipped weights.

    Extract frozen penultimate features from the student backbone (identical
    for train-membership since the backbone is frozen at eval), then run
    stratified k-fold logistic regression on those features. This measures how
    linearly separable the student's learned representation is on a proper
    held-out fold — a conservative, honest signal that is independent of which
    images the shipped classifier head happened to memorize.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        return "\n[kfold] scikit-learn not available — skipped.", None

    model, arch = build_torch_model(checkpoint, arch)
    model.to(device)

    # Frozen penultimate features (dim depends on arch: distilled=2560, v5=1280).
    ys = np.array([lbl for _, lbl in samples])
    feat_chunks = []
    with torch.no_grad():
        bs = 64
        for start in range(0, len(samples), bs):
            chunk = samples[start:start + bs]
            batch = torch.stack([
                pil_to_torch_normalized(load_resized_rgb(p)) for p, _ in chunk
            ]).to(device)
            feat_chunks.append(_frozen_features(model, arch, batch))
    feats = np.concatenate(feat_chunks, axis=0)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_acc = []
    toxic_idx = {CANONICAL_CLASSES.index(c) for c in TOXIC_CLASSES}
    fold_toxic_recall = []
    for tr, te in skf.split(feats, ys):
        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
        clf.fit(feats[tr], ys[tr])
        pred = clf.predict(feats[te])
        fold_acc.append(np.mean(pred == ys[te]))
        # toxic recall on this fold (argmax rule, no threshold)
        at = sum(1 for y in ys[te] if y in toxic_idx)
        hit = sum(1 for y, pr in zip(ys[te], pred)
                  if y in toxic_idx and pr in toxic_idx)
        fold_toxic_recall.append(hit / at if at else float("nan"))

    acc = np.array(fold_acc)
    tr_rec = np.array(fold_toxic_recall)
    lines = [
        f"\n{'=' * 68}",
        f"HONEST ESTIMATE — {k}-fold linear probe on frozen student features",
        f"{'=' * 68}",
        f"  (logistic regression on the {feats.shape[1]}-d pooled features; measures",
        "   representation quality on proper held-out folds, independent of",
        "   which images the shipped head memorized. NOT the shipped head.)",
        f"  accuracy per fold:   {['%.3f' % a for a in acc]}",
        f"  accuracy:            {acc.mean():.4f} +/- {acc.std():.4f}",
        f"  toxic-recall/fold:   {['%.3f' % t for t in tr_rec]}",
        f"  toxic-recall:        {tr_rec.mean():.4f} +/- {tr_rec.std():.4f}",
    ]
    return "\n".join(lines), {
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "toxic_recall_mean": float(tr_rec.mean()),
        "toxic_recall_std": float(tr_rec.std()),
    }


# ─── Main ───────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate the LeafAlert student + Core ML model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--checkpoint", type=str,
                    default=str(root / "checkpoints" / "student_distilled.pth"))
    ap.add_argument("--arch", choices=["auto", "distilled", "v5"], default="auto",
                    help="Torch checkpoint architecture. 'auto' inspects the "
                         "state_dict keys (head.* -> v5, classifier./spatial_attn. "
                         "-> distilled); 'distilled' = PlantDetectorNet, 'v5' = "
                         "PlantDetectorV5. The Core ML path is arch-agnostic.")
    ap.add_argument("--coreml", type=str,
                    default=str(root / "LeafAlert" / "Resources" / "MLModels"
                                / "PlantDetector.mlpackage"))
    ap.add_argument("--data", type=str, default=None,
                    help="Dataset root containing the 4 canonical class subdirs. "
                         "If omitted, chosen from --split: 'all'->TrainingData, "
                         "'held-out'->TrainingData/Testing.")
    ap.add_argument("--split", type=str, default="held-out",
                    help="'all' (every image under TrainingData; TRAIN-SET / "
                         "OPTIMISTIC), 'held-out' (TrainingData/Testing; zero "
                         "content-hash overlap with train — honest estimate), or "
                         "any free-text label when --data is given explicitly.")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="Default safety threshold for the toxic decision")
    ap.add_argument("--sweep", type=str, default="0.3,0.4,0.5,0.6,0.65,0.7,0.8,0.9",
                    help="Comma-separated thresholds for the safety sweep")
    ap.add_argument("--kfold", type=int, default=0,
                    help="If >0, also run a k-fold linear-probe honest estimate")
    ap.add_argument("--skip-coreml", action="store_true",
                    help="Skip the Core ML pass (torch only)")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, evaluate only the first N samples per class "
                         "(debug/smoke)")
    ap.add_argument("--json", type=str, default=None,
                    help="If set, write the full results as JSON to this path")
    return ap.parse_args()


def _resolve_data_dir(args, root: Path) -> Path:
    """Pick the data dir from --data, else from the --split shortcut."""
    if args.data is not None:
        return Path(args.data)
    if args.split == "held-out":
        return root / "TrainingData" / "Testing"
    if args.split == "all":
        return root / "TrainingData"
    # Unknown split with no --data: fall back to held-out.
    return root / "TrainingData" / "Testing"


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    data_dir = _resolve_data_dir(args, root)
    optimistic = (args.split == "all"
                  and data_dir.resolve() == (root / "TrainingData").resolve())
    samples = list_dataset(data_dir)
    if not samples:
        print(f"ERROR: no images found under {data_dir} in canonical class dirs "
              f"{CANONICAL_CLASSES}")
        sys.exit(1)

    if args.limit > 0:
        per_class = defaultdict(list)
        for s in samples:
            per_class[s[1]].append(s)
        samples = []
        for idx in sorted(per_class):
            samples.extend(per_class[idx][:args.limit])

    counts = Counter(lbl for _, lbl in samples)
    print(f"Data dir: {data_dir}")
    print(f"Split label: {args.split}")
    if optimistic:
        print("*** TRAIN-SET / OPTIMISTIC: the shipped checkpoint trained on")
        print("*** these images. Treat every number below as an UPPER BOUND,")
        print("*** NOT a generalization estimate.")
    elif args.split == "held-out":
        print("*** HELD-OUT: *_test_* images, zero content-hash overlap with the")
        print("*** training split — an honest generalization estimate.")
    print("Class counts: " + ", ".join(
        f"{CANONICAL_CLASSES[i]}={counts.get(i, 0)}" for i in range(4)))
    print(f"Total: {len(samples)} images")

    y_true = [lbl for _, lbl in samples]
    thresholds = [float(x) for x in args.sweep.split(",")]
    if args.threshold not in thresholds:
        thresholds = sorted(set(thresholds + [args.threshold]))

    device = get_device()
    print(f"Torch device: {device}")

    results = {
        "data_dir": str(data_dir),
        "split": args.split,
        "optimistic_train_set": bool(optimistic),
        "class_names": CANONICAL_CLASSES,
        "counts": {CANONICAL_CLASSES[i]: counts.get(i, 0) for i in range(4)},
        "n": len(samples),
        "threshold": args.threshold,
        "sweep": thresholds,
        "models": {},
    }

    def _pack(y_true, probs):
        argmax = probs.argmax(axis=1)
        cm = confusion_matrix(y_true, argmax)
        prf = per_class_prf(cm)
        return {
            "accuracy": float(np.mean(argmax == np.array(y_true))),
            "macro_f1": float(np.mean([prf[i][2] for i in range(4)])),
            "confusion_matrix": cm.tolist(),
            "per_class": {CANONICAL_CLASSES[i]: {
                "precision": prf[i][0], "recall": prf[i][1],
                "f1": prf[i][2], "support": prf[i][3]} for i in prf},
            "safety_at_threshold": safety_metrics(y_true, probs, args.threshold),
            "safety_sweep": [safety_metrics(y_true, probs, t) for t in thresholds],
        }

    # Torch pass
    print("\nRunning torch student...")
    pt_probs, torch_arch = run_torch(Path(args.checkpoint), samples, device,
                                     arch=args.arch)
    rep, _ = format_report(f"torch student [{torch_arch}] (softmaxed logits)",
                           y_true, pt_probs, args.split, thresholds)
    print(rep)
    results["models"]["torch"] = {"arch": torch_arch, **_pack(y_true, pt_probs)}

    # Core ML pass
    cm_probs = None
    if not args.skip_coreml:
        print("\nRunning Core ML model...")
        cm_probs = run_coreml(Path(args.coreml), samples)
        rep, _ = format_report("Core ML (baked normalize+softmax)", y_true,
                               cm_probs, args.split, thresholds)
        print(rep)
        results["models"]["coreml"] = _pack(y_true, cm_probs)

        rep, agr = agreement_report(pt_probs, cm_probs)
        print(rep)
        results["agreement"] = agr

    # K-fold honest estimate
    if args.kfold > 0:
        rep, kf = kfold_linear_probe(Path(args.checkpoint), samples, device,
                                     arch=torch_arch, k=args.kfold)
        print(rep)
        if kf is not None:
            results["kfold_linear_probe"] = kf

    if args.json:
        jp = Path(args.json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        with open(jp, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote JSON results to: {jp}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
