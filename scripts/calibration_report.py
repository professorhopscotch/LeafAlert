#!/usr/bin/env python3
"""
Confidence calibration report for the LeafAlert PlantDetector student.

Answers: are the softmax probabilities well-calibrated? Over- or under-confident?
Is the default 0.65 alert threshold sensible for a *safety* app (where a toxic
plant predicted "safe" is the costly error)? Should we temperature-scale?

What it computes
----------------
  * Overall accuracy, toxic recall, and safe->toxic / toxic->safe confusion.
  * Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) on the
    top-1 confidence, plus a reliability table (per-bin accuracy vs confidence).
  * Per-class calibration (ECE computed on each class's predicted-as-that-class
    confidences).
  * A single-fit temperature T* (minimizing NLL on the logits) and the ECE it
    would yield — i.e. whether temperature scaling is warranted.
  * A threshold sweep for the "toxic alert" decision: for each threshold t, the
    fraction of toxic plants that would fire an alert (toxic recall / catch rate)
    and the false-alarm rate on safe plants (safe images whose max-toxic prob
    >= t). Makes the safety/nuisance tradeoff explicit around the default 0.65.

ALERT SEMANTICS (matches the app's intent): the app alerts when a *toxic* class
is predicted with confidence >= threshold. So the decision score for "should we
warn the user" is max over the three toxic classes of their probability. A
toxic plant is "caught" if that toxic score >= t; a safe plant is a "false
alarm" if that toxic score >= t. This is the safety-relevant framing and is what
the sweep reports (it does NOT require the toxic class to be the argmax — a plant
warned at 0.66 poison_oak is an alert even if safe_plants were 0.70; see
--alert-mode to switch to argmax-gated semantics).

Preprocessing parity: uses the SAME transform as on-device Vision (.scaleFill,
i.e. transforms.Resize((224,224)) squash, NO center crop) + ImageNet normalize,
imported from distill_model.test_transforms so numbers are comparable to device.

Usage
-----
    python3 scripts/calibration_report.py                      # clean held-out
    python3 scripts/calibration_report.py --data-dir TrainingData_split/test
    python3 scripts/calibration_report.py --bins 15 --json out.json
    python3 scripts/calibration_report.py --alert-mode argmax

Real numbers only. Everything printed is measured on the checkpoint + the images
in --data-dir. NOTHING is trained or modified.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

# Import the model + the on-device-parity eval transform so this stays in lockstep
# with the training/serving preprocessing contract.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_model import PlantDetectorNet, test_transforms  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "student_distilled.pth"
# Canonical ImageFolder-alphabetical class order.
CLASS_ORDER = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]
TOXIC_CLASSES = ["poison_ivy", "poison_oak", "poison_sumac"]
SAFE_CLASS = "safe_plants"


# ─── Inference ───────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint: Path, device, arch: str = "auto"):
    """Loads either the old distilled PlantDetectorNet or a v5-recipe checkpoint
    (any backbone). evaluate_model.build_torch_model auto-detects the
    architecture from the state_dict keys and, for v5, rebuilds the recorded
    backbone/head — this script used to hard-code PlantDetectorNet and could not
    report calibration for the shipped v5+ models at all."""
    from evaluate_model import build_torch_model
    model, resolved = build_torch_model(checkpoint, arch)
    print(f"  torch architecture: {resolved}")
    return model.eval().to(device)


def collect_logits(model, data_dir: Path, device, batch_size: int = 64):
    """Return (logits[N,C], labels[N], class_names) with parity preprocessing.

    class_names come from ImageFolder and are asserted to equal CLASS_ORDER so a
    silently-reordered dataset can't corrupt the toxic/safe split.
    """
    ds = datasets.ImageFolder(str(data_dir), transform=test_transforms)
    if ds.classes != CLASS_ORDER:
        raise SystemExit(
            f"Class order mismatch in {data_dir}:\n"
            f"  found:    {ds.classes}\n"
            f"  expected: {CLASS_ORDER}\n"
            "Toxic/safe labelling would be wrong; aborting."
        )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels), ds.classes


# ─── Calibration metrics ─────────────────────────────────────────────

def ece_mce(confidences: np.ndarray, correct: np.ndarray, n_bins: int):
    """Expected & Maximum Calibration Error on top-1 confidence.

    Returns (ece, mce, bins) where bins is a list of dicts per occupied bin with
    lo/hi/count/avg_conf/accuracy/gap for the reliability diagram.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(confidences)
    ece = 0.0
    mce = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on the right so conf==1.0 lands somewhere
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "avg_conf": None, "accuracy": None, "gap": None})
            continue
        avg_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(avg_conf - acc)
        ece += (cnt / N) * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "count": cnt,
                     "avg_conf": avg_conf, "accuracy": acc, "gap": gap})
    return ece, mce, bins


def per_class_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int):
    """Per-class ECE. For class k, take every sample PREDICTED as k, use its
    prob[k] as confidence and (true==k) as correctness. Empty classes -> None."""
    preds = probs.argmax(axis=1)
    out = {}
    for k, name in enumerate(CLASS_ORDER):
        mask = preds == k
        if mask.sum() == 0:
            out[name] = {"n_predicted": 0, "ece": None, "avg_conf": None,
                         "precision": None}
            continue
        conf = probs[mask, k]
        correct = (labels[mask] == k).astype(np.float64)
        ece, _, _ = ece_mce(conf, correct, n_bins)
        out[name] = {
            "n_predicted": int(mask.sum()),
            "ece": float(ece),
            "avg_conf": float(conf.mean()),
            "precision": float(correct.mean()),
        }
    return out


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor):
    """Fit a single scalar temperature T minimizing NLL of softmax(logits / T).
    Returns (T*, nll_before, nll_after). Optimized with LBFGS on log(T) to keep
    T > 0."""
    logits = logits.detach().double()
    labels = labels.detach().long()
    log_T = torch.zeros(1, dtype=torch.double, requires_grad=True)
    nll = torch.nn.functional.cross_entropy

    nll_before = float(nll(logits, labels).item())
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_T.exp().item())
    nll_after = float(nll(logits / T, labels).item())
    return T, nll_before, nll_after


# ─── Safety metrics ──────────────────────────────────────────────────

def toxic_alert_score(probs: np.ndarray):
    """Per-sample decision score for 'warn the user': max prob over toxic classes."""
    toxic_idx = [CLASS_ORDER.index(c) for c in TOXIC_CLASSES]
    return probs[:, toxic_idx].max(axis=1)


def threshold_sweep(probs, labels, thresholds, alert_mode: str):
    """For each threshold produce toxic catch-rate (recall) and safe false-alarm.

    alert_mode:
      'toxic_score' (default): alert iff max-toxic-prob >= t (argmax-agnostic).
      'argmax': alert iff argmax is a toxic class AND its prob >= t.
    """
    toxic_idx = [CLASS_ORDER.index(c) for c in TOXIC_CLASSES]
    safe_idx = CLASS_ORDER.index(SAFE_CLASS)
    preds = probs.argmax(axis=1)

    if alert_mode == "argmax":
        argmax_is_toxic = np.isin(preds, toxic_idx)
        argmax_toxic_prob = probs[np.arange(len(probs)), preds]
        score = np.where(argmax_is_toxic, argmax_toxic_prob, 0.0)
    else:
        score = toxic_alert_score(probs)

    is_toxic_gt = np.isin(labels, toxic_idx)
    is_safe_gt = labels == safe_idx
    n_toxic = int(is_toxic_gt.sum())
    n_safe = int(is_safe_gt.sum())

    rows = []
    for t in thresholds:
        alert = score >= t
        toxic_caught = int((alert & is_toxic_gt).sum())
        safe_alarm = int((alert & is_safe_gt).sum())
        rows.append({
            "threshold": float(t),
            "toxic_recall": toxic_caught / n_toxic if n_toxic else None,
            "toxic_caught": toxic_caught,
            "toxic_total": n_toxic,
            "toxic_missed": n_toxic - toxic_caught,
            "false_alarm_rate": safe_alarm / n_safe if n_safe else None,
            "false_alarms": safe_alarm,
            "safe_total": n_safe,
        })
    return rows


# ─── Reporting ───────────────────────────────────────────────────────

def confusion(probs, labels):
    preds = probs.argmax(axis=1)
    C = len(CLASS_ORDER)
    m = np.zeros((C, C), dtype=int)
    for t, p in zip(labels, preds):
        m[t, p] += 1
    return m


def print_report(logits, labels, class_names, n_bins, thresholds, alert_mode):
    probs = F.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (preds == labels_np).astype(np.float64)
    N = len(labels_np)

    toxic_idx = [CLASS_ORDER.index(c) for c in TOXIC_CLASSES]
    safe_idx = CLASS_ORDER.index(SAFE_CLASS)
    is_toxic = np.isin(labels_np, toxic_idx)

    print("=" * 72)
    print("LeafAlert PlantDetector — Confidence Calibration Report")
    print("=" * 72)
    print(f"Samples: {N}   Classes: {class_names}")
    print(f"Class counts: " + ", ".join(
        f"{c}={int((labels_np==i).sum())}" for i, c in enumerate(CLASS_ORDER)))
    print(f"Overall accuracy (argmax): {correct.mean():.4f}")
    print(f"Mean top-1 confidence:     {conf.mean():.4f}")
    print(f"Confidence − accuracy gap: {conf.mean()-correct.mean():+.4f}  "
          f"({'OVER-confident' if conf.mean()>correct.mean() else 'UNDER-confident'})")

    # Toxic recall (argmax any toxic class == true toxic)
    toxic_correct_argmax = int((is_toxic & np.isin(preds, toxic_idx)).sum())
    print(f"\nToxic argmax-recall (predicted some toxic class | truly toxic): "
          f"{toxic_correct_argmax}/{int(is_toxic.sum())} = "
          f"{toxic_correct_argmax/max(1,int(is_toxic.sum())):.4f}")
    # toxic -> safe misses (the safety-critical failure)
    toxic_to_safe = int((is_toxic & (preds == safe_idx)).sum())
    print(f"Toxic → safe_plants argmax misses (SAFETY-CRITICAL false negatives): "
          f"{toxic_to_safe}/{int(is_toxic.sum())} = "
          f"{toxic_to_safe/max(1,int(is_toxic.sum())):.4f}")

    # ── Overall calibration ──
    ece, mce, bins = ece_mce(conf, correct, n_bins)
    print("\n" + "-" * 72)
    print(f"OVERALL CALIBRATION (top-1 confidence, {n_bins} bins)")
    print("-" * 72)
    print(f"ECE = {ece:.4f}   MCE = {mce:.4f}")
    print("\nReliability table:")
    print(f"  {'bin':>13} {'n':>5} {'avg_conf':>9} {'accuracy':>9} {'gap':>7}")
    for b in bins:
        if b["count"] == 0:
            continue
        print(f"  [{b['lo']:.2f},{b['hi']:.2f}] {b['count']:5d} "
              f"{b['avg_conf']:9.4f} {b['accuracy']:9.4f} "
              f"{b['avg_conf']-b['accuracy']:+7.4f}")

    # ── Per-class calibration ──
    print("\n" + "-" * 72)
    print("PER-CLASS CALIBRATION (over samples predicted as each class)")
    print("-" * 72)
    pce = per_class_ece(probs, labels_np, n_bins)
    print(f"  {'class':>13} {'n_pred':>7} {'precision':>10} {'avg_conf':>9} {'ECE':>7}")
    for name in CLASS_ORDER:
        r = pce[name]
        if r["n_predicted"] == 0:
            print(f"  {name:>13} {0:7d} {'-':>10} {'-':>9} {'-':>7}")
            continue
        print(f"  {name:>13} {r['n_predicted']:7d} {r['precision']:10.4f} "
              f"{r['avg_conf']:9.4f} {r['ece']:7.4f}")

    # ── Temperature scaling ──
    print("\n" + "-" * 72)
    print("TEMPERATURE SCALING (single-fit T*, minimizing NLL)")
    print("-" * 72)
    T, nll_before, nll_after = fit_temperature(logits, labels)
    scaled_probs = F.softmax(logits / T, dim=1).numpy()
    scaled_conf = scaled_probs.max(axis=1)
    scaled_correct = (scaled_probs.argmax(1) == labels_np).astype(np.float64)
    ece_after, mce_after, _ = ece_mce(scaled_conf, scaled_correct, n_bins)
    direction = "temperatures >1 shrink confidence (model was OVER-confident)" \
        if T > 1 else "temperatures <1 sharpen confidence (model was UNDER-confident)"
    print(f"T* = {T:.4f}   ({direction})")
    print(f"NLL: {nll_before:.4f} -> {nll_after:.4f}   "
          f"(Δ {nll_after-nll_before:+.4f})")
    print(f"ECE: {ece:.4f} -> {ece_after:.4f}   (Δ {ece_after-ece:+.4f})")
    print(f"Note: argmax (and thus accuracy) is UNCHANGED by temperature scaling; "
          f"only probabilities/thresholds move.")

    # ── Confusion matrix ──
    print("\n" + "-" * 72)
    print("CONFUSION MATRIX (rows = true, cols = predicted)")
    print("-" * 72)
    m = confusion(probs, labels_np)
    hdr = "".join(f"{c[:9]:>11}" for c in CLASS_ORDER)
    print(f"  {'true\\pred':>13}{hdr}")
    for i, c in enumerate(CLASS_ORDER):
        print(f"  {c:>13}" + "".join(f"{m[i,j]:>11d}" for j in range(len(CLASS_ORDER))))

    # ── Threshold sweep ──
    print("\n" + "-" * 72)
    print(f"THRESHOLD SWEEP — toxic-alert decision  (alert_mode={alert_mode})")
    print("-" * 72)
    print("  score = max prob over {poison_ivy,poison_oak,poison_sumac}")
    print("  toxic_recall = fraction of TOXIC plants that fire an alert (want high)")
    print("  false_alarm  = fraction of SAFE plants that fire an alert (want low)")
    rows = threshold_sweep(probs, labels_np, thresholds, alert_mode)
    print(f"\n  {'thresh':>7} {'toxic_recall':>13} {'caught/tot':>12} "
          f"{'missed':>7} {'false_alarm':>12} {'FA/tot':>10}")
    for r in rows:
        tag = "  <-- default 0.65" if abs(r["threshold"] - 0.65) < 1e-9 else ""
        print(f"  {r['threshold']:7.2f} {r['toxic_recall']:13.4f} "
              f"{str(r['toxic_caught'])+'/'+str(r['toxic_total']):>12} "
              f"{r['toxic_missed']:7d} {r['false_alarm_rate']:12.4f} "
              f"{str(r['false_alarms'])+'/'+str(r['safe_total']):>10}{tag}")

    return {
        "n_samples": int(N),
        "class_counts": {c: int((labels_np == i).sum()) for i, c in enumerate(CLASS_ORDER)},
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "conf_minus_acc": float(conf.mean() - correct.mean()),
        "toxic_argmax_recall": toxic_correct_argmax / max(1, int(is_toxic.sum())),
        "toxic_to_safe_misses": toxic_to_safe,
        "toxic_total": int(is_toxic.sum()),
        "ece": float(ece),
        "mce": float(mce),
        "reliability_bins": bins,
        "per_class": pce,
        "temperature": {"T_star": T, "nll_before": nll_before,
                        "nll_after": nll_after, "ece_before": float(ece),
                        "ece_after": float(ece_after)},
        "confusion_matrix": m.tolist(),
        "threshold_sweep": rows,
        "alert_mode": alert_mode,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Confidence calibration report for LeafAlert PlantDetector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="Student checkpoint (.pth)")
    p.add_argument("--data-dir", type=Path,
                   default=PROJECT_ROOT / "TrainingData" / "Testing",
                   help="ImageFolder root to evaluate on")
    p.add_argument("--bins", type=int, default=10,
                   help="Number of confidence bins for ECE / reliability")
    p.add_argument("--thresholds", type=str,
                   default="0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
                   help="Comma-separated thresholds for the toxic-alert sweep")
    p.add_argument("--alert-mode", choices=["toxic_score", "argmax"],
                   default="toxic_score",
                   help="toxic_score: alert iff max-toxic-prob>=t; "
                        "argmax: alert iff argmax is toxic and its prob>=t")
    p.add_argument("--arch", choices=["auto", "distilled", "v5"], default="auto",
                   help="Checkpoint architecture; auto-detected from the keys.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--json", type=Path, default=None,
                   help="Optional path to write full metrics as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    device = get_device()
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir:   {args.data_dir}")
    model = load_model(args.checkpoint, device, args.arch)
    logits, labels, class_names = collect_logits(
        model, args.data_dir, device, args.batch_size)
    report = print_report(logits, labels, class_names, args.bins,
                          thresholds, args.alert_mode)
    report["meta"] = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "bins": args.bins,
        "device": str(device),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON written to: {args.json}")


if __name__ == "__main__":
    main()
