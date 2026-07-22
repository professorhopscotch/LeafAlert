#!/usr/bin/env python3
"""
Out-of-distribution (OOD) report — "is this even a plant we know?"

The classifier is a closed-set 4-way softmax: shown a bird, a rock or a hand it
MUST still pick one of {poison_ivy, poison_oak, poison_sumac, safe_plants}, and
it can do so confidently. For a safety app, confidently labelling a non-plant as
a toxic plant is a false alarm that erodes trust; labelling it "safe" is worse.

This measures two things on real data:
  1. HARM TODAY — what fraction of non-plant images currently produce a toxic
     alert under the shipped per-class thresholds.
  2. SEPARABILITY — whether an OOD score can tell in-distribution (real plant
     photos) from out-of-distribution (non-plant) images, and where to set it.

Scores compared:
  * energy   E(x) = -logsumexp(logits)  (Liu et al. 2020) — lower = more
    in-distribution. Needs LOGITS, which softmax probabilities cannot recover
    (softmax is shift-invariant, so log(Z) is unrecoverable from p alone).
  * msp      max softmax probability (Hendrycks & Gimpel) — the weaker classic
    baseline, computable from probabilities alone.

Usage:
    python3 scripts/ood_report.py --ood-dir data_staging/ood
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v5 import PlantDetectorV5, CLASS_LABELS          # noqa: E402
from coreml_export import IMAGENET_MEAN, IMAGENET_STD       # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
STD = np.array(IMAGENET_STD, dtype=np.float32)
TOXIC = CLASS_LABELS[:3]

# Must mirror LeafAlert/Models/DetectionResult.swift → ToxicityThresholds.
BASE_ALERT = {"poison_ivy": 0.40, "poison_oak": 0.40, "poison_sumac": 0.52}
UNCERTAINTY_MARGIN = 0.20


def load_model(ckpt: Path) -> PlantDetectorV5:
    m = PlantDetectorV5(num_classes=len(CLASS_LABELS), head="linear", pretrained=False)
    m.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
    m.eval()
    return m


def logits_for(model, path: str, image_size: int = 224) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    a = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    t = torch.from_numpy(a.transpose(2, 0, 1)[None]).float()
    with torch.no_grad():
        return model(t)[0].numpy()


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def energy(z: np.ndarray) -> float:
    """E(x) = -logsumexp(logits). Lower = more in-distribution."""
    return float(-(np.log(np.exp(z - z.max()).sum()) + z.max()))


def severity(probs: np.ndarray) -> str:
    """Mirror the app's alert decision so we can count real-world harm."""
    tox = {c: float(probs[CLASS_LABELS.index(c)]) for c in TOXIC}
    top = max(tox, key=tox.get)
    if probs[CLASS_LABELS.index(top)] < probs[CLASS_LABELS.index("safe_plants")]:
        return "ignore"
    thr = BASE_ALERT[top]
    if tox[top] >= thr:
        return "alert"
    if tox[top] >= thr - UNCERTAINTY_MARGIN:
        return "uncertain"
    return "ignore"


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """P(random OOD scores higher than random ID). Rank-based, ties handled."""
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gather(model, files):
    out = []
    for f in files:
        try:
            z = logits_for(model, f)
        except Exception:
            continue
        p = softmax(z)
        out.append({"file": f, "energy": energy(z), "msp": float(p.max()),
                    "sev": severity(p), "pred": CLASS_LABELS[int(p.argmax())]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "student_v5_full.pth"))
    ap.add_argument("--id-dir", default=str(PROJECT_ROOT / "TrainingData" / "Testing"),
                    help="In-distribution images (frozen held-out set).")
    ap.add_argument("--ood-dir", default=str(PROJECT_ROOT / "data_staging" / "ood"),
                    help="Out-of-distribution (non-plant) images.")
    ap.add_argument("--id-retention", type=float, default=0.95,
                    help="Fraction of real plant photos the gate must keep.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    model = load_model(Path(args.checkpoint))

    id_files = [f for c in CLASS_LABELS
                for f in sorted(glob.glob(f"{args.id_dir}/{c}/*.jp*g"))]
    ood_files = sorted(glob.glob(f"{args.ood_dir}/**/*.jp*g", recursive=True)) + \
        sorted(glob.glob(f"{args.ood_dir}/**/*.png", recursive=True))
    if not ood_files:
        raise SystemExit(f"No OOD images under {args.ood_dir}")

    idr = gather(model, id_files)
    ood = gather(model, ood_files)
    print(f"in-distribution: {len(idr)} images | out-of-distribution: {len(ood)} images\n")

    # 1. Harm today
    n_alert = sum(1 for r in ood if r["sev"] == "alert")
    n_any = sum(1 for r in ood if r["sev"] in ("alert", "uncertain"))
    print("=== HARM TODAY (no OOD gate) ===")
    print(f"  non-plant images producing a FULL TOXIC ALERT : {n_alert}/{len(ood)} = {n_alert/len(ood):.1%}")
    print(f"  non-plant images surfaced as toxic at all     : {n_any}/{len(ood)} = {n_any/len(ood):.1%}")
    preds = {}
    for r in ood:
        preds[r["pred"]] = preds.get(r["pred"], 0) + 1
    print(f"  predicted-class distribution on non-plants    : {preds}\n")

    # 2. Separability
    e_id = np.array([r["energy"] for r in idr])
    e_ood = np.array([r["energy"] for r in ood])
    m_id = np.array([-r["msp"] for r in idr])     # negate: higher = more OOD
    m_ood = np.array([-r["msp"] for r in ood])
    print("=== SEPARABILITY (higher AUROC = better OOD detector) ===")
    print(f"  energy AUROC : {auroc(e_id, e_ood):.3f}")
    print(f"  MSP    AUROC : {auroc(m_id, m_ood):.3f}")
    print(f"  energy  ID  mean {e_id.mean():+.2f}  (p5 {np.percentile(e_id,5):+.2f} / p95 {np.percentile(e_id,95):+.2f})")
    print(f"  energy  OOD mean {e_ood.mean():+.2f}  (p5 {np.percentile(e_ood,5):+.2f} / p95 {np.percentile(e_ood,95):+.2f})\n")

    # 3. Operating point: keep --id-retention of real plants, reject what we can
    thr = float(np.percentile(e_id, 100 * args.id_retention))
    rejected_ood = float((e_ood > thr).mean())
    lost_id = float((e_id > thr).mean())
    # How much of the ACTUAL harm does it remove?
    harm_blocked = sum(1 for r in ood if r["sev"] in ("alert", "uncertain") and r["energy"] > thr)
    print(f"=== RECOMMENDED GATE (keep {args.id_retention:.0%} of real plants) ===")
    print(f"  energy threshold      : {thr:+.3f}   (flag as 'not a recognized plant' when energy > this)")
    print(f"  OOD correctly rejected: {rejected_ood:.1%}")
    print(f"  real plants lost      : {lost_id:.1%}")
    print(f"  false toxic surfacings blocked: {harm_blocked}/{n_any}"
          + (f" = {harm_blocked/n_any:.1%}" if n_any else ""))

    if args.json:
        Path(args.json).write_text(json.dumps({
            "threshold": thr, "energy_auroc": auroc(e_id, e_ood),
            "msp_auroc": auroc(m_id, m_ood),
            "ood_rejected": rejected_ood, "id_lost": lost_id,
            "ood_alert_rate": n_alert / len(ood), "ood_surfaced_rate": n_any / len(ood),
        }, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
