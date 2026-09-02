#!/usr/bin/env python3
"""Operating-point report: what the APP would do with a checkpoint on the frozen
held-out set, using the app's own per-class thresholds and uncertainty band.

evaluate_model.py sweeps a single uniform threshold. The app does not work that
way: it takes the top-1 class and its softmax probability, applies a per-class
alert threshold (ToxicityThresholds.baseAlert at the neutral sensitivity) and
surfaces a near-miss band (uncertaintyMargin) as "Possible … verify visually".
This script reproduces exactly that decision so the headline numbers in
ML_QUALITY.md are traceable to one command:

  confident toxic->safe miss   top-1 is safe_plants on a toxic image (the dangerous error)
  full-alert toxic recall      top-1 toxic AND p >= threshold[class]
  toxic surfaced               top-1 toxic AND p >= threshold[class] - margin  (alert or "verify")
  safe->toxic false alarm      top-1 toxic AND surfaced, on a safe image
  overall accuracy / per-class recall (argmax)

The thresholds are PARSED FROM THE SWIFT SOURCE (LeafAlert/Models/DetectionResult.swift)
so this report cannot drift from what ships. --blur K additionally reports the
same metrics after a horizontal motion blur of K pixels applied to the 224×224
parity-resized image (a walking-shake proxy; K=15 is the historical setting).

Usage:
  python scripts/operating_point.py --checkpoint checkpoints/student_v8_gbif.pth [--blur 15] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_v5 import load_v5, CLASS_LABELS, IMAGE_SIZE          # noqa: E402
from evaluate_model import (list_dataset, load_resized_rgb,       # noqa: E402
                            pil_to_torch_normalized, get_device, CANONICAL_CLASSES)

SWIFT_THRESHOLDS = ROOT / "LeafAlert" / "Models" / "DetectionResult.swift"
TOXIC = [c for c in CLASS_LABELS if c != "safe_plants"]


def parse_app_thresholds(path: Path = SWIFT_THRESHOLDS) -> dict:
    """Read baseAlert, defaultAlert, neutralSensitivity and uncertaintyMargin
    from the Swift source. Fails loudly if the shape of the file changes."""
    src = path.read_text()
    base = {m.group(1): float(m.group(2))
            for m in re.finditer(r'"(poison_\w+)":\s*([0-9.]+)', src)}
    default = float(re.search(r"defaultAlert:\s*Float\s*=\s*([0-9.]+)", src).group(1))
    neutral = float(re.search(r"neutralSensitivity:\s*Float\s*=\s*([0-9.]+)", src).group(1))
    margin = float(re.search(r"uncertaintyMargin:\s*Float\s*=\s*([0-9.]+)", src).group(1))
    assert set(base) == set(TOXIC), f"thresholds parsed for {sorted(base)}, expected {TOXIC}"
    return {"baseAlert": base, "defaultAlert": default, "neutralSensitivity": neutral,
            "uncertaintyMargin": margin}


def alert_threshold(cls: str, thr: dict, sensitivity: float | None = None) -> float:
    """Mirror of ToxicityThresholds.alertThreshold(for:sensitivity:)."""
    s = thr["neutralSensitivity"] if sensitivity is None else sensitivity
    base = thr["baseAlert"].get(cls, thr["defaultAlert"])
    return min(max(base + (s - thr["neutralSensitivity"]), 0.15), 0.95)


def motion_blur(img: Image.Image, k: int) -> Image.Image:
    """Horizontal box motion blur of k pixels on the parity-resized image."""
    a = np.asarray(img, dtype=np.float32)
    kernel = np.ones(k, dtype=np.float32) / k
    pad = k // 2
    padded = np.pad(a, ((0, 0), (pad, k - 1 - pad), (0, 0)), mode="edge")
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        out[..., c] = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded[..., c])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def run(model, samples, device, blur: int = 0, batch: int = 64) -> np.ndarray:
    probs = np.zeros((len(samples), len(CLASS_LABELS)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(samples), batch):
            imgs = []
            for path, _ in samples[s:s + batch]:
                img = load_resized_rgb(Path(path))
                if blur:
                    img = motion_blur(img, blur)
                imgs.append(pil_to_torch_normalized(img))
            x = torch.stack(imgs).to(device)
            probs[s:s + len(imgs)] = torch.softmax(model(x), dim=1).cpu().numpy()
    return probs


def decide(probs: np.ndarray, thr: dict, sensitivity: float | None = None):
    """Per-image app decision: (top1_index, severity) with severity in
    {'alert','uncertain','ignore','safe'}."""
    top = probs.argmax(1)
    out = []
    for i, t in enumerate(top):
        cls = CLASS_LABELS[t]
        if cls == "safe_plants":
            out.append((t, "safe")); continue
        p = float(probs[i, t]); a = alert_threshold(cls, thr, sensitivity)
        sev = "alert" if p >= a else ("uncertain" if p >= a - thr["uncertaintyMargin"] else "ignore")
        out.append((t, sev))
    return out


def metrics(y: np.ndarray, probs: np.ndarray, thr: dict, sensitivity: float | None = None) -> dict:
    dec = decide(probs, thr, sensitivity)
    top = np.array([d[0] for d in dec]); sev = np.array([d[1] for d in dec])
    safe_idx = CLASS_LABELS.index("safe_plants")
    is_toxic = y != safe_idx
    n_tox, n_safe = int(is_toxic.sum()), int((~is_toxic).sum())
    hard_miss = int(((top == safe_idx) & is_toxic).sum())
    full_alert = int(((sev == "alert") & is_toxic).sum())
    surfaced = int((np.isin(sev, ["alert", "uncertain"]) & is_toxic).sum())
    false_alarm = int((np.isin(sev, ["alert", "uncertain"]) & ~is_toxic).sum())
    false_full = int(((sev == "alert") & ~is_toxic).sum())
    per_class = {c: float((top[y == i] == i).mean()) for i, c in enumerate(CLASS_LABELS) if (y == i).any()}
    # Per-class dangerous error: toxic class c predicted safe.
    to_safe = {c: float((top[y == i] == safe_idx).mean()) for i, c in enumerate(CLASS_LABELS) if c != "safe_plants" and (y == i).any()}
    return {
        "n": int(len(y)), "n_toxic": n_tox, "n_safe": n_safe,
        "accuracy": float((top == y).mean()),
        "confident_toxic_to_safe_miss": hard_miss / n_tox,
        "confident_miss_count": hard_miss,
        "full_alert_toxic_recall": full_alert / n_tox,
        "toxic_surfaced": surfaced / n_tox,
        "safe_false_alarm_surfaced": false_alarm / n_safe,
        "safe_false_alarm_full": false_full / n_safe,
        "per_class_recall": per_class,
        "per_class_to_safe": to_safe,
    }


def render(name: str, m: dict) -> str:
    pc = " / ".join(f"{c.split('_')[-1]} {100 * r:.0f}" for c, r in m["per_class_recall"].items())
    return "\n".join([
        f"{name}: n={m['n']} (toxic {m['n_toxic']}, safe {m['n_safe']})",
        f"  confident toxic->safe miss   {100 * m['confident_toxic_to_safe_miss']:5.1f}%   ({m['confident_miss_count']} images)",
        f"  full-alert toxic recall      {100 * m['full_alert_toxic_recall']:5.1f}%",
        f"  toxic surfaced (alert+verify){100 * m['toxic_surfaced']:5.1f}%",
        f"  safe->toxic false alarm      {100 * m['safe_false_alarm_surfaced']:5.1f}%  (full alert {100 * m['safe_false_alarm_full']:.0f}%)",
        f"  overall accuracy             {100 * m['accuracy']:5.1f}%",
        f"  per-class recall (argmax)    {pc}",
        "  toxic->safe by class         " + " / ".join(f"{c.split('_')[-1]} {100 * r:.0f}%" for c, r in m["per_class_to_safe"].items()),
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "TrainingData" / "Testing")
    ap.add_argument("--blur", type=int, default=0, help="also report after a K-pixel horizontal motion blur")
    ap.add_argument("--sensitivity", type=float, default=None, help="override the slider (default: neutral)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    thr = parse_app_thresholds()
    print(f"App thresholds (from Swift): {thr['baseAlert']}  margin={thr['uncertaintyMargin']}  neutral={thr['neutralSensitivity']}")
    model = load_v5(a.checkpoint)
    device = get_device(); model.to(device)
    print(f"Checkpoint: {a.checkpoint.name}  backbone={model.backbone_name}  device={device}")
    samples = list_dataset(a.data)
    y = np.array([lab for _, lab in samples])
    assert list(CANONICAL_CLASSES) == list(CLASS_LABELS), "class order mismatch between evaluate_model and train_v5"

    result = {"checkpoint": str(a.checkpoint), "backbone": model.backbone_name, "thresholds": thr}
    probs = run(model, samples, device)
    result["clean"] = metrics(y, probs, thr, a.sensitivity)
    print(render("CLEAN", result["clean"]))
    if a.blur:
        bprobs = run(model, samples, device, blur=a.blur)
        result[f"blur{a.blur}"] = metrics(y, bprobs, thr, a.sensitivity)
        # The historical "blur flip" number: toxic images the clean model got
        # right (top-1 toxic) that blur pushes to top-1 safe.
        safe_idx = CLASS_LABELS.index("safe_plants")
        clean_ok = (probs.argmax(1) != safe_idx) & (y != safe_idx)
        flipped = clean_ok & (bprobs.argmax(1) == safe_idx)
        result[f"blur{a.blur}"]["toxic_to_safe_flip"] = float(flipped.sum() / max(1, clean_ok.sum()))
        print(render(f"BLUR k={a.blur}", result[f"blur{a.blur}"]))
        print(f"  blur toxic->safe flip        {100 * result[f'blur{a.blur}']['toxic_to_safe_flip']:5.1f}%  (of toxic images correct when clean)")
    if a.json:
        a.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"JSON -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
