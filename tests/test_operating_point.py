"""operating_point.py must mirror ToxicityThresholds exactly, and read the
thresholds from the Swift source rather than a copy that can drift."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import operating_point as op  # noqa: E402
from train_v5 import CLASS_LABELS  # noqa: E402

IVY, OAK, SUMAC, SAFE = (CLASS_LABELS.index(c) for c in ("poison_ivy", "poison_oak", "poison_sumac", "safe_plants"))


def test_thresholds_are_parsed_from_the_swift_source():
    thr = op.parse_app_thresholds()
    assert set(thr["baseAlert"]) == {"poison_ivy", "poison_oak", "poison_sumac"}
    assert 0.15 <= min(thr["baseAlert"].values()) and max(thr["baseAlert"].values()) <= 0.95
    assert 0 < thr["uncertaintyMargin"] < 0.5
    assert thr["neutralSensitivity"] == 0.5


def _probs(rows):
    p = np.zeros((len(rows), len(CLASS_LABELS)), dtype=np.float32)
    for i, (idx, val) in enumerate(rows):
        p[i, idx] = val
        p[i, SAFE if idx != SAFE else IVY] = 1 - val   # runner-up gets the rest
    return p


def test_decision_mirrors_toxicity_thresholds_severity():
    thr = {"baseAlert": {"poison_ivy": 0.40, "poison_oak": 0.40, "poison_sumac": 0.52},
           "defaultAlert": 0.5, "neutralSensitivity": 0.5, "uncertaintyMargin": 0.20}
    probs = _probs([(IVY, 0.45), (IVY, 0.25), (IVY, 0.55), (SUMAC, 0.45), (SUMAC, 0.30), (SAFE, 0.9)])
    # Note: 0.55 on ivy leaves 0.45 for safe, so ivy is still top-1.
    sev = [s for _, s in op.decide(probs, thr)]
    assert sev == ["alert", "uncertain", "alert", "uncertain", "ignore", "safe"]


def test_sensitivity_shifts_every_threshold_like_the_app():
    thr = op.parse_app_thresholds()
    base = op.alert_threshold("poison_ivy", thr)
    assert abs(op.alert_threshold("poison_ivy", thr, sensitivity=0.6) - (base + 0.1)) < 1e-9
    assert op.alert_threshold("poison_ivy", thr, sensitivity=0.0) >= 0.15   # clamped


def test_metrics_count_the_dangerous_error_by_argmax_only():
    thr = {"baseAlert": {"poison_ivy": 0.40, "poison_oak": 0.40, "poison_sumac": 0.52},
           "defaultAlert": 0.5, "neutralSensitivity": 0.5, "uncertaintyMargin": 0.20}
    y = np.array([IVY, IVY, OAK, SUMAC, SAFE, SAFE])
    probs = _probs([(IVY, 0.45),   # alert
                    (SAFE, 0.6),   # toxic image, top-1 safe -> the dangerous miss
                    (OAK, 0.25),   # uncertain (surfaced, not full alert)
                    (SUMAC, 0.30), # ignore: toxic top-1 but below the band -> not surfaced, not a "safe" miss
                    (IVY, 0.45),   # safe image -> false alarm (surfaced + full)
                    (SAFE, 0.8)])  # correct safe
    m = op.metrics(y, probs, thr)
    assert m["confident_miss_count"] == 1 and abs(m["confident_toxic_to_safe_miss"] - 1 / 4) < 1e-9
    assert abs(m["full_alert_toxic_recall"] - 1 / 4) < 1e-9
    assert abs(m["toxic_surfaced"] - 2 / 4) < 1e-9
    assert abs(m["safe_false_alarm_surfaced"] - 1 / 2) < 1e-9
    assert abs(m["accuracy"] - 4 / 6) < 1e-9


def test_motion_blur_preserves_shape_and_smooths():
    from PIL import Image
    a = np.zeros((224, 224, 3), dtype=np.uint8); a[:, 100:104, :] = 255
    b = np.asarray(op.motion_blur(Image.fromarray(a), 15))
    assert b.shape == a.shape and b.max() < 255 and (b[:, 95:110, 0] > 0).all()
