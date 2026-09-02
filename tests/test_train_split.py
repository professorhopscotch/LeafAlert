"""Regression tests for train_v5's source-disjoint train/val split.

These pin the bug the pilot run exposed: the greedy group splitter starved a
whole class (poison_ivy -> 0 train samples) because the original data pool shares
ONE filename token per class (one giant un-splittable group), which then produced
runaway inverse-frequency class weights ([3.90, 0.08, 0.02, 0.01]) and collapsed
the model onto the starved class.

Run: .venv/bin/python -m pytest tests/test_train_split.py
"""

import pytest
pytest.importorskip("torch")  # heavy ML dep: skipped in the light CI job
pytest.importorskip("timm")  # heavy ML dep: skipped in the light CI job

import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_v5 import (  # noqa: E402
    source_disjoint_split,
    compute_class_weights,
    CLASS_LABELS,
)


def _pool_like_records():
    """Mimic the real pool: each class is dominated by ONE coarse filename token
    (hundreds of distinct plants) plus a few small per-observation groups."""
    recs = []
    # counts roughly matching the real 4-class pool
    big = {0: 300, 1: 300, 2: 400, 3: 400}
    for label, n in big.items():
        cls = CLASS_LABELS[label]
        # one giant single-token group (the original-pool pathology)
        for i in range(n):
            recs.append({"path": f"/x/{cls}/{cls}_{i:04d}.jpg", "label": label, "group": cls})
        # a handful of small, correctly-grouped "observation" clusters
        for obs in range(5):
            for j in range(3):
                recs.append({
                    "path": f"/x/{cls}/obs_{cls}_{obs}_{j}.jpg",
                    "label": label, "group": f"obs_{cls}_{obs}",
                })
    return recs


def test_split_never_starves_a_class():
    tr, va = source_disjoint_split(_pool_like_records(), val_frac=0.2, seed=42)
    tc = Counter(r["label"] for r in tr)
    for i in range(len(CLASS_LABELS)):
        assert tc.get(i, 0) > 0, f"class {CLASS_LABELS[i]} was starved from train"


def test_split_respects_val_fraction():
    recs = _pool_like_records()
    tr, va = source_disjoint_split(recs, val_frac=0.2, seed=42)
    frac = len(va) / (len(tr) + len(va))
    # The old bug drove this to ~0.41; a correct split lands near 0.2.
    assert 0.12 <= frac <= 0.30, f"val fraction {frac:.2f} is far from target 0.2"


def test_split_is_source_disjoint():
    tr, va = source_disjoint_split(_pool_like_records(), val_frac=0.2, seed=42)
    assert not (set(r["group"] for r in tr) & set(r["group"] for r in va)), \
        "a source group leaked across the train/val boundary"


def test_split_is_deterministic():
    a = source_disjoint_split(_pool_like_records(), val_frac=0.2, seed=42)
    b = source_disjoint_split(_pool_like_records(), val_frac=0.2, seed=42)
    key = lambda recs: sorted(r["path"] for r in recs)
    assert key(a[0]) == key(b[0]) and key(a[1]) == key(b[1])


def test_class_weights_are_sane_and_bounded():
    tr, _ = source_disjoint_split(_pool_like_records(), val_frac=0.2, seed=42)
    w = compute_class_weights(tr, len(CLASS_LABELS)).tolist()
    # No runaway weight: the collapse bug produced 3.9 vs 0.01 (390x spread).
    assert max(w) / min(w) < 5.0, f"class-weight spread too large: {w}"


def test_class_weights_reject_a_starved_class():
    starved = [{"path": "/x/a.jpg", "label": 0, "group": "g"}]  # only class 0 present
    with pytest.raises(ValueError):
        compute_class_weights(starved, len(CLASS_LABELS))
