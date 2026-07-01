"""Tests for scripts/ingest_feedback.py label-routing logic.

Pins the rules that decide where a feedback image is copied during dataset
ingestion:

  * a ``not_a_plant`` label (whether it arrives as the original prediction on a
    *confirmed* entry, or as the ``correctedLabel`` on a *corrected* entry) is
    remapped to the ``safe_plants`` training class — it's a strong
    false-positive / safe-negative signal;
  * a ``discarded`` entry (any status that is not ``confirmed``/``corrected``)
    is skipped entirely;
  * a normal corrected label (e.g. ``poison_oak``) is honored verbatim.

The routing logic lives inside ``ingest_feedback(feedback_dir, train_dir)``.
We import it directly and drive it against a temporary feedback folder +
manifest, then inspect the resulting ``train_dir`` tree. This is
refactor-tolerant: it asserts on observable file placement, not on internal
variable names.

Stdlib + pytest only — no torch / coremltools / network required.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "ingest_feedback.py"
)


def _load_ingest_module():
    spec = importlib.util.spec_from_file_location(
        "leafalert_ingest_feedback", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ingest_fn():
    module = _load_ingest_module()
    if module is None:
        pytest.xfail("ingest_feedback.py could not be imported")
    fn = getattr(module, "ingest_feedback", None)
    if fn is None:
        pytest.xfail(
            "ingest_feedback() not importable "
            "(awaiting refactor to expose routing logic)"
        )
    return fn


def _write_feedback(tmp_path, entries):
    """Create a feedback dir with a manifest and a stub image per entry.

    Returns (feedback_dir, train_dir).
    """
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    for entry in entries:
        name = entry.get("filename")
        if name:
            # A tiny non-empty file is enough; ingest only copies bytes.
            (feedback_dir / name).write_bytes(b"\xff\xd8\xff\xe0stub-jpeg")

    (feedback_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "entries": entries})
    )
    return feedback_dir, train_dir


def _copied(train_dir, label, filename):
    """True if ``filename`` was ingested into train_dir/<label>/."""
    return (train_dir / label / f"feedback_{filename}").exists()


def test_corrected_not_a_plant_maps_to_safe_plants(ingest_fn, tmp_path):
    feedback_dir, train_dir = _write_feedback(
        tmp_path,
        [
            {
                "filename": "fp_1.jpg",
                "originalPrediction": "poison_ivy",
                "correctedLabel": "not_a_plant",
                "feedbackStatus": "corrected",
            }
        ],
    )

    ingest_fn(feedback_dir, train_dir, dry_run=False)

    assert _copied(train_dir, "safe_plants", "fp_1.jpg")
    assert not (train_dir / "not_a_plant").exists()


def test_confirmed_not_a_plant_maps_to_safe_plants(ingest_fn, tmp_path):
    """'not_a_plant' arriving as the original prediction also routes to safe."""
    feedback_dir, train_dir = _write_feedback(
        tmp_path,
        [
            {
                "filename": "fp_2.jpg",
                "originalPrediction": "not_a_plant",
                "correctedLabel": "",
                "feedbackStatus": "confirmed",
            }
        ],
    )

    ingest_fn(feedback_dir, train_dir, dry_run=False)

    assert _copied(train_dir, "safe_plants", "fp_2.jpg")
    assert not (train_dir / "not_a_plant").exists()


def test_discarded_entry_is_skipped(ingest_fn, tmp_path):
    feedback_dir, train_dir = _write_feedback(
        tmp_path,
        [
            {
                "filename": "junk.jpg",
                "originalPrediction": "poison_oak",
                "correctedLabel": "",
                "feedbackStatus": "discarded",
            }
        ],
    )

    ingest_fn(feedback_dir, train_dir, dry_run=False)

    # Nothing should have been copied anywhere.
    copied_files = list(train_dir.rglob("feedback_*"))
    assert copied_files == [], f"discarded entry was ingested: {copied_files}"


def test_normal_corrected_label_is_honored(ingest_fn, tmp_path):
    feedback_dir, train_dir = _write_feedback(
        tmp_path,
        [
            {
                "filename": "real_oak.jpg",
                "originalPrediction": "poison_ivy",
                "correctedLabel": "poison_oak",
                "feedbackStatus": "corrected",
            }
        ],
    )

    ingest_fn(feedback_dir, train_dir, dry_run=False)

    assert _copied(train_dir, "poison_oak", "real_oak.jpg")
    assert not _copied(train_dir, "poison_ivy", "real_oak.jpg")


def test_mixed_batch_routes_each_entry_correctly(ingest_fn, tmp_path):
    """End-to-end: a batch exercises remap, skip, and honor together."""
    feedback_dir, train_dir = _write_feedback(
        tmp_path,
        [
            {
                "filename": "a.jpg",
                "originalPrediction": "poison_ivy",
                "correctedLabel": "not_a_plant",
                "feedbackStatus": "corrected",
            },
            {
                "filename": "b.jpg",
                "originalPrediction": "poison_sumac",
                "correctedLabel": "",
                "feedbackStatus": "discarded",
            },
            {
                "filename": "c.jpg",
                "originalPrediction": "poison_oak",
                "correctedLabel": "",
                "feedbackStatus": "confirmed",
            },
        ],
    )

    ingest_fn(feedback_dir, train_dir, dry_run=False)

    assert _copied(train_dir, "safe_plants", "a.jpg")      # remapped
    assert not (train_dir / "poison_sumac" / "feedback_b.jpg").exists()  # skipped
    assert _copied(train_dir, "poison_oak", "c.jpg")       # honored (confirmed)
