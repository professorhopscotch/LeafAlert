"""ingest_feedback_v2 must fail closed on synthetic (DEBUG-injected) entries."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "ingest_feedback_v2.py"


def _load():
    spec = importlib.util.spec_from_file_location("leafalert_ingest_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synthetic_flag_is_skipped_not_ingested():
    v2 = _load()
    entry = {"feedbackStatus": "corrected", "originalPrediction": "poison_ivy",
             "correctedLabel": "safe_plants", "synthetic": True}
    label, status, err = v2.resolve_label(entry)
    assert label is None and status == "synthetic" and err is None


def test_synthetic_status_prefix_is_skipped_too():
    v2 = _load()
    label, status, err = v2.resolve_label({"feedbackStatus": "synthetic_confirmed", "originalPrediction": "poison_oak"})
    assert label is None and status == "synthetic" and err is None


def test_real_correction_still_resolves():
    v2 = _load()
    label, status, err = v2.resolve_label({"feedbackStatus": "corrected", "originalPrediction": "poison_ivy",
                                          "correctedLabel": "safe_plants"})
    assert label == "safe_plants" and status == "corrected" and err is None
