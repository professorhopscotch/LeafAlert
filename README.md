# LeafAlert

An iOS app that watches for poison ivy, poison oak and poison sumac while you
walk, using an on-device Core ML classifier and gait-timed camera captures.
It is a **non-commercial, assistive** tool: it never presents a confident
all-clear, every alert says "verify visually before touching", and a missed
toxic plant is treated as the error that matters most.

## What it does

- **Patrol** — the rear camera runs while you walk; one frame per stride is
  taken at the apex of the walking bounce (sharpest instant) and classified.
  Toxic detections raise a haptic/audio alert and a card you can answer
  (*Correct* / *Wrong* → correction), which feeds the active-learning loop.
- **Map / history** — every detection is logged with location and photo.
- **Plant guide** — what each plant looks like and what to do after contact.
- **Debug dashboard** (DEBUG builds) — live pipeline stats, capture tuning
  sliders, session recording (video + 100 Hz IMU + events) for offline
  analysis, and captured-frame review.

## Build and run

Requires Xcode 15+, iOS 17+. Open `LeafAlert.xcodeproj`, scheme `LeafAlert`.

```bash
xcodebuild build -scheme LeafAlert -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 17' CODE_SIGNING_ALLOWED=NO -derivedDataPath build
```

Tests (47 XCTest unit tests + XCUITest launch-route smoke tests; ~3 min):

```bash
xcodebuild test -scheme LeafAlert -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 17' CODE_SIGNING_ALLOWED=NO -derivedDataPath build
```

The simulator has no camera or motion sensors, so Patrol shows a
"Camera unavailable" state there. DEBUG launch arguments make the rest of the
app drivable headlessly (`xcrun simctl launch booted com.leafalert.app …`):

| Argument | Effect |
|---|---|
| `-launchRoute patrol\|map\|plants\|settings\|debug` | open that screen directly |
| `-autoStartPatrol 1` | start a patrol on the Patrol screen |
| `-injectDetection poison_ivy:0.72` | push a synthetic detection through the live path (with `-autoStartPatrol 1`) |

## ML pipeline (Python)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
```

| Script | Purpose |
|---|---|
| `scripts/fetch_gbif.py`, `fetch_inaturalist.py` | fetch openly licensed images into `data_staging/` |
| `scripts/dataset_qa.py` | quality filters, near-dup and held-out leakage guard, oversized downsizing; commits survivors to the pool |
| `scripts/train_v5.py` | train (`--backbone efficientnet_b0\|efficientnet_b2\|convnext_tiny`) and export Core ML |
| `scripts/evaluate_model.py` | held-out metrics, torch↔Core ML parity, threshold sweep |
| `scripts/operating_point.py` | what the **app** decides on the held-out set, thresholds parsed from the Swift source |
| `scripts/calibration_report.py`, `ood_report.py`, `robustness_report.py` | calibration, out-of-distribution, perturbation robustness |
| `scripts/active_learning.py`, `ingest_feedback_v2.py` | rank uncertain captures; ingest user corrections (fail-closed) |
| `scripts/gait_check.py` | verify the apex sign and measure apex→shutter phase from a recorded walk |
| `scripts/generate_attribution.py` | regenerate `ATTRIBUTION.md` |

The training pool lives in `TrainingData/<class>/`; `TrainingData/Testing` is
the frozen held-out set and is never trained on. Model checkpoints are not
committed; the shipped model is `LeafAlert/Resources/MLModels/PlantDetector.mlpackage`.

**Preprocessing parity contract** (breaks silently if any side drifts):
squash-resize to 224×224 (no crop), per-channel ImageNet normalization and
softmax baked into the exported model, `.scaleFill` on device.

## Documents

- [ML_QUALITY.md](ML_QUALITY.md) — model lineage, held-out/OOD/calibration numbers, accepted and rejected experiments, how numbers are derived
- [DATA_PIPELINE.md](DATA_PIPELINE.md) — data sources, licensing, QA gates
- [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md) — uncertainty ranking and poisoning-resistant feedback ingest
- [GAIT_CAPTURE.md](GAIT_CAPTURE.md) — how gait-timed capture works and the 30-second field check
- [ATTRIBUTION.md](ATTRIBUTION.md) — photographers whose openly licensed images trained the model

## CI

`.github/workflows/ci.yml` builds the app, runs the XCTest/XCUITest suites on
the iPhone 17 simulator, and runs the light Python tests (the ML-stack tests
skip themselves when torch/timm/coremltools are absent).
