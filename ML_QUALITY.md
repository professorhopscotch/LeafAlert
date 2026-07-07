# LeafAlert ML Quality — assessment, protocol, and roadmap

This is the charter for LeafAlert's model quality. The dangerous error for a
toxic-plant detector is a **false negative** (a poison plant called safe), so
every metric and threshold here is weighted toward **toxic-recall**, not overall
accuracy.

## Current measured state (held-out)

Evaluated on `TrainingData/Testing` (n=362), verified **0.0% duplicate** of the
training images, so these are honest held-out numbers, not train-set optimism.
Torch and the shipped Core ML model agree to 96.7% (argmax).

| Metric | Value |
|---|---|
| Overall accuracy | 65% |
| Toxic-recall @ old 0.65 threshold | **43.5%** (missed >half of toxic plants) |
| Toxic argmaxed as "safe" (threshold-free floor) | **19%** |
| Per-class recall (argmax) | ivy **51%**, oak 58%, sumac 80% |
| Train vs held-out toxic-recall @0.65 | 77.6% → 43.5% (severe overfitting) |
| Calibration (ECE) | 4.1% (slightly under-confident) |
| Motion-blur toxic→safe miss-rate | 4.4% → **58%** (confidently wrong) |
| Distillation teacher vs student | 62% vs 77% (teacher is **net-negative**) |

**Root cause:** a small (~1,400-image), low-diversity dataset with no look-alike
hard negatives → memorization, brittle generalization, a mis-set threshold, and a
distillation teacher that injects noise. Thresholds are a bounded stopgap: reaching
≥80% recall costs ~50% false alarms. **Only better data moves the frontier.**

## Threshold / operating point (stopgap, shipped)

The alert threshold and per-class logic live in
`LeafAlert/Models/ToxicityThresholds.swift`. Default sensitivity is now **0.50**
(was 0.65). Detections are surfaced in three severities:

- **alert** — full haptic + audio warning ("Likely …").
- **uncertain** — a near-miss below the alert bar, surfaced as "Possible … · verify
  visually" with a soft silent nudge. The app never presents a confident all-clear.
- **ignore** — below the noise floor.

Re-derive these after every retrain; they are model-specific.

## Evaluation tooling (durable)

```sh
# Confusion matrix, per-class P/R/F1, toxic-recall + miss-rate, threshold sweep
.venv/bin/python scripts/evaluate_model.py --checkpoint checkpoints/student_distilled.pth \
    --coreml LeafAlert/Resources/MLModels/PlantDetector.mlpackage --data TrainingData/Testing --split held-out

.venv/bin/python scripts/calibration_report.py   # ECE, reliability, temperature, threshold sweep
.venv/bin/python scripts/robustness_report.py    # perturbation degradation (blur/occlusion/…)
.venv/bin/python scripts/audit_dataset.py        # counts, dupes, leakage, resolution
```

**Acceptance floor (local gate before shipping any model):** held-out toxic-recall
at the chosen operating point must not regress, and the toxic→safe hard-miss rate
must not increase. `TrainingData/Testing` is the frozen held-out set — never train
or tune on it.

## Roadmap (safety-leverage first)

1. **Now (shipped):** per-class thresholds + "uncertain/verify" abstention band +
   hedged UX + committed eval harness. On held-out: toxic plants **surfaced**
   (alert or "verify visually") rises **43.5% → 80.5%**, full alarms hit 67.6%
   recall, at ~31% false-alarm (mostly soft "possible — verify", not full alarms).
2. **Retrain (existing data):** drop the net-negative distillation; train directly
   with strong **blur / occlusion / defocus** augmentation to kill the motion-blur
   cliff; re-export via `scripts/reexport_coreml.py`; re-derive thresholds.
3. **Data expansion (the real fix):** pull CC-licensed iNaturalist / GBIF images to
   grow volume + seasonal/regional diversity, and add explicit **look-alike hard
   negatives** (Virginia creeper, boxelder, Rubus/blackberry, fragrant & smooth
   sumac) into `safe_plants`; build a frozen, source-split field test set.
4. **Safety architecture:** an OOD / "not a plant" gate (energy score from logits;
   requires exporting logits alongside probabilities), plus active learning from the
   app's existing user-feedback loop.

See also the preprocessing parity contract baked into `scripts/coreml_export.py`.
