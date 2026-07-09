# LeafAlert ML Quality — assessment, protocol, and roadmap

This is the charter for LeafAlert's model quality. The dangerous error for a
toxic-plant detector is a **false negative** (a poison plant called safe), so
every metric and threshold here is weighted toward **toxic-recall**, not overall
accuracy.

## Shipped model: v5 (data-expanded, no distillation)

The current shipped `PlantDetector.mlpackage` is the **v5** model: EfficientNet-B0
(light head, no distillation), trained by `scripts/train_v5.py` on a **5,385-image**
pool (grew from ~1,400 via CC-licensed iNaturalist pulls incl. look-alike hard
negatives), with motion-blur / defocus / occlusion augmentation. Evaluated on the
frozen held-out set (`TrainingData/Testing`, n=362):

| Metric | v4 baseline | **v5 (shipped)** |
|---|---|---|
| Confident toxic→"safe" miss (the dangerous error) | 19.1% | **10.7%** |
| Motion-blur (k=15) toxic→safe flip | ~90% | **12.6%** |
| Toxic surfaced (alert + "verify") at shipped thresholds | 80.5% | **90.5%** |
| Full-alert toxic recall | 67.6% | **85.1%** |
| Safe→toxic false alarm (surfaced) | 31% | **26%** |
| Overall accuracy | 65% | **68.5%** |
| Per-class recall (argmax) | ivy 51 / oak 58 / sumac 80 | ivy 58 / oak 51 / sumac 85 |

v5 Pareto-improves recall **and** false alarms and nearly eliminates the motion-blur
cliff. Remaining weak spot: poison_ivy/oak argmax recall — but their misses are
mostly toxic→**toxic** confusion (still alerts), which is why the *confident*
toxic→safe miss dropped. Provenance: shipped weights come from
`checkpoints/student_v5_full.pth`; re-export with `train_v5.py` (not
`reexport_coreml.py`, which targets the old distilled arch).

## Baseline (v4) that motivated this work — held-out

Evaluated on `TrainingData/Testing` (n=362), verified **0.0% duplicate** of the
training images, so these are honest held-out numbers, not train-set optimism.

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

**Root cause (v4):** a small (~1,400-image), low-diversity dataset with no look-alike
hard negatives → memorization, brittle generalization, a mis-set threshold, and a
distillation teacher that injects noise. v5 addressed all four.

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

1. ✅ **DONE (shipped):** per-class thresholds + "uncertain/verify" abstention band
   + hedged UX + committed eval harness.
2. ✅ **DONE (shipped in v5):** dropped the net-negative distillation; trained
   directly (`train_v5.py`) with blur / occlusion / defocus augmentation — the
   motion-blur cliff fell from ~90% to **12.6%**.
3. ✅ **DONE (shipped in v5):** data expansion — grew the pool to 5,385 images via
   CC-licensed iNaturalist pulls with look-alike hard negatives (Virginia creeper,
   boxelder, Rubus, fragrant/smooth sumac). `TrainingData/Testing` is the frozen
   held-out set. *Follow-ups:* add GBIF for more source diversity; add
   seasonal/regional stratification; the pool is CC-BY-NC-inclusive, so surface a
   NOTICE/credits file and note the NonCommercial provenance at ship time.
4. **Next — Safety architecture:** an OOD / "not a plant" gate (energy score from
   logits; requires exporting logits alongside probabilities), plus active learning
   from the app's existing user-feedback loop. Also worth targeting the residual
   poison_ivy/oak per-class recall with class-specific data.

See also the preprocessing parity contract baked into `scripts/coreml_export.py`.
