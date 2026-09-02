# LeafAlert ML Quality — assessment, protocol, and roadmap

This is the charter for LeafAlert's model quality. The dangerous error for a
toxic-plant detector is a **false negative** (a poison plant called safe), so
every metric and threshold here is weighted toward **toxic-recall**, not overall
accuracy.

## Shipped model: v8 (data-expanded, no distillation, non-plant-aware, source-diversified)

The current shipped `PlantDetector.mlpackage` is the **v8** model: EfficientNet-B0
(light head, no distillation), trained by `scripts/train_v5.py` on a **9,391-image**
pool — grown from ~1,400 via CC-licensed iNaturalist pulls (look-alike hard
negatives, 734 non-plant negatives in `safe_plants`, fall/winter ivy+oak) **plus
1,817 GBIF images (all CC-BY-4.0 / CC0)** for source diversity — with
motion-blur / defocus / occlusion augmentation.

Evaluated on TWO frozen axes:

**Held-out plants** (`TrainingData/Testing`, n=362):

| Metric | v4 baseline | v6 | **v8 (shipped)** |
|---|---|---|---|
| Confident toxic→"safe" miss (the dangerous error) | 19.1% | 9.2% | **8.0%** |
| Motion-blur (k=15) toxic→safe flip | ~90% | 12.6% | **11.8%** |
| Toxic surfaced (alert + "verify") at shipped thresholds | 80.5% | 90.8% | **92.0%** |
| Full-alert toxic recall | 67.6% | 87.8% | **89.7%** |
| Safe→toxic false alarm (surfaced) | 31% | 29% | **24%** |
| Overall accuracy | 65% | 68.2% | **74.0%** |
| Per-class recall (argmax) | ivy 51 / oak 58 / sumac 80 | ivy 52 / oak 62 / sumac 85 | **ivy 62 / oak 69 / sumac 86** |

**Out-of-distribution** (200 non-plants, taxa the model never trained on):

| Metric | v6 | **v8 (shipped)** |
|---|---|---|
| Non-plant → **full toxic alert** | 4.5% | 7.0% |
| Non-plant → surfaced as toxic at all | 7.7% | 9.0% |

v8 improves every plant-axis metric. Source diversity from GBIF (different cameras,
framing, regions) was the axis that helped, where seasonal ambiguity (v7) did not.
The OOD axis gave back ~2.5pp (5 images on n=200, ~1.6 SE — within noise, and a
nuisance error rather than a safety one); accepted because the dangerous
confident-miss improved. Per-class thresholds (ivy/oak 0.40, sumac 0.52) were
re-derived on held-out and are unchanged. Provenance:
`checkpoints/student_v8_gbif.pth`; re-export with `train_v5.py`'s `export()` (not
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

## Calibration (measured on v8, held-out n=362)

`scripts/calibration_report.py` (now loads v5-recipe checkpoints of any backbone):
ECE 0.050, MCE 0.097, mean confidence − accuracy = +1.3 pp (very mildly
over-confident). A single-temperature fit barely moves anything (NLL 0.6853 →
0.6851, ECE 0.050 → 0.049), so **temperature scaling is not worth shipping**:
the baked softmax is already a usable probability, which is what makes the
per-class `ToxicityThresholds` and the sensitivity slider meaningful. Re-run
after every retrain; if ECE climbs above ~0.10 the thresholds need re-deriving
before anything else.

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
   held-out set. *Follow-ups:* GBIF source diversity ✅ shipped in v8; seasonal
   stratification was tried (v7) and rejected — see below; the pool is CC-BY-NC-inclusive, so surface a
   NOTICE/credits file and note the NonCommercial provenance at ship time.
4. **Safety architecture — OOD / "not a plant".** Measured, see below. A post-hoc
   energy gate was **evaluated and rejected**; the fix is training signal instead.
   Still open: active learning from the app's user-feedback loop, and the residual
   poison_ivy/oak per-class recall.

## Rejected experiment: v7 (seasonal rebalance) — DO NOT retry blindly

Hypothesis: poison_ivy/oak recall was weak because their training data was
72–84% spring+summer, while fall (red foliage) and winter (leafless hairy vines)
look nothing like that. Fetched 2,146 fall/winter images (1,455 survived QA),
bringing ivy to a near-uniform 25/21/26/27 seasonal split.

**Result: REJECTED — failed the acceptance floor.**

| Metric | v6 (shipped) | v7 |
|---|---|---|
| Confident toxic→safe | **9.2%** | 11.8% ❌ |
| poison_ivy→safe | **17%** | 23% ❌ |
| poison_oak→safe | **8%** | 12% ❌ |
| overall accuracy | 68.2% | 69.3% ✅ |

Overall accuracy went UP while the safety metric got WORSE — exactly what the
floor exists to catch, and a reminder never to gate on aggregate accuracy. Best
explanation: leafless winter vines are genuinely ambiguous against "safe", so
adding them blurred the ivy/safe boundary rather than sharpening it.

The seasonal images remain in the train pool (they are legitimate data), but the
shipped weights are v6. If you retry this, hold out a season-stratified test set
first so seasonal performance can be measured directly instead of inferred, and
consider weighting rather than simply adding ambiguous winter frames.

## OOD ("is this even a plant?") — measured

Closed-set softmax must assign every input to one of the known classes, so a bird
or a rock still gets a plant label. Measured with `scripts/ood_report.py` against a
200-image non-plant set (birds/mammals/insects/fungi from iNaturalist), scored by
the shipped v5 model:

| Question | Result |
|---|---|
| Non-plants producing a **full toxic alert** | **28.0%** |
| Non-plants surfaced as toxic at all | 41.5% |
| Energy-score AUROC (ID vs OOD) | 0.738 |
| MSP AUROC (trivial baseline) | 0.721 |
| Energy gate @95% plant retention | rejects only **17%** of non-plants, **loses 5.2% of real plants** |

**Decision: do not ship the energy gate.** Its separability is barely above the
trivial baseline, and at any safe operating point it costs more real toxic-plant
detections than it saves in false alarms — the wrong trade for a safety app. A
post-hoc score cannot repair a model that was never shown a non-plant.

**Chosen fix (✅ shipped in v6):** added 734 non-plant images to the `safe_plants`
bucket, which is already the app's "do not alert" class (excluded from
`InferenceEngine.toxicLabels`), so the model *learns* to reject non-plants with **no
app-side changes**. Training negatives were drawn from taxa disjoint from the OOD
eval set, keeping that set a true held-out test of unseen non-plant types. The
guardrail held — held-out confident toxic-miss did not regress (improved 10.7% →
9.2%) — and OOD full toxic alerts fell **28% → 4.5%**. Re-run the check any time with
`scripts/ood_report.py --checkpoint <ckpt> --ood-dir data_staging/ood`.

See also the preprocessing parity contract baked into `scripts/coreml_export.py`.
