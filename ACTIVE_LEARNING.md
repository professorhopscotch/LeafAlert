# LeafAlert Active-Learning Runbook

The continuous-improvement loop: **collect** user feedback from the app →
**triage** it (with a safety gate on dangerous labels) → **prioritize** the most
informative unlabeled captures → **QA / leakage-guard** → **retrain + evaluate**
against the frozen held-out set with a hard acceptance floor.

Every command below is copy-pasteable from the repo root.

```bash
cd /Users/burkley/Documents/claude/LeafAlert
PY=/Users/burkley/Documents/claude/LeafAlert/.venv/bin/python
```

---

## Contracts you MUST NOT break

| Contract | Rule |
| --- | --- |
| **Classes** | `poison_ivy, poison_oak, poison_sumac, safe_plants` — ImageFolder-alphabetical, do NOT reorder. Toxic = first three. `safe_plants` is both the safe-plant class and the app's "do not alert" bucket (it also holds non-plant negatives). |
| **Held-out FREEZE** | `TrainingData/Testing/` (n=362) is frozen. **Never** add to it, never train on it, never tune on it. |
| **Train pool** | New data lands ONLY in `TrainingData/<class>/`, and ONLY via `scripts/dataset_qa.py --commit`. |
| **Preprocessing parity** | Squash-resize to 224×224 (`Resize((224,224))`, **no** center crop) → ImageNet normalize. Export bakes normalize + softmax; the model takes raw 0–255 RGB. `active_learning.py` imports the transform from `train_v5` and asserts parity at runtime. |
| **Shipped model** | v6 = `PlantDetectorV5` (EfficientNet-B0, `head="linear"`), weights `checkpoints/student_v6_nonplant.pth`. |

**Current pool (baseline for this loop):** ivy 1728 · oak 1296 · sumac 669 ·
safe 2426 = **6119** train, **362** frozen held-out.

---

## Read this before you start: what the loop can and cannot fix

The app only creates a feedback image when a toxic class already **beat**
`safe_plants` (`InferenceEngine.swift` returns no `DetectionResult` otherwise,
and `AppState.swift` drops nil results before logging). So a frame the model
calls "safe" never becomes feedback.

**Consequence: the feedback stream structurally cannot contain the model's
dangerous misses** (poison_ivy→safe 18%, poison_oak→safe 8%). Fed only feedback,
this loop reduces **false alerts**, not the miss rate.

To move the miss rate you must feed the loop frames where **no alert fired**.
The available source is `DebugFrameSaver` (`LeafAlert/Stores/DebugFrameSaver.swift`),
which saves *every* captured frame to the app's `Documents/debug_frames/`. Pull
that folder and rank it with `active_learning.py` (Step 3) — those captures are
genuinely unlabeled, which is exactly what the selector wants.

> **Rule of thumb:** feedback pool → fixes false alerts. debug_frames pool →
> fixes dangerous misses. Run both; only the second one moves the acceptance-floor
> metric.

---

## Step 0 — Smoke-test the tooling (~1 min, safe)

```bash
$PY -m py_compile scripts/active_learning.py scripts/ingest_feedback_v2.py
$PY scripts/ingest_feedback_v2.py --feedback-dir feedback          # dry run, writes nothing
$PY scripts/evaluate_model.py --checkpoint checkpoints/student_v6_nonplant.pth \
    --arch v5 --split held-out --limit 3 --skip-coreml --threshold 0.50
```

Until the first sync lands, the ingest command exits **1** with
`ERROR: no feedback manifest at feedback` and a list of the iCloud paths it
probed. That is the expected empty-state, not a failure.

---

## Step 1 — Collect

The app writes feedback via `LeafAlert/Stores/FeedbackExporter.swift` into its
Documents dir as `<timestamp>_<correctedLabel>_<status>.jpg` plus a
`manifest.json` whose entries carry `filename, originalPrediction,
correctedLabel, feedbackStatus, confidence, timestamp, latitude, longitude`.
`feedbackStatus` ∈ `confirmed | corrected | discarded`; `correctedLabel` may be
`not_a_plant` (→ `safe_plants`).

**Two transports — use either or both (ingest de-duplicates downstream):**

**A. LAN server (pull on demand).** Advertises over Bonjour as
`_leafalert._tcp`; the app discovers it automatically. Mac and phone on the same
Wi-Fi.

```bash
$PY scripts/feedback_server.py --port 8847 --output-dir feedback
# health: curl http://localhost:8847/status   ->  {"status":"ok","entries":N}
```

Leave it running, trigger the sync from the app, then Ctrl-C. Files land in
`feedback/` with a merged `manifest.json`.

**B. iCloud Drive (continuous).** If a sync folder was chosen in the app's
Settings, feedback is mirrored there and syncs to the Mac on its own. **No flag
needed** — when `--feedback-dir` has no `manifest.json`, ingest automatically
falls back to, and prints, the first of these that does:

```
~/Library/Mobile Documents/com~apple~CloudDocs/LeafAlert/feedback
~/Library/Mobile Documents/com~apple~CloudDocs/feedback
```

Point at it explicitly only if your sync folder is somewhere else:

```bash
$PY scripts/ingest_feedback_v2.py --feedback-dir "/path/to/your/sync/folder"
```

**Unlabeled captures (the pool that matters — see the warning above).** Pull
`Documents/debug_frames/` off the device (Xcode → Devices and Simulators →
LeafAlert → Download Container, or the Files app if sharing is enabled) into
`data_staging/debug_frames/`. No manifest needed; the selector scores raw images.

**Checkpoint:** `ls feedback/*.jpg | wc -l` and confirm `manifest.json` parses.

---

## Step 2 — Triage

```bash
# 1. DRY RUN first (default). Reports only, writes nothing.
$PY scripts/ingest_feedback_v2.py --feedback-dir feedback --show 20

# 2. Stage the accepted images for QA (still does NOT touch TrainingData)
$PY scripts/ingest_feedback_v2.py --feedback-dir feedback --commit-staging \
    --json data_staging/ingest_report.json
```

Accepted images → `data_staging/feedback/<class>/` + `manifest.jsonl`
(provenance in the shape `dataset_qa.py` reads).
Quarantined images → `data_staging/feedback_needs_review/` + `review_queue.jsonl`.
Re-running is idempotent; already-staged files are left untouched.

### Trust policy

Orthogonal to the safety gate, and applied **after** it — no policy value can
let an un-reviewed dangerous label through.

| Flag | Effect | Use when |
| --- | --- | --- |
| `--trust-policy all` (default) | Accept confirmations + corrections | Normal operation, trusted testers |
| `--trust-policy corrections-only` | Drop confirmations | Confirmations mostly re-teach what the model knows and entrench bias; use once the pool is large |
| `--trust-policy high-confidence` | Require `confidence >= --min-confidence` (0.70) | Noisy or public feedback |
| `--review-all-demotions` | Send **every** toxic→safe correction to review, at any confidence | Before a release; strictest setting |
| `--contradiction-conf 0.70` | Confidence at/above which a toxic→safe demotion is quarantined | Lower it to quarantine more |
| `--no-geo` | Omit raw lat/lng from the staging manifest (keeps the derived `observation_id`) | Any dataset that may outlive the feedback folder |

### The `needs_review` bucket — a human MUST adjudicate this

**Why.** Feedback error is asymmetric in a poison-plant app:

* A wrong **"actually safe"** label teaches the model to **miss a toxic plant**.
  Cost: someone touches poison ivy.
* A wrong **"actually poison ivy"** label causes a false alert. Cost: annoyance.

Those are not the same mistake. Worse, the dangerous label is the one a careless
user most easily produces — dismissing an alert as "safe" takes one tap and no
looking. Bulk-accepting user "safe" labels is the single fastest way to train the
miss rate back in.

So `ingest_feedback_v2.py` quarantines **toxic → safe demotions** that contradict
a confident prediction, and **fails closed**: if `originalPrediction` is missing
or non-canonical, any correction landing on `safe_plants` is quarantined
regardless of confidence. (The manifest arrives over the LAN and is untrusted
input; a format change on the app side must not silently disable the gate.)

The reverse direction (model said safe, user says toxic) is **accepted and given
top priority** — it is the measured v6 weakness, and gating it would starve the
fix of its best signal. Cross-toxic corrections (ivy↔oak↔sumac) are accepted at
low priority: operationally harmless, both still alert.

**Adjudicate from the top of the queue** (it is sorted by priority):

```bash
$PY - <<'EOF'
import json, pathlib
q = pathlib.Path("data_staging/feedback_needs_review/review_queue.jsonl")
for line in q.read_text().splitlines():
    r = json.loads(line)
    if r.get("resolution", "pending") == "pending":
        print(f"{r['priority']:.2f}  {r['filename']}\n"
              f"      model said {r['original_prediction']} @ {r['model_confidence']:.2f}"
              f"  ->  user says {r['proposed_label']}   ({r['review_reason']})")
EOF
```

For each one, **open the image and decide yourself**. You are answering: *is the
plant in this photo actually harmless?* Then:

```bash
# APPROVED (the user was right — it really is safe):
mkdir -p data_staging/feedback_reviewed/safe_plants
mv data_staging/feedback_needs_review/<file>.jpg data_staging/feedback_reviewed/safe_plants/

# REJECTED (the model was right, the user dismissed a real toxic plant):
#   delete the file, or leave it in place — either way it never becomes training data.
```

Record the outcome by setting `"resolution"` to `approved` / `rejected` in
`review_queue.jsonl`. Re-running ingest **preserves** any non-`pending`
resolution, so decisions are never clobbered.

Then run approvals through the same gate as everything else:

```bash
$PY scripts/dataset_qa.py --staged data_staging/feedback_reviewed --tag feedback-reviewed
$PY scripts/dataset_qa.py --staged data_staging/feedback_reviewed --tag feedback-reviewed --commit
```

> **Caveat:** moving files loses the `observation_id` that keeps `train_v5.py`'s
> split observation-disjoint. Keep review batches small, or hand-copy the matching
> rows into a `manifest.jsonl` in `data_staging/feedback_reviewed/`.

**If the review queue is growing faster than you can adjudicate it, that is a
signal — not a backlog to bulk-approve.** Tighten with
`--trust-policy corrections-only` and investigate whether one tester is
mass-dismissing alerts.

---

## Step 3 — Prioritize (rank unlabeled captures by informativeness)

Labeling is the scarce resource. `scripts/active_learning.py` scores an
unlabeled pool with the shipped v6 model and emits a ranked worklist, dropping
anything that would leak the held-out set or duplicate the pool.

```bash
mkdir -p data_staging/worklists

# THE PRIMARY RUN — genuinely unlabeled field captures, safety-weighted
$PY scripts/active_learning.py \
    --pool data_staging/debug_frames \
    --strategy safety --top 50 \
    --out data_staging/worklists/round1.jsonl
```

Writes `round1.jsonl` (line 1 = run provenance, then one record per candidate
with per-class probs, every uncertainty measure, the app's alert severity, and a
`suggested_label` so a human confirms with one keystroke) and a sibling
`round1.txt` table for the labeler.

> **First run is slow.** Indexing the ~6.5k reference images for the leakage
> guard takes 30 s–10 min (I/O bound). It is cached at
> `~/.cache/leafalert/active_learning_ref_index.json`, so re-running with a
> different `--strategy` or `--top` is near-instant.

### Which strategy

| Strategy | Ranks by | Use when |
| --- | --- | --- |
| `safety` | **Recommended for this app.** `0.7 × boundary + 0.3 × threshold-proximity`, where boundary = `2·min(p_toxic, p_safe)` — maximal at the toxic-vs-safe coin flip — plus proximity to the app's real alert thresholds (ivy/oak 0.40, sumac 0.52, window 0.20). | You want labels that change the **alert/no-alert** decision. An ivy-vs-oak tie is worthless (both alert); a toxic-vs-safe tie is the whole ballgame. |
| `margin` (default) | Top-1 minus top-2 probability | General accuracy, broad coverage, per-class recall |
| `entropy` | Full-distribution uncertainty | Similar to margin, more sensitive to flat distributions |
| `least_confidence` | `1 − p_top` | Simple baseline |

Concretely: an ivy/oak near-tie `[.50,.50,0,0]` scores **1.000** under `margin`
but **0.150** under `safety`, while `[.45,.05,0,.50]` — a poison_ivy the app is
currently ignoring — scores **0.950** under `margin` and **0.925** under
`safety`. The two strategies deliberately rank them in opposite order.

> `safety` is principled but **not yet validated against outcomes**. The honest
> test is a controlled round: label N images picked by `margin` vs N by `safety`,
> retrain, compare poison_ivy→safe recall. Do that before treating it as settled.

### Pool caveats

* **Do not use `--pool feedback` as your main pool.** Images the manifest marks
  `confirmed`/`corrected` are skipped by default (a human already labeled them),
  so a feedback folder mostly filters down to `discarded` rows. Watch the
  `skipped: already labeled` line in the accounting block. Use
  `--include-labeled` if you specifically want to re-rank labeled feedback.
* `--limit N` takes the **first N by path** — and feedback filenames start with
  an ISO timestamp, so `--limit` gives you the *oldest* N, not a sample.
* Only `.jpg/.jpeg/.png` are scanned. Other formats are invisible (not reported).
* `--no-leak-guard` exists for iteration only. It prints a warning.
  `dataset_qa.py` remains the authoritative gate.

Label the worklist top-down, stage the results by class, and go to Step 4. Stop
when the marginal image stops being informative — check the printed top-N class
distribution; the tool warns when ≥70% of a round collapses onto one predicted
class, which means the round teaches the model nothing new.

---

## Step 4 — QA / dedup / LEAKAGE GUARD (mandatory)

Nothing enters `TrainingData/` except through this gate. It enforces quality
filters, exact + perceptual de-duplication, and — the point — drops any staged
image that near-duplicates the **frozen held-out set** or the existing pool.

```bash
# DRY RUN first — always
$PY scripts/dataset_qa.py --staged data_staging/feedback --tag feedback

# Commit survivors into TrainingData/<class>/ + append provenance
$PY scripts/dataset_qa.py --staged data_staging/feedback --tag feedback --commit \
    --json data_staging/qa_feedback.json
```

Read the drop report before committing. A nonzero **"leak vs held-out"** count is
expected and healthy — it means the guard caught a user photographing a plant
that a held-out image came from. A *surprisingly high* count means the pool is
contaminated; investigate rather than loosening `--leak-hamming`.

**Checkpoint after commit:**

```bash
find TrainingData/Testing -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l   # MUST still be 362
for d in TrainingData/*/; do echo -n "$d "; \
  find "$d" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l; done
```

If `Testing` ≠ 362, **stop** — something wrote to the frozen set.

---

## Step 5 — Retrain + evaluate

Retrain only when the pool has meaningfully grown (see cadence). This is the
heavy job — hours on MPS/GPU.

```bash
$PY scripts/train_v5.py --epochs 40
```

Outputs `checkpoints/plant_detector_v5.pth` and re-exports
`LeafAlert/Resources/MLModels/PlantDetector.mlpackage` (normalize + softmax
baked). Model selection is on val toxic-recall. The split is
source/observation-disjoint with a hard overlap assertion; the frozen held-out
set is never read.

### Evaluate — the new model AND the incumbent, same command

Like-for-like or it proves nothing:

```bash
# NEW model
$PY scripts/evaluate_model.py \
    --coreml LeafAlert/Resources/MLModels/PlantDetector.mlpackage \
    --data TrainingData/Testing --split held-out \
    --threshold 0.50 --sweep 0.40,0.50,0.52,0.65 \
    --json data_staging/eval_new.json

# INCUMBENT (v6) on the identical set + thresholds
$PY scripts/evaluate_model.py \
    --checkpoint checkpoints/student_v6_nonplant.pth --arch v5 --skip-coreml \
    --data TrainingData/Testing --split held-out \
    --threshold 0.50 --sweep 0.40,0.50,0.52,0.65 \
    --json data_staging/eval_v6.json
```

### Acceptance floor — a release GATE, not a suggestion

| Metric | Report column | v6 baseline | Requirement |
| --- | --- | --- | --- |
| **Confident toxic→safe miss** | `->safe(hard)` / `toxic_to_safe_hard` | **9.2%** | **Must not exceed 9.2%.** Threshold-free (argmax-based), so it is directly comparable at any sweep row. |
| **Toxic-recall** | `toxic-recall` / `toxic_recall` | 87.8% full-alert | **Must not drop** vs the incumbent at the **same** threshold. |
| Toxic surfaced (alert + verify) | `1 − miss-total` | 90.8% | Should not drop |
| poison_ivy recall | per-class table | — | Must not regress |
| safe→toxic false alarm | `safe->toxic alarm` | ~29% | Soft cap — should not blow up |

**Hard rule: if the confident toxic→safe miss rises above 9.2%, or toxic-recall
regresses at the shipping threshold, DO NOT SHIP.** A safety-critical poison
detector must never get worse at catching poison. A recall gain paid for with a
higher hard-miss rate is not a gain.

Also re-check that non-plants still route to the do-not-alert bucket (v6:
4.5% full toxic alert on OOD):

```bash
$PY scripts/ood_report.py --checkpoint checkpoints/plant_detector_v5.pth \
    --ood-dir data_staging/ood
```

### After acceptance

1. Re-derive the per-class thresholds in `LeafAlert/Models/DetectionResult.swift`
   (`ToxicityThresholds`) from the new sweep — they are model-specific.
2. Update the baseline table in `ML_QUALITY.md` so the next round's
   "never regress" gate has a reference.
3. Regenerate attribution if the pool grew: `$PY scripts/generate_attribution.py`.
4. Commit the new `.mlpackage`. Do not overwrite the prior one until the gate passes —
   it is the rollback target.

---

## Recommended cadence

| Cadence | Action | Command |
| --- | --- | --- |
| **Continuous** | iCloud sync folder configured in the app; nothing to run | — |
| **Weekly** | Pull feedback (LAN server if not using iCloud), run ingest **dry run**, skim the counts | `feedback_server.py`, then `ingest_feedback_v2.py --feedback-dir feedback` |
| **Weekly** | Adjudicate `needs_review` — keep it at zero. This is a small, high-value human task; let it rot and it stops being done at all. | see Step 2 |
| **Every 2–4 weeks** (or when ≥200 new unlabeled captures exist) | Pull `debug_frames`, run one active-learning round with `--strategy safety --top 50`, label it | `active_learning.py` |
| **After every labeling round** | Stage → `dataset_qa.py` dry run → `--commit`. Verify `Testing` is still 362. | `dataset_qa.py` |
| **When the pool grows ≥10% (~600 net images) or every 6–8 weeks, whichever comes first** | Retrain + evaluate against the acceptance floor | `train_v5.py`, `evaluate_model.py` |
| **Before any release** | Full eval + OOD check + re-derive thresholds; run ingest with `--review-all-demotions` | see Step 5 |

Retraining more often than the pool grows is wasted compute — a 40-epoch run on
50 new images will not move a held-out set of 362.

---

## What NOT to do

* **Never train, tune, or threshold-fit on `TrainingData/Testing/`.** It is the
  only honest measurement in the project. Adding one image to it silently
  inflates every safety number you report. If the count is not 362, stop.
* **Never bypass `dataset_qa.py`.** No `cp` into `TrainingData/`. Feedback is the
  most likely leakage vector here — a user can photograph the same plant a
  held-out image came from, and the same photo can arrive twice via iCloud *and*
  the LAN server. `--no-leak-guard` in `active_learning.py` is for iteration
  only; it is not a QA path.
* **Never bulk-accept low-confidence user "safe" labels.** That is the exact
  input that trains the dangerous miss back in, and it is the easiest wrong label
  for a user to produce. Every toxic→safe demotion gets a human eye. Do not
  "clear the queue" by approving in bulk.
* **Never loosen `--leak-hamming` / `--dup-hamming` to raise the keep rate.** A
  high drop count is the guard working. Fix the pool, not the gate.
* **Never ship a model whose confident toxic→safe miss exceeds 9.2%** or whose
  toxic-recall regressed, regardless of how much overall accuracy improved.
* **Never change preprocessing on one side only.** Train and device must stay on
  squash-resize 224×224 + ImageNet normalize. A silent desync degrades accuracy
  with no error anywhere.
* **Never reorder or rename the classes.** Indices are baked into the exported
  model and the app.
* **Do not treat feedback volume as progress.** A thousand confirmations of
  correct alerts teach the model almost nothing. Fifty well-chosen labels from an
  active-learning round are worth more.

---

## Files in this loop

| Path | Role |
| --- | --- |
| `LeafAlert/Stores/FeedbackExporter.swift` | Writes feedback images + `manifest.json` on device; mirrors to iCloud |
| `LeafAlert/Stores/DebugFrameSaver.swift` | Saves every captured frame → the unlabeled pool |
| `scripts/feedback_server.py` | LAN/Bonjour receiver → `feedback/` |
| `scripts/ingest_feedback_v2.py` | Triage + asymmetric safety gate → `data_staging/feedback/` |
| `scripts/ingest_feedback.py` | Legacy direct-to-train-pool path. Superseded; prefer v2 |
| `scripts/active_learning.py` | Ranks unlabeled captures by informativeness |
| `scripts/dataset_qa.py` | **The gate.** QA + dedup + leakage guard → `TrainingData/` |
| `scripts/train_v5.py` | Trains v5/v6 arch + exports Core ML |
| `scripts/evaluate_model.py` | Held-out safety metrics + threshold sweep |
| `scripts/ood_report.py` | Non-plant false-alert rate |
| `DATA_PIPELINE.md` | Bulk data-expansion runbook (fetch → QA → retrain) |
| `ML_QUALITY.md` | Model quality charter + baseline numbers |

## The feedback card (fixed 2026-09-02)

The loop only works if people can answer the card. Since c28ec68 the card
disappeared 2.5 s after the last actionable detection — it was bound to the
same property as the bounding box, which is expired quickly on purpose — so in
practice *Correct* / *Wrong* was almost never tappable. The card now stays for
`AppState.detectionCardLifetime` (20 s) or until answered or dismissed; only the
box expires after 2.5 s. `-injectDetection poison_ivy:0.72` (DEBUG, with
`-autoStartPatrol 1`) pushes a synthetic detection through the live path so the
card and the correction flow can be exercised in the simulator; three XCUITests
cover it.
