# LeafAlert Data-Expansion & Retrain Runbook

A sequenced, copy-pasteable runbook to close the model's held-out quality gap
(baseline: **held-out toxic-recall 43.5% @ 0.65**, poison_ivy recall 51%,
**58% of toxic images flip toxic→safe under motion blur**, ~1400 train images
with **no look-alike hard negatives**, distillation net-negative).

The plan: **fetch** more toxic + look-alike data (season/region-stratified) →
**QA / dedup / leakage-guard** → **commit** into `TrainingData/` → **retrain**
with `train_v5.py` (no distillation, field-failure augmentation) → **re-export**
Core ML → **evaluate against the frozen held-out set** with a hard acceptance
floor (toxic-recall and toxic→safe miss-rate must **improve, never regress**).

All commands run from the repo root
`/Users/burkley/Documents/claude/LeafAlert` using the project venv.

```bash
cd /Users/burkley/Documents/claude/LeafAlert
PY=/Users/burkley/Documents/claude/LeafAlert/.venv/bin/python
```

---

## Contracts you MUST NOT break

| Contract | Rule |
| --- | --- |
| **Classes** | `poison_ivy, poison_oak, poison_sumac, safe_plants` (ImageFolder-alphabetical, do NOT reorder). Toxic = first three. `safe_plants` = negative / look-alike bucket. |
| **Held-out FREEZE** | `TrainingData/Testing/` (n=362) is the frozen, leakage-free held-out set. **NEVER** add fetched data to it. Every new image is de-duplicated against it (perceptual hash, Hamming ≤ 5). |
| **Train pool** | New data lands ONLY in `TrainingData/{class}/`. |
| **Preprocessing parity** | Squash-resize whole image to 224×224 (`Resize((224,224))`, NO center crop) → per-channel ImageNet normalize `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]`. Export bakes normalize + softmax; the model takes raw 0–255 RGB. Do not change this on either side. |

**Current train-pool baseline (before this run):**

| class | train pool now | held-out (frozen) |
| --- | --- | --- |
| poison_ivy | 332 | 84 |
| poison_oak | 309 | 78 |
| poison_sumac | 396 | 100 |
| safe_plants | 400 | 100 |
| **total** | **1437** | **362** |

---

## Recommended target counts

Bring each **toxic** class to **~800–1200** train-pool images, and grow the
**look-alike hard negatives** (the single biggest missing ingredient) to
**~800–1000** `safe_plants` with real ivy/oak/sumac mimics. Fetch **more than the
gap** because QA + dedup + leakage-guard typically drops **25–40%** of raw fetches.

| class | now | target | net to add | fetch (≈1.5× gap) |
| --- | --- | --- | --- | --- |
| poison_ivy | 332 | **1000** | ~668 | ~1000 |
| poison_oak | 309 | **1000** | ~691 | ~1050 |
| poison_sumac | 396 | **1000** | ~604 | ~900 |
| safe_plants (look-alikes) | 400 | **900** | ~500 | ~800 |
| **total** | 1437 | **~3900** | ~2460 | ~3750 |

Target a roughly balanced pool so the inverse-frequency class weights in
`train_v5.py` don't have to work as hard. `safe_plants` should be *look-alikes*,
not random flora — a Virginia-creeper/boxelder/sumac-mimic negative is worth far
more than a photo of a rose.

### Stratify every toxic class across season and region

The motion-blur cliff and the poison_ivy recall hole both partly reflect
**appearance drift** (red spring leaves, green summer, fall color, leafless
winter vines) the model never saw. Split each toxic class's fetch into seasonal
buckets and pull US-wide (not one region):

- **Spring (red/emerging):** `--months 3,4,5`
- **Summer (green):** `--months 6,7,8`
- **Fall (color):** `--months 9,10`
- **Winter (leafless vines):** `--months 11,12,1,2`

Aim for roughly **35% summer / 25% spring / 25% fall / 15% winter** per toxic
class. iNaturalist defaults to `--place-id 1` (United States); leave it US-wide.

---

## Data volume & time estimates

| Stage | Volume | Wall-clock (rough) | Notes |
| --- | --- | --- | --- |
| Fetch (iNat + GBIF) | ~3750 raw images, ~1.5–4 GB | **1–3 h** | Rate-limited: iNat ≤60 req/min, ≤10k/day; GBIF politeness sleep. Resumable — safe to Ctrl-C and re-run. |
| QA / dedup / leakage | scans ~3750 staged + ~1800 reference | **3–8 min** | O(n²) near-dup scan; fine at this size. |
| Commit | copies survivors | seconds | Idempotent. |
| **Retrain (`train_v5.py --epochs 40`)** | ~3900 images | **1–4 h GPU / much longer CPU** | The heavy job. Run on a machine with a GPU/MPS. |
| Re-export Core ML | — | <1 min | |
| Evaluate held-out | 362 images | 1–2 min | |

> **These heavy jobs (bulk fetch, full training) are intentionally NOT run by the
> pipeline-building agent.** Run them yourself on real hardware.

---

## Licensing & attribution

- **iNaturalist:** default `--license cc0,cc-by` (commercial-safe). Add
  `cc-by-nc` only if NonCommercial use is acceptable — it roughly triples
  availability but adds a NonCommercial constraint. Each photo's license +
  attribution + photographer are recorded per-row in the staging manifest.
- **GBIF:** restricted to `{CC0_1_0, CC_BY_4_0}` at both the server (OR-filter)
  and per-media level. Per-media licenses that are present but disallowed are
  now **rejected** (not silently relabeled with the occurrence license).
- **Attribution obligation:** CC-BY (and CC-BY-NC) require crediting the
  photographer. The commit manifest (`qa_commit_manifest.json`) and
  `TrainingData/manifest.jsonl` carry `license` + `attribution` + `source_url`
  for every committed image — keep them; surface them in an app "Image credits"
  / NOTICE file at ship time.
- **GBIF citation:** the occurrence key per image makes the pull reproducible;
  cite GBIF if you publish the dataset.

---

# The pipeline

## Step 0 — Smoke-test the tooling (safe, tiny, ~1 min)

Prove every stage works end-to-end before committing to a multi-hour fetch.

```bash
# Fetchers: tiny live pulls (a few real images each)
$PY scripts/fetch_inaturalist.py --verify-taxa                       # all 11 taxon IDs OK
$PY scripts/fetch_inaturalist.py --taxon "Toxicodendron radicans" --limit 3
$PY scripts/fetch_gbif.py --classes poison_ivy --limit 3 --dry-run

# QA: dry-run against the real pool + held-out (writes nothing)
$PY scripts/dataset_qa.py --staged data_staging

# Trainer: wiring check (offline, tiny subset, no export)
$PY scripts/train_v5.py --smoke
```

---

## Step 1 — Fetch (season/region-stratified)

Fetches stage into `data_staging/{inaturalist,gbif}/<class>/` with a per-row
`manifest.jsonl` (provenance + license + attribution + observation id). Both
fetchers are **idempotent/resumable** — re-running the same command tops up
toward the target and re-downloads nothing already staged. Every fetched image
is de-duplicated against the frozen held-out set at fetch time.

### 1a. iNaturalist — primary source, season-stratified toxic classes

`--per-taxon` is a **total** target per taxon (counts what is already staged).
The toxic classes have 1–2 taxa each, so per-taxon ≈ per-class for those.

```bash
# poison_ivy (taxa: T. radicans + T. rydbergii) — ~1000 total, season-stratified
for M in "3,4,5" "6,7,8" "9,10" "11,12,1,2"; do
  $PY scripts/fetch_inaturalist.py --class poison_ivy --per-taxon 260 --months "$M"
done

# poison_oak (T. diversilobum + T. pubescens)
for M in "3,4,5" "6,7,8" "9,10" "11,12,1,2"; do
  $PY scripts/fetch_inaturalist.py --class poison_oak --per-taxon 260 --months "$M"
done

# poison_sumac (T. vernix — one taxon; a wetland species, fewer obs)
for M in "3,4,5" "6,7,8" "9,10" "11,12,1,2"; do
  $PY scripts/fetch_inaturalist.py --class poison_sumac --per-taxon 240 --months "$M"
done

# safe_plants — look-alike HARD NEGATIVES (6 taxa; ~130 each ≈ 780). No season
# stratification needed, but keep it US-wide for variety.
$PY scripts/fetch_inaturalist.py --class safe_plants --per-taxon 140
```

Useful flags: `--license cc0,cc-by,cc-by-nc` (more data, NonCommercial),
`--top-up-new N` (fetch N NEW each run instead of a total target),
`--place-id <id>` (region), `--dry-run`, `--max-pages` (raise if a season is
sparse and under-delivers).

> **Note:** iNat now filters on the **photo** license only (the redundant
> observation-license filter was removed — it wrongly dropped ~10% of usable
> CC photos). Held-out matches and failed downloads are remembered so resumed
> runs don't re-fetch them.

### 1b. GBIF — top-up + reproducibility, always dedup against held-out

`--limit` is NEW images per class per run. Always pass `--dedupe-against
TrainingData/Testing`. Use `--year-from/--year-to` if you want to bias toward
recent (higher-res phone) photos.

```bash
$PY scripts/fetch_gbif.py --classes poison_ivy poison_oak poison_sumac \
    --limit 250 --dedupe-against TrainingData/Testing

$PY scripts/fetch_gbif.py --classes safe_plants \
    --limit 200 --dedupe-against TrainingData/Testing
```

GBIF normalizes iNat media to full-res `/original.jpg`. Disallowed per-media
licenses are rejected; undecodable downloads are dropped (no null-phash rows).

### Fetch checkpoint

```bash
find data_staging -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
for c in poison_ivy poison_oak poison_sumac safe_plants; do
  n=$(find data_staging -type d -name "$c" -exec find {} -type f \; 2>/dev/null | wc -l)
  echo "  staged $c: $n"
done
```

---

## Step 2 — QA / dedup / leakage-guard (DRY-RUN first)

`dataset_qa.py` runs quality filters (corrupt, min-side, aspect, greyscale),
within-batch dedup (exact SHA + perceptual dHash/aHash), cross-class conflict
drops, and the **leakage guard** (drops any staged image within Hamming ≤ 5 of
ANY image in `TrainingData/Testing/**` OR the existing train pool, across all
classes). **Default is dry-run** — it changes nothing.

Run it once per staging tree (iNat and GBIF land under different subdirs; QA
expects `<staged>/<class>/` layout, so point it at each source root):

```bash
# iNat
$PY scripts/dataset_qa.py --staged data_staging/inaturalist --tag inat
# GBIF
$PY scripts/dataset_qa.py --staged data_staging/gbif --tag gbif
```

Read the report. Expect the **leak_heldout** count to be **0** (fetchers already
guard); if it's nonzero, that is a leakage alarm — investigate before committing.
A 25–40% total drop rate (near-dups + pool overlap) is normal.

Tunables (defaults are contract-correct): `--min-side 128`, `--min-aspect 0.4`,
`--max-aspect 2.5`, `--sat-thresh 0.04`, `--dup-hamming 5`, `--leak-hamming 5`,
`--keep-crossclass`, `--json <report.json>`.

---

## Step 3 — Commit into `TrainingData/`

Re-run each with `--commit`. Survivors copy into `TrainingData/<class>/` as
`qa_<tag>_<class>_NNNN.ext`; **Testing is never written**. Commit is idempotent
(already-committed images re-match the pool as `leak_pool` and are skipped).

```bash
$PY scripts/dataset_qa.py --staged data_staging/inaturalist --tag inat --commit
$PY scripts/dataset_qa.py --staged data_staging/gbif --tag gbif --commit
```

Each commit:
- carries **license + attribution + source URL + observation id** from the
  staging `manifest.jsonl` into `qa_commit_manifest.json` (the earlier
  `.jsonl`-vs-`.json` mismatch that dropped all provenance is fixed);
- appends **`TrainingData/manifest.jsonl`** rows mapping each committed file to
  its source **observation id** — this is what makes the train/val split
  **observation-disjoint** for fetched data (so near-duplicate frames of the
  *same physical plant* cannot straddle the split and inflate val).

Watch the commit output: if it prints `no observation ids available`, provenance
did not carry through — do not proceed, the split will silently fall back to
filename-hash sharding.

### Commit checkpoint

```bash
for c in poison_ivy poison_oak poison_sumac safe_plants; do
  n=$(find "TrainingData/$c" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
  echo "  train pool $c: $n"
done
test -f TrainingData/manifest.jsonl && echo "provenance rows: $(wc -l < TrainingData/manifest.jsonl)"
# Held-out MUST be unchanged:
echo "held-out (must be 362): $(find TrainingData/Testing -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
```

---

## Step 4 — Retrain (`train_v5.py`) — HEAVY, run on GPU/MPS

Direct (no-distillation) EfficientNet-B0 + light head, inverse-frequency class
weights + label smoothing, and the **field-failure augmentation** that targets
the 58% motion-blur cliff (directional motion blur, defocus/Gaussian blur,
random-erasing occlusion). The split is source/observation-disjoint by
construction with a hard overlap assertion; the frozen held-out set is never read.

```bash
$PY scripts/train_v5.py --epochs 40
# options: --head bottleneck (try the small 1280->256->4 head) | --no-export
```

Outputs `checkpoints/plant_detector_v5.pth` and, unless `--no-export`, the Core
ML model at `LeafAlert/Resources/MLModels/PlantDetector.mlpackage` (normalize +
softmax baked). Model selection is on **val toxic-recall** (the safety metric),
tie-broken by val accuracy.

---

## Step 5 — Re-export Core ML (if you trained with `--no-export`)

The trainer exports through the shared `scripts/coreml_export.py` (bakes
per-channel normalize + softmax; takes raw 0–255 RGB), preserving device parity.
If you skipped export, re-derive it from the checkpoint:

```bash
$PY scripts/reexport_coreml.py     # or scripts/coreml_export.py — bakes normalize+softmax
```

Verify the exported model has the right IO contract (inputs `['image']`, outputs
class-label + probs).

---

## Step 6 — Evaluate against the FROZEN held-out set + acceptance floor

Evaluate the **Core ML** model (architecture-agnostic and self-contained — this
is the correct path for the v5 model). Compare against the same-threshold
baseline.

```bash
$PY scripts/evaluate_model.py \
    --coreml LeafAlert/Resources/MLModels/PlantDetector.mlpackage \
    --data TrainingData/Testing \
    --split held-out \
    --threshold 0.65 \
    --sweep 0.3,0.4,0.5,0.6,0.65,0.7,0.8,0.9 \
    --json /tmp/leafalert_eval_v5.json
```

The report prints, per threshold: **`toxic_recall`**, `toxic_miss_total`,
**`toxic_to_safe_hard`** (toxic argmax'd as safe_plants — the dangerous miss),
`toxic_below_thr` (abstain), and `safe_false_alarm`, plus a per-class
precision/recall table and confusion matrix.

### Acceptance floor — a release GATE, not a suggestion

At the shipping threshold (**0.65**), on `TrainingData/Testing`, versus the
current production model measured on the **same** held-out set and threshold:

| Metric | Baseline (v-prev) | Requirement | Stretch |
| --- | --- | --- | --- |
| **held-out toxic-recall** | 0.435 @ 0.65 | **must IMPROVE, never regress** (≥ 0.435; target **≥ 0.75**) | ≥ 0.85 |
| **toxic→safe hard miss** (`toxic_to_safe_hard`) | high | **must DROP, never rise** (target **≤ 0.15**) | ≤ 0.08 |
| **poison_ivy recall** (per-class table) | 0.51 | must not regress; target **≥ 0.70** | ≥ 0.80 |
| **safe→toxic false-alarm** | — | should not blow up (soft cap **≤ 0.20**) | ≤ 0.10 |

**Hard rule:** if toxic-recall regresses OR toxic→safe hard-miss rises versus the
prior model on this frozen set, **do not ship** — a safety-critical poison
detector must never get worse at catching poison. Re-check the confusion matrix
to see whether false-alarm was traded for recall.

### Robustness re-check (the 58% motion-blur cliff)

`robustness_report.py` and `calibration_report.py` are **torch-only** and
hardcode `distill_model.PlantDetectorNet` — they will **not** load the v5
checkpoint as-is. Two options:

1. **Preferred:** point them at `train_v5.PlantDetectorV5` (add a `--head` flag
   or state-dict auto-detect), then:
   ```bash
   $PY scripts/robustness_report.py --data-dir TrainingData/Testing --threshold 0.65
   $PY scripts/calibration_report.py --data-dir TrainingData/Testing
   ```
2. **Interim:** apply the augmentation corruptions manually to the held-out set
   and re-run `evaluate_model.py --coreml` on the corrupted copy, comparing
   `toxic_to_safe_hard` against the clean run.

Acceptance: **motion-blur toxic→safe flip-rate must drop well below the 58%
baseline** (target ≤ 20%). This is the specific failure the v5 augmentation was
built to fix; verify it moved.

### After acceptance

Re-derive `LeafAlert/Models/ToxicityThresholds.swift` from the new
model's calibration/sweep (thresholds are model-specific), and record the new
baseline numbers so the next iteration's "never regress" gate has a reference.

---

## Rollback / safety

- The frozen held-out set is structurally excluded everywhere — if any checkpoint
  shows the held-out count ≠ 362, stop and investigate (something wrote to
  `Testing/`).
- `git status` should show only new files under `data_staging/`,
  `TrainingData/<class>/`, `TrainingData/manifest.jsonl`, `checkpoints/`, and the
  re-exported `.mlpackage`.
- If acceptance fails, the prior `.mlpackage` in git history is the rollback
  target — do not overwrite it in a commit until the new model passes the gate.

## Oversized originals (added 2026-09-02)

`dataset_qa.py` no longer admits huge originals verbatim. Anything above
`--max-megapixels` (default 20) is **downsized on commit** to a long edge of
`--downsize-long-edge` px (default 2048; EXIF orientation baked in, JPEG q95)
and the manifest records `downsized_to_long_edge`. Nothing is dropped for
size. Motivation: a 101.8 MP and a 43 MP GBIF scan were in the pool, decoded in
full on every epoch for a 224-px input. Those two existing pool files were left
untouched (the pool is not mutated mid-experiment); re-commit them through QA
if you want them downsized.
