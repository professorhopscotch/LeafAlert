#!/usr/bin/env python3
"""
Active-learning-aware ingest of app user feedback into a QA staging area.

This is the RICHER path alongside scripts/ingest_feedback.py (which stays as-is
and still copies straight into TrainingData_split/train/). The difference that
matters:

    ingest_feedback.py     manifest.json -> TrainingData_split/train/<cls>/
    ingest_feedback_v2.py  manifest.json -> data_staging/feedback/<cls>/
                                         -> scripts/dataset_qa.py  (QA + leakage
                                            guard) -> TrainingData/<cls>/

This script NEVER writes into TrainingData/. Everything it accepts is *staged*
and must still pass scripts/dataset_qa.py, which owns de-duplication and the
leakage guard against the frozen held-out set (TrainingData/Testing/). Feedback
images are the single most likely source of test-set leakage in this project:
a user can photograph the same plant that a held-out image came from, and a
user can submit the same photo twice from two sync paths (iCloud + LAN server).
Routing feedback through dataset_qa is what keeps held-out recall honest.


WHY THIS IS NOT JUST "COPY THE USER'S LABEL"
--------------------------------------------
User feedback is noisy, and in a poison-plant app the noise is ASYMMETRIC:

  * A wrong "this is actually safe" label teaches the model to MISS a toxic
    plant. Cost: someone touches poison ivy.
  * A wrong "this is actually poison ivy" label teaches the model to over-alert
    on a harmless plant. Cost: an annoying false alarm.

Those are not the same mistake, so this script does not treat them the same.
The dangerous direction is a *toxic -> safe demotion*: the model said
poison_ivy/oak/sumac and the user says safe_plants. There is also a strong
incentive for that exact label to be wrong -- a user who is tired of an alert
can dismiss it as "safe" without looking closely, which is precisely the input
that would train the miss back in.

So: a toxic -> safe demotion that contradicts a CONFIDENT model prediction is
never trained on silently. It goes to a needs_review bucket for a human. See
classify_contradiction() for the full direction taxonomy.

The reverse direction (model said safe_plants, user says poison_ivy) is the
*valuable* one: it is exactly the measured weakness of the shipped v6 model
(poison_ivy -> safe_plants 18%, poison_oak -> safe_plants 8% on held-out).
Those corrections are accepted and given the highest active-learning priority.
Flagging them as "contradicts the model" and dropping them would throw away the
best training signal the feedback stream produces.

Cross-toxic corrections (ivy <-> oak <-> sumac) are accepted with low priority:
that confusion is large but operationally harmless, since both still alert.


ACTIVE LEARNING
---------------
Every entry gets a `priority` score (see score_priority) so scarce human review
and labeling effort goes to the most informative images first, rather than to
whatever happened to be uploaded most recently. Roughly:

  confidently WRONG   > uncertain            > confidently right (redundant)
  correction          > confirmation
  targets a known safety weakness            > operationally harmless confusion

needs_review rows are written out sorted by priority, so the top of the review
queue is the highest-value work.


USAGE
-----
    # dry run (default): report only, write nothing
    python3 scripts/ingest_feedback_v2.py --feedback-dir feedback

    # actually stage accepted images for QA
    python3 scripts/ingest_feedback_v2.py --feedback-dir feedback --commit-staging

    # then hand off to the QA + leakage guard (this is what enters the pool)
    python3 scripts/dataset_qa.py --staged data_staging/feedback --tag feedback
    python3 scripts/dataset_qa.py --staged data_staging/feedback --tag feedback --commit

    # stricter label trust
    python3 scripts/ingest_feedback_v2.py --feedback-dir feedback \
        --trust-policy corrections-only --commit-staging
    python3 scripts/ingest_feedback_v2.py --feedback-dir feedback \
        --review-all-demotions --commit-staging

Idempotent: re-running rewrites the staging manifest in place and skips images
already staged. Exit codes: 0 success, 1 bad arguments / unreadable input.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fixed, ImageFolder-alphabetical. Do not reorder.
CANONICAL_CLASSES = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]
TOXIC_CLASSES = {"poison_ivy", "poison_oak", "poison_sumac"}
SAFE_CLASS = "safe_plants"

# The app's "do not alert" bucket is the same class as the safe-plant class,
# and now also holds non-plant negatives (v6 is non-plant-aware).
NOT_A_PLANT = "not_a_plant"

DEFAULT_FEEDBACK_DIR = PROJECT_ROOT / "feedback"
DEFAULT_STAGING_DIR = PROJECT_ROOT / "data_staging" / "feedback"

ICLOUD_SEARCH_PATHS = [
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "LeafAlert" / "feedback",
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "feedback",
]

# Statuses that mean "user threw this away" -> never training data.
SKIP_STATUSES = {"discarded", "none", ""}

PROVENANCE_SOURCE = "user_feedback"
# User-submitted photos: the app user holds the copyright. Recorded honestly so
# generate_attribution.py / NOTICE never implies an open license we don't have.
PROVENANCE_LICENSE = "user-submitted"
PROVENANCE_ATTRIBUTION = "LeafAlert app user feedback (in-app capture)"


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_confidence(raw):
    """
    Confidence is a softmax probability in [0,1]. Be defensive anyway: if a
    build ever emits 0-100, a naive '>= 0.70' test would call EVERY entry
    high-confidence and silently disable the contradiction gate. That failure
    would be invisible and safety-relevant, so clamp explicitly.

    Returns a float in [0,1], or None if unparseable.
    """
    if raw is None:
        return None
    try:
        c = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(c) or math.isinf(c):
        return None
    if c > 1.0:
        c = c / 100.0
    return max(0.0, min(1.0, c))


def parse_timestamp(raw):
    """ISO8601 (the app writes .withInternetDateTime) -> aware datetime, or None."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def path_is_inside(candidate, root):
    """True iff `candidate` resolves to something inside `root`.

    Guards the untrusted manifest's `filename` field against path traversal
    ('../..'), absolute paths, and symlinks pointing out of the feedback folder.
    """
    try:
        return Path(candidate).resolve().is_relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False


def sanitize_stem(name):
    """Filesystem-safe stem. App filenames embed an ISO timestamp with '-' and '+'."""
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem) or "unnamed"


def observation_group_id(entry, filename, geo_precision, time_bucket_min):
    """
    Group id for observation-disjoint train/val splitting (train_v5.py reads
    `observation_id` out of TrainingData/manifest.jsonl).

    Two photos of the SAME physical plant must not straddle the train/val split
    or val accuracy is inflated. A user who alerts on one plant typically fires
    several frames from the same spot seconds apart, so we group by rounded
    location + a coarse time bucket. ~4 decimal places of latitude is ~11 m.

    With no usable location we fall back to a per-file unique id, which is the
    conservative choice: it never *merges* two genuinely different plants, it
    only fails to merge duplicates of one (and dataset_qa's near-dup pass
    catches the worst of that anyway).
    """
    lat = entry.get("latitude")
    lng = entry.get("longitude")
    try:
        lat = float(lat)
        lng = float(lng)
        usable = (
            not math.isnan(lat) and not math.isnan(lng)
            and not math.isinf(lat) and not math.isinf(lng)
            # exact 0,0 is the app's "no fix" sentinel, not Null Island
            and (abs(lat) > 1e-9 or abs(lng) > 1e-9)
            and -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0
        )
    except (TypeError, ValueError):
        usable = False

    dt = parse_timestamp(entry.get("timestamp"))
    if usable and dt is not None:
        bucket = int(dt.timestamp() // (max(1, time_bucket_min) * 60))
        key = f"{round(lat, geo_precision)},{round(lng, geo_precision)},{bucket}"
        return "fb_obs_" + _sha1(key)[:12]
    return "fb_single_" + _sha1(filename)[:12]


def verify_image(path):
    """
    Cheap corrupt-file check so a broken JPEG is reported here with a clear
    reason instead of dying later. dataset_qa still does the real quality pass
    (resolution, aspect, saturation) -- this is not a substitute for it.

    Returns (ok: bool, detail: str).
    """
    try:
        from PIL import Image
    except ImportError:
        return True, "pil_unavailable_skipped"
    try:
        with Image.open(path) as im:
            im.verify()
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Label resolution
# ─────────────────────────────────────────────────────────────────────────────
def resolve_label(entry):
    """
    Map a manifest row to a training label.

        confirmed -> originalPrediction
        corrected -> correctedLabel
        discarded -> skip

    'not_a_plant' -> safe_plants: safe_plants is both the safe-plant class and
    the app's do-not-alert bucket, and v6 is deliberately non-plant-aware, so a
    non-plant negative belongs there.

    Returns (label, status, error) -- exactly one of label/error is non-None.
    """
    status = str(entry.get("feedbackStatus", "") or "").strip().lower()
    original = str(entry.get("originalPrediction", "") or "").strip()
    corrected = str(entry.get("correctedLabel", "") or "").strip()

    # Fail closed on anything the app marks as synthetic (DEBUG-injected
    # detections). The app no longer exports these at all; this guards older
    # builds and hand-edited manifests. Skipped, never an error.
    if entry.get("synthetic") is True or status.startswith("synthetic"):
        return None, "synthetic", None

    if status in SKIP_STATUSES:
        return None, status, None  # skipped, not an error

    if status == "confirmed":
        label = original
    elif status == "corrected":
        # A "correction" that names nothing, or names the original label, is
        # really a confirmation. Treat it as one rather than dropping it.
        label = corrected or original
    else:
        return None, status, f"unknown feedbackStatus '{status}'"

    if label == NOT_A_PLANT:
        label = SAFE_CLASS

    if not label:
        return None, status, "no label on entry"
    if label not in CANONICAL_CLASSES:
        return None, status, f"non-trainable label '{label}'"

    return label, status, None


def safety_of(label):
    return "toxic" if label in TOXIC_CLASSES else "safe"


def classify_contradiction(original, label, status):
    """
    Direction taxonomy for a correction. Returns one of:

      'none'            confirmed, or the user agreed with the model
      'toxic_to_safe'   model said toxic, user says safe   <-- DANGEROUS
      'safe_to_toxic'   model said safe, user says toxic   <-- VALUABLE
      'toxic_to_toxic'  ivy <-> oak <-> sumac              <-- harmless
      'unknown_origin'  we cannot tell (no/!canonical originalPrediction)

    Only 'toxic_to_safe' can teach the model to miss a toxic plant, which is
    why it is the only direction gated on confidence below.
    """
    if status != "corrected":
        return "none"
    if original not in CANONICAL_CLASSES:
        # Cannot reason about direction. Treat as unknown, not as safe.
        return "unknown_origin"
    if original == label:
        return "none"

    was, now = safety_of(original), safety_of(label)
    if was == "toxic" and now == "safe":
        return "toxic_to_safe"
    if was == "safe" and now == "toxic":
        return "safe_to_toxic"
    return "toxic_to_toxic"


def score_priority(status, direction, conf, label):
    """
    Active-learning value. Higher = review/train on this sooner.

    The shape encodes three ideas:
      1. A correction carries error signal; a confirmation mostly re-teaches
         what the model already knows (and, unchecked, entrenches its bias).
      2. Confidently WRONG is the most informative thing that can happen. For a
         confirmation the informative case is the opposite: right but unsure,
         i.e. sitting on the decision boundary.
      3. Errors on the measured safety weakness (poison_ivy/oak predicted as
         safe_plants: 18% / 8% held-out) are worth more than the large but
         operationally harmless ivy<->oak confusion.
    """
    c = 0.5 if conf is None else conf

    if status == "corrected":
        score = 1.0 + c          # confidently wrong -> top of the queue
    else:
        score = 0.3 + (1.0 - c)  # right but unsure -> boundary case

    if direction == "safe_to_toxic":
        score += 1.0             # directly attacks the dangerous miss
        if label in ("poison_ivy", "poison_oak"):
            score += 0.25        # the two measured weak spots
    elif direction == "toxic_to_safe":
        score += 0.5             # false-alert reduction: valuable, but review it
    elif direction == "unknown_origin":
        score += 0.25

    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Trust policy
# ─────────────────────────────────────────────────────────────────────────────
def apply_trust_policy(policy, status, conf, min_conf):
    """
    Decide whether an otherwise-valid entry is a training candidate.

      all              accept confirmed + corrected.
      corrections-only accept only 'corrected'. Confirmations are the bulk of
                       the stream and mostly duplicate what the model already
                       gets right; training on them re-weights toward easy
                       examples and can entrench existing bias.
      high-confidence  require model confidence >= --min-confidence. For a
                       confirmation that means model and user agree AND the
                       model was sure (strongest possible label). For a
                       correction it means the model was confidently wrong,
                       which is the highest-information case. Low-confidence
                       rows are dropped as too ambiguous to trust.

    This is orthogonal to the contradiction gate below -- needs_review applies
    under every policy.

    Returns (accepted: bool, reason: str|None).
    """
    if policy == "all":
        return True, None

    if policy == "corrections-only":
        if status != "corrected":
            return False, "policy:corrections-only (confirmation dropped)"
        return True, None

    if policy == "high-confidence":
        if conf is None:
            return False, "policy:high-confidence (no confidence recorded)"
        if conf < min_conf:
            return False, f"policy:high-confidence (conf {conf:.2f} < {min_conf:.2f})"
        return True, None

    return True, None


def contradiction_gate(direction, label, conf, contradiction_conf,
                       review_all_demotions):
    """
    The safety gate. Returns (needs_review: bool, reason: str|None).

    A toxic -> safe demotion is the one label error that can get someone hurt,
    and it is also the one a careless user is most likely to produce (dismissing
    an alert is easier than reporting one). When it contradicts a CONFIDENT
    model prediction, the disagreement is either a real model false positive
    worth fixing or a bad label worth catching -- and we cannot tell which from
    here. So it goes to a human instead of into the training pool.

    FAIL CLOSED on an unverifiable direction. If `originalPrediction` is missing
    or is not one of the canonical class names, classify_contradiction() cannot
    tell a toxic->safe demotion from a harmless relabel -- and the manifest is
    untrusted input (scripts/feedback_server.py accepts it over the LAN, and the
    app's field format can change between builds). A correction whose *result*
    is safe_plants is therefore reviewed whenever the direction cannot be
    verified, regardless of confidence: with no known prediction, `confidence`
    is not a number we can reason about either. Otherwise a single non-canonical
    originalPrediction string turns every dangerous demotion into a silent
    accept, which is precisely the failure this gate exists to prevent.

    Deliberately NOT gated: safe -> toxic. That contradicts the model too, but
    the error direction is safe, and it is exactly the failure this model has
    (18% of held-out poison_ivy is called safe_plants). Sending those to a
    review queue would starve the fix.
    """
    if direction == "unknown_origin" and label == SAFE_CLASS:
        return True, ("correction to safe_plants with a missing/non-canonical "
                      "originalPrediction — direction unverifiable, cannot rule "
                      "out a toxic->safe demotion")

    if direction != "toxic_to_safe":
        return False, None

    if review_all_demotions:
        return True, "toxic->safe demotion (--review-all-demotions)"

    if conf is None:
        return True, "toxic->safe demotion with no confidence recorded"

    if conf >= contradiction_conf:
        return True, (
            f"toxic->safe demotion contradicting confident model "
            f"(conf {conf:.2f} >= {contradiction_conf:.2f})"
        )
    return False, None


# ─────────────────────────────────────────────────────────────────────────────
# Manifest IO
# ─────────────────────────────────────────────────────────────────────────────
def load_feedback_manifest(feedback_dir):
    """Read the app/server manifest.json -> list of entry dicts."""
    manifest_path = feedback_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest.json at {manifest_path}", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot parse {manifest_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("entries", [])
    else:
        entries = []

    if not isinstance(entries, list):
        print(f"ERROR: 'entries' in {manifest_path} is not a list", file=sys.stderr)
        sys.exit(1)
    return [e for e in entries if isinstance(e, dict)]


def load_existing_jsonl(path):
    """Existing staging manifest -> {staged_basename: row}. Absent -> {}."""
    out = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            bn = row.get("filename") or (row.get("path") or "").rsplit("/", 1)[-1]
            if bn:
                out[bn] = row
    except OSError:
        pass
    return out


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────
def process(entries, feedback_dir, args):
    """
    Evaluate every manifest row. Pure decision pass -- no files are written
    here, so --dry-run and --commit-staging make identical decisions.
    """
    accepted, needs_review, skipped, unresolvable = [], [], [], []
    seen_filenames = set()

    for idx, entry in enumerate(entries):
        filename = str(entry.get("filename", "") or "").strip()
        original = str(entry.get("originalPrediction", "") or "").strip()
        conf = normalize_confidence(entry.get("confidence"))

        if not filename:
            unresolvable.append({"row": idx, "filename": "<missing>",
                                 "reason": "manifest row has no filename"})
            continue

        if filename in seen_filenames:
            skipped.append({"filename": filename, "reason": "duplicate manifest row"})
            continue
        seen_filenames.add(filename)

        label, status, err = resolve_label(entry)

        if err is not None:
            unresolvable.append({"row": idx, "filename": filename, "reason": err})
            continue

        if label is None:
            skipped.append({"filename": filename, "reason": f"feedbackStatus '{status}'"})
            continue

        src = feedback_dir / filename
        # The manifest is UNTRUSTED input (feedback_server.py accepts uploads over
        # the LAN). A filename like '../../TrainingData/Testing/<cls>/x.jpg' or an
        # absolute path would otherwise pull an arbitrary file — including a FROZEN
        # held-out image — into the staging area. Refuse anything that does not
        # resolve inside the feedback dir.
        if not path_is_inside(src, feedback_dir):
            unresolvable.append({"row": idx, "filename": filename,
                                 "reason": "filename escapes the feedback dir "
                                           "(path traversal or absolute path)"})
            continue

        if not src.is_file():
            unresolvable.append({"row": idx, "filename": filename,
                                 "reason": "image file missing from feedback dir"})
            continue

        ok, detail = verify_image(src)
        if not ok:
            unresolvable.append({"row": idx, "filename": filename,
                                 "reason": f"unreadable image ({detail})"})
            continue

        direction = classify_contradiction(original, label, status)
        priority = score_priority(status, direction, conf, label)

        record = {
            "src": src,
            "filename": filename,
            "label": label,
            "status": status,
            "original": original,
            "confidence": conf,
            "direction": direction,
            "priority": priority,
            "timestamp": entry.get("timestamp"),
            "latitude": entry.get("latitude"),
            "longitude": entry.get("longitude"),
            "observation_id": observation_group_id(
                entry, filename, args.geo_precision, args.time_bucket_min
            ),
        }

        # Safety gate first: it must not be bypassable by a trust policy.
        review, reason = contradiction_gate(
            direction, label, conf, args.contradiction_conf,
            args.review_all_demotions
        )
        if review:
            record["review_reason"] = reason
            needs_review.append(record)
            continue

        ok_policy, why = apply_trust_policy(
            args.trust_policy, status, conf, args.min_confidence
        )
        if not ok_policy:
            skipped.append({"filename": filename, "reason": why})
            continue

        accepted.append(record)

    needs_review.sort(key=lambda r: -r["priority"])
    accepted.sort(key=lambda r: -r["priority"])
    return accepted, needs_review, skipped, unresolvable


def stage_records(records, root, dry_run, existing_rows, conflicts, include_geo):
    """
    Copy records into root/<class>/ and build provenance rows in the exact shape
    scripts/dataset_qa.py:load_staging_manifest() reads (it keys on basename and
    pulls class/source/license/attribution/url/observation_id).

    Idempotent: an image already staged under the same class is left alone. An
    image already staged under a DIFFERENT class is a real label conflict --
    reported, never silently moved, because silently relabeling a staged image
    is exactly the kind of quiet mutation this script exists to prevent.
    """
    rows, staged_new, already = [], 0, 0
    # sanitize_stem() is many-to-one ('a-b.jpg' and 'a+b.jpg' both -> 'a_b'), so two
    # DIFFERENT source images can want the same staged name. Left unchecked the
    # second one is silently not copied while its manifest row overwrites the
    # first's — the staged image would then carry the other image's provenance
    # (observation_id, timestamp, source filename). Report it instead.
    seen_staged = {}

    for rec in records:
        cls = rec["label"]
        staged_name = f"fb_{sanitize_stem(rec['filename'])}.jpg"
        dest = root / cls / staged_name

        clash = seen_staged.get(staged_name)
        if clash is not None and clash != rec["filename"]:
            conflicts.append({
                "filename": rec["filename"],
                "reason": (f"staged name '{staged_name}' already claimed this run by "
                           f"'{clash}' — distinct images, same sanitized name"),
            })
            continue
        seen_staged[staged_name] = rec["filename"]

        prior = existing_rows.get(staged_name)
        if prior and prior.get("class") and prior["class"] != cls:
            conflicts.append({
                "filename": rec["filename"],
                "reason": (f"already staged as '{prior['class']}', "
                           f"manifest now says '{cls}'"),
            })
            continue

        other = [c for c in CANONICAL_CLASSES
                 if c != cls and (root / c / staged_name).is_file()]
        if other:
            conflicts.append({
                "filename": rec["filename"],
                "reason": f"file already staged under '{other[0]}', now labeled '{cls}'",
            })
            continue

        row = {
            # dataset_qa resolves basename from path/file/filename; give both.
            "path": f"{cls}/{staged_name}",
            "filename": staged_name,
            "class": cls,
            "source": PROVENANCE_SOURCE,
            "license": PROVENANCE_LICENSE,
            "attribution": PROVENANCE_ATTRIBUTION,
            "url": None,
            "observation_id": rec["observation_id"],
            # Feedback-specific provenance, carried for auditability.
            "source_filename": rec["filename"],
            "feedback_status": rec["status"],
            "original_prediction": rec["original"],
            "corrected_label": rec["label"],
            "model_confidence": rec["confidence"],
            "contradiction_direction": rec["direction"],
            "priority": rec["priority"],
            "timestamp": rec["timestamp"],
        }
        if include_geo:
            row["latitude"] = rec["latitude"]
            row["longitude"] = rec["longitude"]
        rows.append(row)

        if dest.is_file():
            already += 1
            continue

        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rec["src"], dest)
        staged_new += 1

    return rows, staged_new, already


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def bar(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def report(accepted, needs_review, skipped, unresolvable, conflicts, args,
           staged_new, already, staging_root, review_root):
    bar("ACCEPTED (staged for dataset_qa)")
    if accepted:
        per_class = {}
        for r in accepted:
            per_class.setdefault(r["label"], []).append(r)
        for cls in CANONICAL_CLASSES:
            group = per_class.get(cls, [])
            if not group:
                continue
            tag = "[TOXIC]" if cls in TOXIC_CLASSES else "[safe/neg]"
            print(f"  {tag:11s} {cls:14s} {len(group):4d}")
            for r in group[:args.show]:
                conf = "n/a" if r["confidence"] is None else f"{r['confidence']:.2f}"
                print(f"      prio {r['priority']:5.2f}  {r['status']:9s} "
                      f"{r['original'] or '?':>13s} -> {r['label']:<13s} "
                      f"conf {conf}  {r['filename']}")
            if len(group) > args.show:
                print(f"      ... {len(group) - args.show} more")
    else:
        print("  (none)")
    print(f"\n  total accepted: {len(accepted)}")

    bar("NEEDS REVIEW (NOT staged for training)")
    if needs_review:
        print("  Highest active-learning priority first. Each of these would, if\n"
              "  trained on blindly, risk teaching the model that a toxic plant is\n"
              "  safe. A human must confirm the label before it enters the pool.\n")
        for r in needs_review:
            conf = "n/a" if r["confidence"] is None else f"{r['confidence']:.2f}"
            print(f"  prio {r['priority']:5.2f}  {r['filename']}")
            print(f"      {r['original'] or '?'} (conf {conf}) -> {r['label']}")
            print(f"      reason: {r['review_reason']}")
    else:
        print("  (none)")
    print(f"\n  total needs_review: {len(needs_review)}")

    bar("SKIPPED")
    if skipped:
        reasons = {}
        for s in skipped:
            reasons.setdefault(s["reason"], []).append(s["filename"])
        for reason, files in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(files):4d}  {reason}")
            for f in files[:args.show]:
                print(f"          {f}")
            if len(files) > args.show:
                print(f"          ... {len(files) - args.show} more")
    else:
        print("  (none)")
    print(f"\n  total skipped: {len(skipped)}")

    bar("UNRESOLVABLE")
    if unresolvable:
        for u in unresolvable:
            print(f"  row {u['row']:4d}  {u['filename']}")
            print(f"           {u['reason']}")
    else:
        print("  (none)")
    print(f"\n  total unresolvable: {len(unresolvable)}")

    if conflicts:
        bar("STAGING CONFLICTS (not staged)")
        for c in conflicts:
            print(f"  {c['filename']}\n      {c['reason']}")
        print(f"\n  total conflicts: {len(conflicts)}")

    bar("SUMMARY")
    total = len(accepted) + len(needs_review) + len(skipped) + len(unresolvable)
    print(f"  manifest rows evaluated : {total}")
    print(f"  accepted                : {len(accepted)}")
    print(f"  needs_review            : {len(needs_review)}")
    print(f"  skipped                 : {len(skipped)}")
    print(f"  unresolvable            : {len(unresolvable)}")
    print(f"  staging conflicts       : {len(conflicts)}")
    print(f"  trust policy            : {args.trust_policy}")
    print(f"  contradiction threshold : {args.contradiction_conf:.2f}"
          f"{'  (--review-all-demotions)' if args.review_all_demotions else ''}")

    if args.dry_run:
        print(f"\n  DRY RUN -- nothing written.")
        print(f"  Would stage {staged_new} new image(s) into {staging_root}")
        if already:
            print(f"  ({already} already staged)")
        if needs_review:
            print(f"  Would write {len(needs_review)} review row(s) into {review_root}")
        print("\n  Re-run with --commit-staging to write.")
    else:
        print(f"\n  Staged {staged_new} new image(s) into {staging_root}")
        if already:
            print(f"  ({already} already staged, left untouched)")
        if needs_review:
            print(f"  Wrote {len(needs_review)} review row(s) into {review_root}")
        print("\n  NOTHING has entered TrainingData yet. Next, run the QA + leakage guard:")
        print(f"    python3 scripts/dataset_qa.py --staged {staging_root} --tag feedback")
        print(f"    python3 scripts/dataset_qa.py --staged {staging_root} --tag feedback --commit")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Active-learning-aware ingest of user feedback into a QA staging area.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Accepted images are STAGED only. scripts/dataset_qa.py is what "
               "commits them into TrainingData, with the leakage guard.",
    )
    ap.add_argument("--feedback-dir", type=Path, default=DEFAULT_FEEDBACK_DIR,
                    help="Feedback folder containing manifest.json + images.")
    ap.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR,
                    help="Staging root; images land in <staging-dir>/<class>/.")
    ap.add_argument("--review-dir", type=Path, default=None,
                    help="Needs-review output (default: <staging-dir>_needs_review). "
                         "MUST be outside --staging-dir so dataset_qa never sees it.")

    ap.add_argument("--trust-policy", choices=["all", "corrections-only", "high-confidence"],
                    default="all",
                    help="Which entries are training candidates. See module docstring.")
    ap.add_argument("--min-confidence", type=float, default=0.70,
                    help="Confidence floor for --trust-policy high-confidence.")
    ap.add_argument("--contradiction-conf", type=float, default=0.70,
                    help="A toxic->safe correction contradicting a prediction at or "
                         "above this confidence goes to needs_review.")
    ap.add_argument("--review-all-demotions", action="store_true",
                    help="Send EVERY toxic->safe correction to needs_review regardless "
                         "of model confidence (strictest, recommended before a release).")

    ap.add_argument("--geo-precision", type=int, default=4,
                    help="Decimal places of lat/lng used to group same-plant captures "
                         "into one observation_id (4 ~= 11 m).")
    ap.add_argument("--time-bucket-min", type=int, default=10,
                    help="Minutes per time bucket for observation grouping.")
    ap.add_argument("--no-geo", action="store_true",
                    help="Omit raw lat/lng from the staging manifest (the derived "
                         "observation_id is still written).")

    ap.add_argument("--commit-staging", action="store_true",
                    help="Actually write. Default is a dry run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Force dry run (wins over --commit-staging).")
    ap.add_argument("--show", type=int, default=8,
                    help="Max example rows printed per group.")
    ap.add_argument("--json", type=Path, default=None,
                    help="Also write the full decision report as JSON here.")
    args = ap.parse_args()

    # Dry run is the default and --dry-run always wins.
    args.dry_run = args.dry_run or not args.commit_staging

    if not 0.0 <= args.min_confidence <= 1.0:
        ap.error("--min-confidence must be in [0,1]")
    if not 0.0 <= args.contradiction_conf <= 1.0:
        ap.error("--contradiction-conf must be in [0,1]")

    feedback_dir = args.feedback_dir
    if not feedback_dir.is_dir() or not (feedback_dir / "manifest.json").exists():
        for p in ICLOUD_SEARCH_PATHS:
            if p.is_dir() and (p / "manifest.json").exists():
                print(f"Using iCloud feedback folder: {p}")
                feedback_dir = p
                break
        else:
            print(f"ERROR: no feedback manifest at {feedback_dir}", file=sys.stderr)
            print("Searched iCloud paths:", file=sys.stderr)
            for p in ICLOUD_SEARCH_PATHS:
                print(f"  {p}", file=sys.stderr)
            sys.exit(1)

    staging_root = args.staging_dir.resolve()
    review_root = (args.review_dir.resolve() if args.review_dir
                   else staging_root.parent / (staging_root.name + "_needs_review"))

    # The review bucket must not sit inside the staging root: dataset_qa walks
    # class-named subdirs of --staged, and un-reviewed images must never be
    # picked up by it.
    if review_root == staging_root or staging_root in review_root.parents:
        print(f"ERROR: --review-dir ({review_root}) is inside --staging-dir "
              f"({staging_root}); dataset_qa would ingest un-reviewed images.",
              file=sys.stderr)
        sys.exit(1)

    bar("LeafAlert — Active-Learning Feedback Ingest (v2)")
    print(f"  feedback dir : {feedback_dir}")
    print(f"  staging dir  : {staging_root}")
    print(f"  review dir   : {review_root}")
    print(f"  mode         : {'DRY RUN' if args.dry_run else 'COMMIT STAGING'}")

    entries = load_feedback_manifest(feedback_dir)
    print(f"  manifest rows: {len(entries)}")

    accepted, needs_review, skipped, unresolvable = process(entries, feedback_dir, args)

    manifest_path = staging_root / "manifest.jsonl"
    existing_rows = load_existing_jsonl(manifest_path)
    conflicts = []
    rows, staged_new, already = stage_records(
        accepted, staging_root, args.dry_run, existing_rows, conflicts,
        include_geo=not args.no_geo,
    )

    # A conflict means the record never staged; drop it from the accepted set so
    # the printed counts match what is actually on disk.
    if conflicts:
        bad = {c["filename"] for c in conflicts}
        accepted = [r for r in accepted if r["filename"] not in bad]

    if not args.dry_run:
        # Merge with any prior batch, keyed on staged basename -> idempotent.
        merged = dict(existing_rows)
        for row in rows:
            merged[row["filename"]] = row
        write_jsonl(manifest_path, [merged[k] for k in sorted(merged)])

        if needs_review:
            review_root.mkdir(parents=True, exist_ok=True)
            queue = []
            for r in needs_review:
                dest = review_root / f"fb_{sanitize_stem(r['filename'])}.jpg"
                if not dest.is_file():
                    shutil.copy2(r["src"], dest)
                queue.append({
                    "filename": dest.name,
                    "source_filename": r["filename"],
                    "proposed_label": r["label"],
                    "original_prediction": r["original"],
                    "model_confidence": r["confidence"],
                    "feedback_status": r["status"],
                    "contradiction_direction": r["direction"],
                    "review_reason": r["review_reason"],
                    "priority": r["priority"],
                    "timestamp": r["timestamp"],
                    "observation_id": r["observation_id"],
                    "resolution": "pending",
                })
            existing_q = load_existing_jsonl(review_root / "review_queue.jsonl")
            for q in queue:
                # Never clobber a resolution a human already recorded.
                prior = existing_q.get(q["filename"])
                if prior and prior.get("resolution", "pending") != "pending":
                    continue
                existing_q[q["filename"]] = q
            write_jsonl(
                review_root / "review_queue.jsonl",
                sorted(existing_q.values(), key=lambda r: -float(r.get("priority") or 0)),
            )

    report(accepted, needs_review, skipped, unresolvable, conflicts, args,
           staged_new, already, staging_root, review_root)

    if args.json:
        def clean(recs):
            return [{k: v for k, v in r.items() if k != "src"} for r in recs]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "feedback_dir": str(feedback_dir),
            "staging_dir": str(staging_root),
            "review_dir": str(review_root),
            "dry_run": args.dry_run,
            "trust_policy": args.trust_policy,
            "contradiction_conf": args.contradiction_conf,
            "review_all_demotions": args.review_all_demotions,
            "accepted": clean(accepted),
            "needs_review": clean(needs_review),
            "skipped": skipped,
            "unresolvable": unresolvable,
            "conflicts": conflicts,
        }, indent=2, sort_keys=True, default=str))
        print(f"\n  JSON report: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
