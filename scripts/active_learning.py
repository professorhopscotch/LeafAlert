#!/usr/bin/env python3
"""
ACTIVE-LEARNING SELECTOR — rank unlabeled images by how much a human label is worth.

The app produces far more images than anyone will ever label. This script scores an
UNLABELED pool with the shipped model and emits a ranked worklist, so the scarce
resource (a human deciding "ivy / oak / sumac / safe") is spent on the images that
will actually move the model.

    scored pool  ->  dedup + leakage guard  ->  uncertainty scoring  ->  ranked worklist
                                                                        (JSONL + text)

WHY NOT JUST "LOWEST CONFIDENCE"
--------------------------------
Classic active learning treats every uncertain image as equally valuable. LeafAlert
is a SAFETY app, and its errors are not symmetric:

  * poison_ivy -> safe_plants (18% on held-out) and poison_oak -> safe_plants (8%)
    are the DANGEROUS errors. A user gets no alert and walks into the plant.
  * ivy <-> oak confusion is large but operationally HARMLESS: the app's alert logic
    takes the top toxic class and compares it to that class's threshold, so an image
    the model calls 50/50 ivy-vs-oak still alerts. Labeling it buys almost nothing
    in the field.

A pure `margin` or `entropy` ranking cannot tell those apart — a 50/50 ivy/oak split
is the single most "uncertain" image possible and will sit at the top of the worklist
forever, burning labeling budget on a distinction the app does not act on. Hence the
`safety` strategy below.

STRATEGIES  (--strategy)
------------------------
All strategies produce a `score` in [0, 1] where HIGHER = label this one first, so
worklists are comparable across strategies. Each record also carries the raw
`strategy_value` in its native units.

  margin           (default)  strategy_value = p1 - p2 (top prob minus runner-up).
                   Smaller margin = the model is torn between two classes.
                   score = 1 - (p1 - p2).
                   Best general-purpose signal: it points at the decision boundary
                   rather than at high-entropy mush.

  least_confidence strategy_value = 1 - max(p). score = same.
                   Simple; ignores the shape of the rest of the distribution.

  entropy          strategy_value = H(p) in nats. score = H(p) / ln(K)  (K = 4).
                   Rewards mass spread over MANY classes; on a 4-way problem this
                   over-selects genuinely ambiguous foliage shots.

  safety           SAFETY-WEIGHTED (this app's recommended strategy for shipping).
                   Scores how close the image is to the decision the app ACTUALLY
                   makes, which is not "which of 4 classes" but "do I alert?":

                     boundary term = 2 * min(p_toxic, p_safe)
                        where p_toxic = p_ivy + p_oak + p_sumac and p_safe = p(safe).
                        Because the two sum to 1, this is 1.0 exactly at the
                        toxic-vs-safe coin flip and 0.0 when the call is settled.
                        This is the term that targets the measured toxic->safe
                        failure mode: a label here either creates a new hard toxic
                        example or a new hard negative.

                     threshold term = max(0, 1 - |p_top_toxic - thr(top_toxic)| / w)
                        thr is the app's per-class alert threshold (ivy/oak 0.40,
                        sumac 0.52 — mirrors ToxicityThresholds in
                        LeafAlert/Models/DetectionResult.swift), w = --safety-window
                        (default 0.20, the app's UNCERTAINTY_MARGIN). 1.0 when the
                        top toxic probability sits exactly on the threshold that
                        flips alert <-> no-alert, decaying to 0 a full window away.

                     score = wb*boundary + wt*threshold   (weights normalized to 1;
                                                           defaults 0.7 / 0.3)

                   The two strategies ORDER these two images oppositely, which is the
                   whole point (verified numerically, weights 0.7/0.3, window 0.20):

                     A = [.50, .50, 0, 0]   ivy-vs-oak tie; the app alerts either way
                         margin score 1.000  (ranked FIRST — maximally "uncertain")
                         safety score 0.150  (ranked last — nothing to gain)

                     B = [.45, .05, 0, .50]  toxic-vs-safe coin flip; the app IGNORES
                         it today (p_safe edges out p_ivy) so the user is NOT warned
                         margin score 0.950  (ranked second)
                         safety score 0.925  (ranked FIRST)

                   B is the measured poison_ivy -> safe_plants failure mode. Under
                   `margin` the budget goes to A first; under `safety` it goes to B.

DEDUP + LEAKAGE GUARD
---------------------
A candidate is DROPPED before ranking if it is a perceptual near-duplicate of:
  (a) the FROZEN held-out set TrainingData/Testing  -> labeling it and training on it
      would leak the test set and turn the safety metrics into fiction; or
  (b) the existing train pool TrainingData/<class>/ -> we already own that image, a
      human label adds nothing.
Optionally also near-duplicates WITHIN the candidate pool itself (default ON): a
burst of 30 frames of the same leaf is one label's worth of information, not 30. The
highest-scoring frame of each near-duplicate cluster survives.

The hashing (dhash + ahash, Hamming <= --leak-hamming on BOTH) and the match rule
itself are IMPORTED from scripts/dataset_qa.py — not reimplemented — so this
selector's notion of "duplicate" can never drift from the gate that images must later
pass to enter the train pool. Nothing selected here is a surprise rejection
downstream. dataset_qa.py remains the authoritative gate; this is a pre-filter so
labeling effort is not wasted.

Indexing the ~6.5k reference images is the slow part (measured 30s warm, 580s cold —
it is I/O bound on decoding every train-pool image). Hashes are therefore CACHED
outside the repo, keyed by absolute path + mtime + size, so only new or modified
reference images are re-hashed. Re-running with a different --strategy or --top is
then near-instant. `--no-cache` bypasses the cache and calls dataset_qa's
build_reference_index() verbatim, which is the reference implementation the cached
path is checked against.

PREPROCESSING PARITY
--------------------
Scoring uses train_v5.build_val_transforms() verbatim — Resize((224,224)) squash, NO
center crop, then per-channel ImageNet normalize. That is the same transform used for
validation and the same geometry as the on-device .scaleFill path. It is imported
rather than re-declared, and _assert_parity() re-checks the pipeline at runtime, so a
silent desync cannot make this worklist rank images the deployed model never sees the
same way.

OUTPUT
------
  --out worklist.jsonl   line 1: {"record_type": "meta", ...} run provenance
                         then one {"record_type": "candidate", ...} per selected image
                         with path, prediction, per-class probs, every uncertainty
                         measure, the strategy value, the app's alert severity, and a
                         suggested_label (= the model's prediction) so a human can
                         confirm with one keystroke instead of typing a class.
  --out-text worklist.txt  human-readable table (defaults to --out with a .txt suffix)
  stdout                 always: the counts + top-N class distribution, so the
                         operator can see selection collapsing onto one class.

USAGE
-----
    # rank a folder of unlabeled field photos, safety-weighted, top 50
    python3 scripts/active_learning.py --pool feedback --strategy safety --top 50 \
        --out worklist/round1.jsonl

    # default margin strategy, print only
    python3 scripts/active_learning.py --pool data_staging/unlabeled --top 20

Exit codes: 0 on success (even if 0 candidates survive), 1 on bad arguments or an
unusable pool/checkpoint.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

# Make sibling scripts importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_v5 import (                                    # noqa: E402
    PlantDetectorV5, CLASS_LABELS, IMAGE_SIZE, build_val_transforms,
)
from coreml_export import IMAGENET_MEAN, IMAGENET_STD     # noqa: E402
# Hashing / leakage guard: IMPORTED, never mirrored, so semantics cannot drift from
# the gate in dataset_qa.py that every image must pass to enter the train pool.
from dataset_qa import (                                  # noqa: E402
    dhash, ahash, hamming, build_reference_index, leak_match, list_images,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "student_v6_nonplant.pth"
DEFAULT_POOL_DIR = PROJECT_ROOT / "TrainingData"            # train pool (guard against)
DEFAULT_HELDOUT_DIR = PROJECT_ROOT / "TrainingData" / "Testing"  # FROZEN held-out
# Cache lives OUTSIDE the repo on purpose: it is a derived artifact, it would
# otherwise sit untracked in git status, and TrainingData/ must not be written to.
DEFAULT_REF_CACHE = Path.home() / ".cache" / "leafalert" / "active_learning_ref_index.json"
REF_CACHE_VERSION = 1

TOXIC_CLASSES = CLASS_LABELS[:3]           # poison_ivy, poison_oak, poison_sumac
SAFE_CLASS = "safe_plants"
SAFE_IDX = CLASS_LABELS.index(SAFE_CLASS)
TOXIC_IDX = [CLASS_LABELS.index(c) for c in TOXIC_CLASSES]

# Mirrors LeafAlert/Models/DetectionResult.swift -> ToxicityThresholds.
BASE_ALERT = {"poison_ivy": 0.40, "poison_oak": 0.40, "poison_sumac": 0.52}
UNCERTAINTY_MARGIN = 0.20

# The pool is UNLABELED user/field imagery: jpg/jpeg/png per the app's exporter.
POOL_EXTS = {".jpg", ".jpeg", ".png"}

STRATEGIES = ("margin", "entropy", "least_confidence", "safety")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing parity
# ─────────────────────────────────────────────────────────────────────────────
def _assert_parity(tf) -> None:
    """Fail loudly if the imported val transform ever stops matching the contract.

    The contract (train <-> device): squash-resize the WHOLE image to 224x224 with no
    crop, then per-channel ImageNet normalize. If someone adds a CenterCrop to
    train_v5's val pipeline, this selector would silently rank images on a view the
    deployed model never sees. Cheap check, catches an expensive class of bug.
    """
    steps = list(getattr(tf, "transforms", []))
    names = [type(s).__name__ for s in steps]

    for bad in ("CenterCrop", "RandomCrop", "RandomResizedCrop", "FiveCrop", "TenCrop"):
        if bad in names:
            raise SystemExit(
                f"PARITY VIOLATION: val transform contains {bad}; the contract is "
                f"squash-resize with NO crop. Pipeline: {names}"
            )

    resize = next((s for s in steps if isinstance(s, T.Resize)), None)
    if resize is None or tuple(np.atleast_1d(resize.size).tolist()) != (IMAGE_SIZE, IMAGE_SIZE):
        raise SystemExit(
            f"PARITY VIOLATION: expected Resize(({IMAGE_SIZE},{IMAGE_SIZE})) squash; "
            f"got {resize}. Pipeline: {names}"
        )

    norm = next((s for s in steps if isinstance(s, T.Normalize)), None)
    if norm is None or list(norm.mean) != list(IMAGENET_MEAN) or list(norm.std) != list(IMAGENET_STD):
        raise SystemExit(
            f"PARITY VIOLATION: expected ImageNet Normalize(mean={IMAGENET_MEAN}, "
            f"std={IMAGENET_STD}); got {norm}. Pipeline: {names}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt: Path, device: torch.device) -> PlantDetectorV5:
    if not ckpt.exists():
        raise SystemExit(f"ERROR: checkpoint not found: {ckpt}")
    model = PlantDetectorV5(num_classes=len(CLASS_LABELS), head="linear", pretrained=False)
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
    model.eval()
    model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty measures  (all take a length-K probability vector)
# ─────────────────────────────────────────────────────────────────────────────
def least_confidence(p: np.ndarray) -> float:
    """1 - max(p). Higher = the model's best guess is weaker."""
    return float(1.0 - p.max())


def margin(p: np.ndarray) -> float:
    """p1 - p2 (top minus runner-up). SMALLER = more uncertain / closer to a boundary."""
    s = np.sort(p)[::-1]
    return float(s[0] - s[1])


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats. Higher = mass spread over more classes."""
    q = np.clip(p, 1e-12, 1.0)
    return float(-(q * np.log(q)).sum())


def safety_score(p: np.ndarray, window: float, w_boundary: float, w_threshold: float):
    """SAFETY-WEIGHTED informativeness — see the module docstring for the rationale.

    Returns (score, boundary_term, threshold_term, p_toxic, top_toxic_class).

    boundary_term  = 2 * min(p_toxic, p_safe): 1.0 at the toxic-vs-safe coin flip
                     (the decision the app actually makes), 0.0 once it is settled.
    threshold_term = triangular kernel on |p_top_toxic - alert_threshold(top_toxic)|,
                     width `window`: 1.0 when the top toxic probability sits exactly
                     on the threshold that flips alert <-> no-alert.

    Deliberately near-blind to ivy-vs-oak disagreement: when p_toxic ~ 1 the boundary
    term is ~0 no matter how the toxic mass is split, because the user gets warned
    either way. Labeling budget goes to images that change whether a warning happens.
    """
    p_safe = float(p[SAFE_IDX])
    p_toxic = float(sum(p[i] for i in TOXIC_IDX))
    boundary = 2.0 * min(p_toxic, p_safe)      # in [0, 1]

    top_i = max(TOXIC_IDX, key=lambda i: p[i])
    top_toxic = CLASS_LABELS[top_i]
    thr = BASE_ALERT[top_toxic]
    dist = abs(float(p[top_i]) - thr)
    threshold_term = max(0.0, 1.0 - dist / window) if window > 0 else 0.0

    total_w = w_boundary + w_threshold
    score = (w_boundary * boundary + w_threshold * threshold_term) / total_w
    return float(score), float(boundary), float(threshold_term), p_toxic, top_toxic


def app_severity(p: np.ndarray) -> str:
    """What the SHIPPED app would do with this image: alert / uncertain / ignore.

    Mirrors scripts/ood_report.py::severity and the app's ToxicityThresholds so the
    operator can see the real-world consequence of each candidate while labeling.
    """
    top_i = max(TOXIC_IDX, key=lambda i: p[i])
    top = CLASS_LABELS[top_i]
    if float(p[top_i]) < float(p[SAFE_IDX]):
        return "ignore"
    thr = BASE_ALERT[top]
    if float(p[top_i]) >= thr:
        return "alert"
    if float(p[top_i]) >= thr - UNCERTAINTY_MARGIN:
        return "uncertain"
    return "ignore"


def score_record(p: np.ndarray, strategy: str, window: float, wb: float, wt: float) -> dict:
    """Compute every uncertainty measure plus the chosen strategy's ranking score.

    `score` is always normalized so HIGHER = label this first, regardless of strategy.
    `strategy_value` keeps the strategy's native units (e.g. margin stays p1 - p2).
    """
    lc = least_confidence(p)
    mg = margin(p)
    ent = entropy(p)
    ent_norm = ent / math.log(len(CLASS_LABELS))
    sf, boundary, thr_term, p_toxic, top_toxic = safety_score(p, window, wb, wt)

    if strategy == "margin":
        score, value = 1.0 - mg, mg
    elif strategy == "least_confidence":
        score, value = lc, lc
    elif strategy == "entropy":
        score, value = ent_norm, ent
    elif strategy == "safety":
        score, value = sf, sf
    else:  # unreachable: argparse constrains choices
        raise ValueError(f"unknown strategy {strategy!r}")

    return {
        "score": float(score),
        "strategy_value": float(value),
        "uncertainty": {
            "least_confidence": lc,
            "margin": mg,
            "entropy": ent,
            "entropy_norm": ent_norm,
            "safety": sf,
            "safety_boundary_term": boundary,
            "safety_threshold_term": thr_term,
        },
        "p_toxic": p_toxic,
        "p_safe": float(p[SAFE_IDX]),
        "top_toxic_class": top_toxic,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pool discovery + optional feedback-manifest enrichment
# ─────────────────────────────────────────────────────────────────────────────
def list_pool_images(pool_root: Path) -> list:
    """Recursive, deterministic listing of candidate images (jpg/jpeg/png)."""
    files = [
        p for p in pool_root.rglob("*")
        if p.is_file() and p.suffix.lower() in POOL_EXTS and not p.name.startswith(".")
    ]
    # Deterministic order -> deterministic tie-breaking -> idempotent worklists.
    return sorted(set(files), key=lambda p: str(p))


def load_feedback_manifest(path: Path) -> dict:
    """Best-effort read of a FeedbackExporter manifest.json -> {filename: entry}.

    Written by LeafAlert/Stores/FeedbackExporter.swift and consumed by
    scripts/ingest_feedback.py; shape is {"version": 1, "entries": [ {...} ]} with
    fields filename / originalPrediction / correctedLabel / feedbackStatus /
    confidence / timestamp / latitude / longitude.

    Used here only to ENRICH and to avoid re-labeling what a human already labeled.
    Never fatal: a missing or malformed manifest just means no enrichment.
    """
    if not path or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    out = {}
    for e in (data.get("entries") or []):
        if isinstance(e, dict) and e.get("filename"):
            out[Path(str(e["filename"])).name] = e
    return out


def feedback_info(entry: dict) -> dict:
    """Normalize the app-side fields we care about for the worklist."""
    return {
        "feedback_status": entry.get("feedbackStatus"),
        "on_device_prediction": entry.get("originalPrediction"),
        "corrected_label": entry.get("correctedLabel"),
        "on_device_confidence": entry.get("confidence"),
        "timestamp": entry.get("timestamp"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring pass (decode once: model input AND perceptual hashes)
# ─────────────────────────────────────────────────────────────────────────────
def score_pool(model, files, transform, batch_size, device):
    """Run the model over every candidate, computing hashes from the same decode.

    Returns (records, unreadable) where each record has path/probs/dhash/ahash.
    """
    records, unreadable = [], []
    buf_x, buf_rec = [], []

    def flush():
        if not buf_x:
            return
        x = torch.stack(buf_x).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
        for rec, p in zip(buf_rec, probs):
            rec["probs"] = p.astype(np.float64)
            records.append(rec)
        buf_x.clear()
        buf_rec.clear()

    for f in files:
        try:
            with Image.open(f) as im:
                im.load()               # full decode -> catches truncation
                im = im.convert("RGB")
            tensor = transform(im)
            # sha over the decoded RGB pixels, computed exactly as dataset_qa does,
            # so leak_match()'s exact-duplicate fast path actually fires. Without it
            # every candidate falls through to the O(references) Hamming scan and a
            # byte-identical held-out image is reported as a mere 'near' match.
            rec = {
                "path": f,
                "dhash": dhash(im),
                "ahash": ahash(im),
                "sha": hashlib.sha256(np.asarray(im, dtype=np.uint8).tobytes()).hexdigest(),
            }
        except Exception as e:          # noqa: BLE001 - want every failure mode
            unreadable.append((f, f"{type(e).__name__}: {e}"))
            continue
        buf_x.append(tensor)
        buf_rec.append(rec)
        if len(buf_x) >= batch_size:
            flush()
    flush()
    return records, unreadable


# ─────────────────────────────────────────────────────────────────────────────
# Cached reference index for the leakage guard
#
# dataset_qa.build_reference_index() re-decodes every one of the ~6.5k held-out +
# train-pool images on every run (measured 30s warm / 580s cold — it is I/O bound).
# That makes iterating on --strategy / --top painful, and pain is how leakage guards
# end up getting switched off. Only the FILE WALK and the hash cache live here; the
# hashes (dhash/ahash) and the match rule (leak_match) are still dataset_qa's, so the
# duplicate semantics cannot drift. --no-cache calls dataset_qa's implementation
# verbatim and is the reference this path is validated against.
# ─────────────────────────────────────────────────────────────────────────────
def _iter_reference_files(pool_root, heldout_root, exclude_names):
    """Exactly the traversal build_reference_index() performs, same origin tags.

    Held-out is yielded FIRST so its origin wins on an exact-hash tie (a held-out
    collision is the more alarming one to report). One level deep: <root>/<class>/*.
    """
    for root, prefix, is_pool in ((heldout_root, "heldout", False),
                                  (pool_root, "pool", True)):
        if root is None or not Path(root).is_dir():
            continue
        for cls_dir in sorted(Path(root).iterdir()):
            if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                continue
            if is_pool and cls_dir.name in exclude_names:
                continue
            for f in list_images(cls_dir):
                yield f, f"{prefix}:{cls_dir.name}/{f.name}", prefix


def _load_ref_cache(cache_path):
    if not cache_path or not Path(cache_path).is_file():
        return {}
    try:
        blob = json.loads(Path(cache_path).read_text())
    except Exception:
        return {}                       # corrupt cache -> just rebuild
    if blob.get("version") != REF_CACHE_VERSION:
        return {}
    entries = blob.get("entries")
    return entries if isinstance(entries, dict) else {}


def _save_ref_cache(cache_path, entries):
    """Atomic write so an interrupted run cannot leave a half-written cache."""
    try:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": REF_CACHE_VERSION,
                                   "entries": entries}))
        os.replace(tmp, cache_path)
    except Exception as e:              # noqa: BLE001 - cache is an optimization
        print(f"  (could not write hash cache: {type(e).__name__}: {e})")


def build_reference_index_cached(pool_root, heldout_root, exclude_names,
                                 cache_path, refresh=False):
    """Same return shape as dataset_qa.build_reference_index(), plus cache stats.

    Cache entries are keyed by absolute path and validated against mtime_ns + size,
    so an edited or replaced reference image is re-hashed automatically and a stale
    hash can never silently weaken the guard. Entries for files that no longer exist
    are pruned by construction (only files seen this run are written back).
    """
    cache = {} if refresh else _load_ref_cache(cache_path)
    dhash_list, sha_map = [], {}
    stats = {"heldout": 0, "pool": 0, "unreadable": 0}
    fresh, hits, misses = {}, 0, 0

    for f, origin, prefix in _iter_reference_files(pool_root, heldout_root, exclude_names):
        key = str(f.resolve())
        try:
            st = f.stat()
        except OSError:
            stats["unreadable"] += 1
            continue
        dh = ah = sha = None
        ent = cache.get(key)
        # Reuse ONLY on an exact (mtime_ns, size) match: a replaced or edited
        # reference image must be re-hashed, never guarded against a stale hash.
        if isinstance(ent, dict) and ent.get("m") == st.st_mtime_ns \
                and ent.get("s") == st.st_size:
            try:
                dh, ah, sha = int(ent["d"], 16), int(ent["a"], 16), str(ent["h"])
                hits += 1
            except (KeyError, TypeError, ValueError):
                dh = ah = sha = None    # malformed row -> fall through and re-hash
        if dh is None:
            try:
                with Image.open(f) as im:
                    im.load()
                    im = im.convert("RGB")
                dh, ah = dhash(im), ahash(im)
                sha = hashlib.sha256(np.asarray(im, dtype=np.uint8).tobytes()).hexdigest()
            except Exception:
                stats["unreadable"] += 1
                continue
            misses += 1
        fresh[key] = {"m": st.st_mtime_ns, "s": st.st_size,
                      "d": format(dh, "x"), "a": format(ah, "x"), "h": sha}
        dhash_list.append((dh, ah, origin))
        sha_map.setdefault(sha, origin)
        stats["heldout" if prefix == "heldout" else "pool"] += 1

    if cache_path:
        _save_ref_cache(cache_path, fresh)
    return {"dhash": dhash_list, "sha": sha_map, "stats": stats,
            "cache": {"hits": hits, "misses": misses, "path": str(cache_path or "")}}


# ─────────────────────────────────────────────────────────────────────────────
# Within-pool near-duplicate collapse (diversity)
# ─────────────────────────────────────────────────────────────────────────────
def collapse_pool_duplicates(ranked, dup_hamming):
    """Keep the highest-scoring representative of each near-duplicate cluster.

    `ranked` must already be sorted best-first, so the survivor of a burst of frames
    is the most informative one. Mutates records in place (sets 'drop'/'drop_detail')
    and returns the survivors.
    """
    kept, survivors = [], []
    for r in ranked:
        dup_of = None
        for kdh, kah, kname in kept:
            if hamming(r["dhash"], kdh) <= dup_hamming and \
               hamming(r["ahash"], kah) <= dup_hamming:
                dup_of = kname
                break
        if dup_of is not None:
            r["drop"] = "dup_within_pool"
            r["drop_detail"] = f"near-dup of higher-scoring candidate {dup_of}"
        else:
            kept.append((r["dhash"], r["ahash"], Path(r["path"]).name))
            survivors.append(r)
    return survivors


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
def build_candidate_json(rank, rec, pool_root, strategy):
    p = rec["probs"]
    pred_i = int(np.argmax(p))
    s = rec["scored"]
    path = Path(rec["path"])
    try:
        rel = str(path.relative_to(pool_root))
    except ValueError:
        rel = path.name
    return {
        "record_type": "candidate",
        "rank": rank,
        "path": str(path.resolve()),
        "relpath": rel,
        "prediction": CLASS_LABELS[pred_i],
        "confidence": float(p[pred_i]),
        # The model's own call, offered so a human can accept with one keystroke
        # instead of typing a class. Ingest maps 'not_a_plant' -> safe_plants.
        "suggested_label": CLASS_LABELS[pred_i],
        "probs": {c: float(p[i]) for i, c in enumerate(CLASS_LABELS)},
        "p_toxic": s["p_toxic"],
        "p_safe": s["p_safe"],
        "app_severity": app_severity(p),
        "strategy": strategy,
        "strategy_value": s["strategy_value"],
        "score": s["score"],
        "uncertainty": s["uncertainty"],
        "feedback": rec.get("feedback"),
    }


def write_text_worklist(path: Path, cands, meta):
    lines = []
    lines.append("LeafAlert active-learning worklist")
    lines.append(f"generated   : {meta['generated']}")
    lines.append(f"strategy    : {meta['strategy']}  (higher score = label first)")
    lines.append(f"checkpoint  : {meta['checkpoint']}")
    lines.append(f"pool        : {meta['pool']}")
    lines.append(f"selected    : {len(cands)} of {meta['eligible']} eligible "
                 f"({meta['scored']} scored, {meta['dropped_total']} dropped)")
    lines.append("")
    lines.append("Confirm the suggested label, or write the correct one. Classes: "
                 + ", ".join(CLASS_LABELS) + " (or not_a_plant -> safe_plants).")
    lines.append("")
    header = (f"{'#':>4}  {'score':>6}  {'suggested':<13} {'conf':>5}  "
              f"{'sev':<9} {'ivy':>5} {'oak':>5} {'sum':>5} {'safe':>5}  image")
    lines.append(header)
    lines.append("-" * len(header))
    for c in cands:
        pr = c["probs"]
        lines.append(
            f"{c['rank']:>4}  {c['score']:>6.3f}  {c['suggested_label']:<13} "
            f"{c['confidence']:>5.2f}  {c['app_severity']:<9} "
            f"{pr['poison_ivy']:>5.2f} {pr['poison_oak']:>5.2f} "
            f"{pr['poison_sumac']:>5.2f} {pr['safe_plants']:>5.2f}  {c['relpath']}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Rank an unlabeled image pool by how informative a human label would be.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pool", required=True,
                    help="Directory of UNLABELED candidate images (searched recursively).")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="Shipped model weights to score with.")
    ap.add_argument("--strategy", default="margin", choices=STRATEGIES,
                    help="Ranking strategy. 'safety' is the recommended one for shipping.")
    ap.add_argument("--top", type=int, default=50,
                    help="How many candidates to emit (0 = all eligible).")
    ap.add_argument("--out", default=None,
                    help="Write the ranked worklist here as JSONL.")
    ap.add_argument("--out-text", default=None,
                    help="Human-readable worklist path (default: --out with .txt suffix).")

    # Leakage guard / dedup
    ap.add_argument("--pool-dir", default=str(DEFAULT_POOL_DIR),
                    help="Existing TRAIN pool to guard against (already-owned images).")
    ap.add_argument("--heldout-dir", default=str(DEFAULT_HELDOUT_DIR),
                    help="FROZEN held-out set to guard against. Never written to.")
    ap.add_argument("--leak-hamming", type=int, default=5,
                    help="Max Hamming distance (dhash AND ahash) counted as a duplicate.")
    ap.add_argument("--no-leak-guard", action="store_true",
                    help="Skip the dedup/leakage guard. Iteration convenience ONLY.")
    ap.add_argument("--no-dedup-within-pool", action="store_true",
                    help="Keep near-duplicate candidates (e.g. every frame of a burst).")
    ap.add_argument("--ref-cache", default=str(DEFAULT_REF_CACHE),
                    help="Where to cache reference-image hashes (kept outside the repo).")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Ignore the cached hashes and re-hash every reference image.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Bypass the cache entirely; use dataset_qa's reference "
                         "implementation verbatim (slow, but the ground truth).")

    # Safety strategy tuning
    ap.add_argument("--safety-window", type=float, default=UNCERTAINTY_MARGIN,
                    help="Width of the alert-threshold proximity kernel.")
    ap.add_argument("--safety-boundary-weight", type=float, default=0.7,
                    help="Weight on the toxic-vs-safe boundary term.")
    ap.add_argument("--safety-threshold-weight", type=float, default=0.3,
                    help="Weight on the alert-threshold proximity term.")

    # Feedback-stream integration
    ap.add_argument("--feedback-manifest", default=None,
                    help="FeedbackExporter manifest.json (default: <pool>/manifest.json).")
    ap.add_argument("--include-labeled", action="store_true",
                    help="Also score images the manifest marks confirmed/corrected "
                         "(default: skip them, a human already labeled those).")

    # Runtime
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu",
                    help="torch device (cpu recommended: deterministic, parity-safe).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Score at most this many pool images (0 = no limit).")
    args = ap.parse_args()

    pool_root = Path(args.pool)
    if not pool_root.is_dir():
        print(f"ERROR: --pool is not a directory: {pool_root}", file=sys.stderr)
        sys.exit(1)
    if args.top < 0:
        print("ERROR: --top must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.safety_boundary_weight < 0 or args.safety_threshold_weight < 0 or \
            (args.safety_boundary_weight + args.safety_threshold_weight) <= 0:
        print("ERROR: safety weights must be >= 0 and not both zero", file=sys.stderr)
        sys.exit(1)
    if args.safety_window <= 0:
        print("ERROR: --safety-window must be > 0", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)

    print(f"Pool        : {pool_root}   (UNLABELED candidates)")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Strategy    : {args.strategy}")
    print(f"Device      : {device}")

    # ── Candidates ────────────────────────────────────────────────────────
    files = list_pool_images(pool_root)
    if not files:
        print(f"ERROR: no {'/'.join(sorted(POOL_EXTS))} images under {pool_root}",
              file=sys.stderr)
        sys.exit(1)

    manifest_path = Path(args.feedback_manifest) if args.feedback_manifest \
        else (pool_root / "manifest.json")
    fb = load_feedback_manifest(manifest_path)
    if fb:
        print(f"Feedback    : {len(fb)} manifest entries from {manifest_path}")

    n_already_labeled = 0
    if fb and not args.include_labeled:
        keep = []
        for f in files:
            e = fb.get(f.name)
            if e and e.get("feedbackStatus") in ("confirmed", "corrected"):
                n_already_labeled += 1   # a human already spent effort here
                continue
            keep.append(f)
        files = keep

    n_found = len(files) + n_already_labeled
    if args.limit and len(files) > args.limit:
        files = files[:args.limit]

    if not files:
        print("\nNo unlabeled candidates left after manifest filtering "
              f"({n_already_labeled} already labeled). Nothing to do.")
        sys.exit(0)

    # ── Model ─────────────────────────────────────────────────────────────
    transform = build_val_transforms(IMAGE_SIZE)
    _assert_parity(transform)      # train<->device preprocessing contract
    model = load_model(Path(args.checkpoint), device)

    t0 = time.time()
    print(f"\nScoring {len(files)} candidate image(s)...")
    records, unreadable = score_pool(model, files, transform, args.batch_size, device)
    print(f"  scored {len(records)} in {time.time() - t0:.1f}s"
          + (f"  ({len(unreadable)} unreadable)" if unreadable else ""))

    # ── Dedup + leakage guard ─────────────────────────────────────────────
    for r in records:
        r["drop"] = None
        r["drop_detail"] = None

    heldout_root = Path(args.heldout_dir)
    train_pool_root = Path(args.pool_dir)
    n_leak_heldout = n_leak_pool = n_ref_indexed = 0

    if args.no_leak_guard:
        print("\n*** --no-leak-guard: NOT checking candidates against the frozen "
              "held-out set or the train pool. ***")
        print("    Anything selected may already be owned, or may be a held-out image "
              "whose label would leak the test set.")
        print("    dataset_qa.py is still the authoritative gate before any image "
              "enters the train pool.")
    else:
        # Exclude the nested held-out dir from the pool scan so it is not indexed
        # twice and the 'heldout:' vs 'pool:' origin tags stay meaningful.
        exclude = set()
        try:
            if heldout_root.resolve().parent == train_pool_root.resolve():
                exclude.add(heldout_root.name)
        except OSError:
            pass

        t1 = time.time()
        print("\nIndexing reference set for the leakage guard "
              "(frozen held-out + existing train pool)...")
        if args.no_cache:
            print("  --no-cache: re-hashing every reference image "
                  "(dataset_qa.build_reference_index verbatim; this is slow)")
            ref = build_reference_index(train_pool_root, heldout_root,
                                        exclude_names=exclude)
        else:
            ref = build_reference_index_cached(
                train_pool_root, heldout_root, exclude,
                Path(args.ref_cache), refresh=args.refresh_cache)
        print(f"  indexed {ref['stats']['heldout']} held-out + {ref['stats']['pool']} "
              f"train-pool images in {time.time() - t1:.1f}s")
        if "cache" in ref:
            c = ref["cache"]
            print(f"  hash cache: {c['hits']} hit / {c['misses']} re-hashed  "
                  f"-> {c['path']}")
        if ref["stats"]["unreadable"]:
            print(f"  ({ref['stats']['unreadable']} reference image(s) unreadable "
                  f"and therefore NOT guarded against)")
        n_ref_indexed = ref["stats"]["heldout"] + ref["stats"]["pool"]

        heldout_resolved = heldout_root.resolve() if heldout_root.exists() else None
        for r in records:
            # Path-level check first: a candidate literally inside the frozen set.
            if heldout_resolved is not None:
                try:
                    Path(r["path"]).resolve().relative_to(heldout_resolved)
                    r["drop"] = "leak_heldout"
                    r["drop_detail"] = "candidate path is inside the frozen held-out set"
                    n_leak_heldout += 1
                    continue
                except ValueError:
                    pass
            m = leak_match(r, ref, args.leak_hamming)
            if m is not None:
                origin, kind = m
                if origin.startswith("heldout:"):
                    r["drop"] = "leak_heldout"
                    n_leak_heldout += 1
                else:
                    r["drop"] = "leak_pool"
                    n_leak_pool += 1
                r["drop_detail"] = f"{kind} match to {origin}"

    live = [r for r in records if r["drop"] is None]

    # ── Score + rank ──────────────────────────────────────────────────────
    for r in live:
        r["scored"] = score_record(r["probs"], args.strategy, args.safety_window,
                                   args.safety_boundary_weight,
                                   args.safety_threshold_weight)
        if fb:
            e = fb.get(Path(r["path"]).name)
            r["feedback"] = feedback_info(e) if e else None
        else:
            r["feedback"] = None

    # Deterministic: score desc, then path asc so equal scores never reshuffle.
    live.sort(key=lambda r: (-r["scored"]["score"], str(r["path"])))

    n_dup_within = 0
    if not args.no_dedup_within_pool:
        before = len(live)
        live = collapse_pool_duplicates(live, args.leak_hamming)
        n_dup_within = before - len(live)

    n_eligible = len(live)
    selected = live if args.top == 0 else live[:args.top]

    cands = [build_candidate_json(i + 1, r, pool_root, args.strategy)
             for i, r in enumerate(selected)]

    # ── Summary ───────────────────────────────────────────────────────────
    dropped_total = n_leak_heldout + n_leak_pool + n_dup_within

    hr("CANDIDATE ACCOUNTING")
    print(f"  images found in pool        : {n_found}")
    if n_already_labeled:
        print(f"  skipped: already labeled    : {n_already_labeled}  "
              f"(manifest feedbackStatus confirmed/corrected)")
    if args.limit and n_found - n_already_labeled > args.limit:
        print(f"  capped by --limit           : {args.limit}")
    if unreadable:
        print(f"  unreadable / corrupt        : {len(unreadable)}")
    print(f"  scored by the model         : {len(records)}")
    print(f"  dropped: LEAK vs held-out   : {n_leak_heldout}"
          + ("   *** would have leaked the frozen test set ***" if n_leak_heldout else ""))
    print(f"  dropped: dup of train pool  : {n_leak_pool}   (already owned)")
    print(f"  dropped: dup within pool    : {n_dup_within}   (redundant frames)")
    print(f"  ---------------------------------------")
    print(f"  eligible candidates         : {n_eligible}")
    _sel_label = f"selected (--top {args.top if args.top else 'all'})"
    print(f"  {_sel_label:<28}: {len(cands)}")

    if cands:
        hr(f"TOP-{len(cands)} COMPOSITION  (strategy={args.strategy})")
        dist = Counter(c["prediction"] for c in cands)
        sev = Counter(c["app_severity"] for c in cands)
        for c in CLASS_LABELS:
            n = dist.get(c, 0)
            bar = "#" * int(round(30 * n / len(cands)))
            flag = "[TOXIC]" if c in TOXIC_CLASSES else "[safe] "
            print(f"  {c:<14}{flag} {n:>4}  {n/len(cands):>6.1%}  {bar}")
        print(f"\n  app severity if shipped as-is: "
              + ", ".join(f"{k}={v}" for k, v in sorted(sev.items())))
        scores = [c["score"] for c in cands]
        print(f"  score range: {min(scores):.3f} .. {max(scores):.3f}  "
              f"(mean {sum(scores)/len(scores):.3f})")

        # Selection collapse is the classic active-learning failure: the worklist
        # fills with one class and the round teaches the model nothing new.
        top_cls, top_n = dist.most_common(1)[0]
        if len(cands) >= 5 and top_n / len(cands) >= 0.70:
            print(f"\n  WARNING: {top_n/len(cands):.0%} of the worklist is predicted "
                  f"'{top_cls}' — selection is collapsing onto one class.")
            print("  Consider a different --strategy, a larger --top, or a more "
                  "diverse pool before labeling.")

        hr("WORKLIST PREVIEW (label these first)")
        for c in cands[:10]:
            pr = c["probs"]
            print(f"  #{c['rank']:<3} score={c['score']:.3f} "
                  f"{args.strategy}={c['strategy_value']:.3f}  "
                  f"suggest={c['suggested_label']:<13} conf={c['confidence']:.2f} "
                  f"sev={c['app_severity']:<9}")
            print(f"        p(ivy)={pr['poison_ivy']:.3f} p(oak)={pr['poison_oak']:.3f} "
                  f"p(sumac)={pr['poison_sumac']:.3f} p(safe)={pr['safe_plants']:.3f}")
            print(f"        {c['path']}")
        if len(cands) > 10:
            print(f"  ... and {len(cands) - 10} more (see --out)")
    else:
        hr("WORKLIST")
        print("  0 eligible candidates — every image was a duplicate of data we "
              "already own, or of the frozen held-out set.")

    # ── Write outputs ─────────────────────────────────────────────────────
    meta = {
        "record_type": "meta",
        "tool": "active_learning.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "classes": CLASS_LABELS,
        "pool": str(pool_root.resolve()),
        "strategy": args.strategy,
        "top": args.top,
        "preprocessing": {
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "crop": None,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "safety_params": {
            "alert_thresholds": BASE_ALERT,
            "window": args.safety_window,
            "boundary_weight": args.safety_boundary_weight,
            "threshold_weight": args.safety_threshold_weight,
        },
        "guard": {
            "enabled": not args.no_leak_guard,
            "leak_hamming": args.leak_hamming,
            "heldout_dir": str(heldout_root),
            "train_pool_dir": str(train_pool_root),
            "dedup_within_pool": not args.no_dedup_within_pool,
            "reference_images_indexed": n_ref_indexed,
        },
        "counts": {
            "found": n_found,
            "already_labeled_skipped": n_already_labeled,
            "unreadable": len(unreadable),
            "scored": len(records),
            "dropped_leak_heldout": n_leak_heldout,
            "dropped_dup_train_pool": n_leak_pool,
            "dropped_dup_within_pool": n_dup_within,
            "dropped_total": dropped_total,
            "eligible": n_eligible,
            "selected": len(cands),
        },
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:      # overwrite: idempotent re-runs
            fh.write(json.dumps(meta) + "\n")
            for c in cands:
                fh.write(json.dumps(c) + "\n")
        print(f"\nWorklist (JSONL): {out_path}   [1 meta line + {len(cands)} candidates]")

        text_path = Path(args.out_text) if args.out_text else out_path.with_suffix(".txt")
        # `--out worklist.txt` would make with_suffix('.txt') a no-op, and the text
        # table would silently overwrite the JSONL that was just written (while the
        # log still claims both were saved). Never let the two collide.
        if text_path.resolve() == out_path.resolve():
            text_path = out_path.with_name(out_path.name + ".txt")
        write_text_worklist(text_path, cands, {
            "generated": meta["generated"],
            "strategy": args.strategy,
            "checkpoint": meta["checkpoint"],
            "pool": meta["pool"],
            "eligible": n_eligible,
            "scored": len(records),
            "dropped_total": dropped_total,
        })
        print(f"Worklist (text) : {text_path}")
    elif args.out_text:
        text_path = Path(args.out_text)
        write_text_worklist(text_path, cands, {
            "generated": meta["generated"],
            "strategy": args.strategy,
            "checkpoint": meta["checkpoint"],
            "pool": meta["pool"],
            "eligible": n_eligible,
            "scored": len(records),
            "dropped_total": dropped_total,
        })
        print(f"\nWorklist (text) : {text_path}")
    else:
        print("\n(no --out given: nothing written. Add --out worklist.jsonl to save.)")

    if unreadable:
        print(f"\n{len(unreadable)} unreadable image(s):")
        for f, err in unreadable[:5]:
            print(f"  {f}: {err}")
        if len(unreadable) > 5:
            print(f"  ... and {len(unreadable) - 5} more")

    print()


if __name__ == "__main__":
    main()
