#!/usr/bin/env python3
"""
Quality-control, de-duplication, and LEAKAGE GUARD for freshly staged images.

This is the gate every fetched/scraped image must pass before it is allowed
into the LeafAlert training pool. It is deliberately STRICT: the value of the
frozen held-out set (TrainingData/Testing) depends entirely on new training
data never overlapping it. If this guard is loose, held-out metrics get
silently inflated and the safety-critical recall numbers become fiction.

What it does
------------
Given a *staging* directory laid out by class:

    <staged>/poison_ivy/*.jpg
    <staged>/poison_oak/*.jpg
    <staged>/poison_sumac/*.jpg
    <staged>/safe_plants/*.jpg

it decides, per image, KEEP or DROP with an auditable reason:

  1. QUALITY FILTERS
       - unreadable / corrupt / truncated file
       - below minimum resolution (min side)
       - degenerate / extreme aspect ratio
       - effectively greyscale (near-zero colour saturation) — poison-plant
         ID leans heavily on colour, so mono images are near-useless and
         often line art / diagrams

  2. DE-DUPLICATION (perceptual dHash + aHash, Hamming distance)
       - exact decoded-pixel duplicates (SHA-256)
       - near-duplicates WITHIN a staged class (redundant)
       - near-duplicates ACROSS staged classes (label conflict / ambiguous;
         dropped from BOTH by default — we cannot trust either label)

  3. LEAKAGE GUARD  (the whole point)
       Drop any staged image that is a near-duplicate (Hamming <= --leak-hamming)
       of ANY image in:
         (a) the frozen held-out set  TrainingData/Testing/**   -> would leak
             the test set into training and inflate held-out metrics
         (b) the existing train pool   TrainingData/{class}/**   -> redundant,
             already have it
       This is checked across ALL classes, not just the matching class:
       a staged poison_ivy that duplicates a Testing/safe_plants image is
       still leakage and is still dropped.

  4. REPORT
       Per-class kept/dropped counts, with a breakdown of drop reasons, plus
       (in commit mode) a JSON manifest recording provenance, license/
       attribution (if present in a sibling staging manifest), and the exact
       source->destination mapping for every committed file.

  5. --commit
       Copy survivors into  TrainingData/<class>/  (the TRAIN POOL ONLY —
       NEVER TrainingData/Testing) with source-tagged filenames so their
       origin stays visible forever. Default is DRY-RUN.

The perceptual-hash implementation here is byte-for-byte the same construction
as scripts/audit_dataset.py (dependency-free, PIL + numpy), so hashes are
directly comparable between the two tools.

Usage
-----
    # dry-run QA report on a staging dir (safe default, changes nothing)
    python3 scripts/dataset_qa.py --staged staged_downloads

    # actually copy survivors into the train pool + write a manifest
    python3 scripts/dataset_qa.py --staged staged_downloads --commit

    # tune strictness
    python3 scripts/dataset_qa.py --staged staged_downloads \
        --min-side 128 --dup-hamming 5 --leak-hamming 5 \
        --min-aspect 0.4 --max-aspect 2.5 --sat-thresh 0.05

    # keep across-class near-dup collisions instead of dropping both
    python3 scripts/dataset_qa.py --staged staged_downloads --keep-crossclass

Exit codes: 0 on success (even if everything was dropped), 1 on bad arguments
or a missing/unreadable staging directory.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

# Be strict: DETECT truncated files rather than silently padding them.
ImageFile.LOAD_TRUNCATED_IMAGES = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical class order (ImageFolder-alphabetical). DO NOT change.
CANONICAL_CLASSES = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]
TOXIC_CLASSES = {"poison_ivy", "poison_oak", "poison_sumac"}

# The frozen, leakage-free held-out set. NEVER commit into it; ALWAYS guard
# staged data against it.
DEFAULT_TRAIN_POOL = PROJECT_ROOT / "TrainingData"
DEFAULT_HELDOUT = PROJECT_ROOT / "TrainingData" / "Testing"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual hashing (dependency-free) — identical construction to
# scripts/audit_dataset.py so hashes are cross-comparable.
# ─────────────────────────────────────────────────────────────────────────────
def _gray_small(img, size):
    return np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)


def dhash(img, size=8):
    """Difference hash: compares adjacent pixels. size*size bits (64 for size=8)."""
    px = _gray_small(img, size + 1)  # need one extra column
    diff = px[:, 1:] > px[:, :-1]
    return _bits_to_int(diff.flatten())


def ahash(img, size=8):
    """Average hash: pixel > mean. size*size bits."""
    px = _gray_small(img, size)
    diff = px > px.mean()
    return _bits_to_int(diff.flatten())


def _bits_to_int(bits):
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a, b):
    return bin(a ^ b).count("1")


# ─────────────────────────────────────────────────────────────────────────────
# Colour saturation (greyscale detection)
# ─────────────────────────────────────────────────────────────────────────────
def mean_saturation(img):
    """
    Mean HSV saturation in [0, 1]. Near-zero => effectively greyscale
    (scanned line art, B/W botanical plates, desaturated diagrams), which are
    near-useless for a colour-dependent poison-plant classifier.

    Computed on a downscaled copy for speed; the mean is scale-stable.
    """
    small = img.convert("RGB").resize((64, 64), Image.BILINEAR)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    return float(hsv[:, :, 1].mean() / 255.0)


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem helpers
# ─────────────────────────────────────────────────────────────────────────────
def list_images(d):
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    )


def staged_class_dirs(staged_root):
    """
    Class subdirectories under the staging root. Only recognised canonical
    class names are treated as class buckets; anything else (e.g. a
    'manifest.json' or an 'unsorted/' dir) is ignored, and reported.
    """
    found = {}
    extras = []
    for p in sorted(staged_root.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            if p.name in CANONICAL_CLASSES:
                found[p.name] = p
            else:
                extras.append(p.name)
    return found, extras


# ─────────────────────────────────────────────────────────────────────────────
# Reference index (train pool + held-out) for the leakage guard
# ─────────────────────────────────────────────────────────────────────────────
def build_reference_index(pool_root, heldout_root, exclude_names=None):
    """
    Build the set of perceptual/exact hashes for every image we must NOT
    duplicate: the frozen held-out set AND the existing train pool, across ALL
    classes. Returns a dict with:
        'dhash': [ (dhash, ahash, origin_str) ... ]
        'sha':   { sha256 -> origin_str }
    origin_str is a short human tag like 'heldout:poison_ivy/foo.jpg'.

    exclude_names: directory names directly under pool_root to skip when scanning
    the pool (used to avoid double-scanning the nested Testing dir as a class).
    """
    exclude_names = set(exclude_names or [])
    dhash_list = []
    sha_map = {}
    stats = {"heldout": 0, "pool": 0, "unreadable": 0}

    def scan(root, origin_prefix, is_pool):
        if not root or not root.is_dir():
            return
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                continue
            if is_pool and cls_dir.name in exclude_names:
                continue
            for f in list_images(cls_dir):
                try:
                    with Image.open(f) as im:
                        im.load()
                        im = im.convert("RGB")
                    dh, ah = dhash(im), ahash(im)
                    sha = hashlib.sha256(np.asarray(im, dtype=np.uint8).tobytes()).hexdigest()
                except Exception:
                    stats["unreadable"] += 1
                    continue
                origin = f"{origin_prefix}:{cls_dir.name}/{f.name}"
                dhash_list.append((dh, ah, origin))
                sha_map.setdefault(sha, origin)
                stats["heldout" if origin_prefix == "heldout" else "pool"] += 1

    # Held-out first so its origin tag wins on exact-hash ties (more alarming).
    scan(heldout_root, "heldout", is_pool=False)
    scan(pool_root, "pool", is_pool=True)
    return {"dhash": dhash_list, "sha": sha_map, "stats": stats}


def leak_match(rec, ref, leak_hamming):
    """
    Return (origin, kind) if `rec` duplicates something in the reference index,
    else None. kind is 'exact' or 'near'. Exact SHA match wins.
    """
    if rec.get("sha") and rec["sha"] in ref["sha"]:
        return ref["sha"][rec["sha"]], "exact"
    dh, ah = rec.get("dhash"), rec.get("ahash")
    if dh is None:
        return None
    for rdh, rah, origin in ref["dhash"]:
        if hamming(dh, rdh) <= leak_hamming and hamming(ah, rah) <= leak_hamming:
            return origin, "near"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Staged scan + quality filters
# ─────────────────────────────────────────────────────────────────────────────
def scan_staged(class_dirs, min_side, min_aspect, max_aspect, sat_thresh, max_pixels=None):
    """
    Load every staged image, run quality filters, compute hashes.
    Returns a list of per-file records. A record that fails a quality filter
    gets rec['drop'] set to the reason and has no hashes.
    """
    records = []
    for cls, cdir in class_dirs.items():
        for f in list_images(cdir):
            rec = {"cls": cls, "path": str(f), "name": f.name, "drop": None,
                   "drop_detail": None}
            try:
                with Image.open(f) as im:
                    im.load()  # force full decode -> catches truncation
                    im = im.convert("RGB")
            except Exception as e:  # noqa: BLE001 - want every failure mode
                rec["drop"] = "corrupt"
                rec["drop_detail"] = f"{type(e).__name__}: {e}"
                records.append(rec)
                continue

            w, h = im.size
            rec["w"], rec["h"] = w, h
            # Oversized originals (a 100 MP GBIF scan turned up in the pool) are
            # not dropped — they are downsized on commit. Decoding them every
            # epoch costs seconds and warns as a decompression bomb.
            rec["oversized"] = bool(max_pixels and w * h > max_pixels)
            side = min(w, h)
            aspect = (w / h) if h else 0.0
            rec["aspect"] = round(aspect, 3)

            if side < min_side:
                rec["drop"] = "low_resolution"
                rec["drop_detail"] = f"min_side={side} < {min_side}"
                records.append(rec)
                continue

            if aspect < min_aspect or aspect > max_aspect:
                rec["drop"] = "bad_aspect"
                rec["drop_detail"] = f"aspect={aspect:.3f} not in [{min_aspect},{max_aspect}]"
                records.append(rec)
                continue

            sat = mean_saturation(im)
            rec["saturation"] = round(sat, 4)
            if sat < sat_thresh:
                rec["drop"] = "greyscale"
                rec["drop_detail"] = f"mean_saturation={sat:.4f} < {sat_thresh}"
                records.append(rec)
                continue

            # Passed quality gate -> compute hashes for dedup/leakage.
            rec["dhash"] = dhash(im)
            rec["ahash"] = ahash(im)
            rec["sha"] = hashlib.sha256(np.asarray(im, dtype=np.uint8).tobytes()).hexdigest()
            records.append(rec)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# De-duplication among the staged survivors
# ─────────────────────────────────────────────────────────────────────────────
def dedup_staged(live, dup_hamming, keep_crossclass):
    """
    Mutates `live` records in place, setting rec['drop'] for dropped items.
    `live` is the list of records that passed quality AND leakage guards
    (i.e. rec['drop'] is None and hashes are present).

    Order of operations:
      1. across-class near-dups -> ambiguous label; by default drop BOTH
         (reason 'crossclass_conflict'). With keep_crossclass, keep both.
         This runs FIRST so a cross-class EXACT duplicate (Hamming 0) is treated
         as an ambiguous-label conflict and both sides drop, rather than being
         silently collapsed to one arbitrary label by same-pixel dedup.
      2. exact SHA duplicates WITHIN a class -> keep first, drop rest
         (reason 'exact_dup')
      3. within-class near-dups -> keep first, drop rest (reason 'near_dup')

    Deterministic: records are processed in their existing (sorted) order.
    """
    survivors = [r for r in live if r["drop"] is None]

    # 1) across-class near-duplicate collisions (label conflict)
    #    O(n^2) but n is small (a staging batch), like audit_dataset.py.
    if not keep_crossclass:
        conflicted = set()  # ids of records to drop
        details = {}
        n = len(survivors)
        for i in range(n):
            a = survivors[i]
            for j in range(i + 1, n):
                b = survivors[j]
                if a["cls"] == b["cls"]:
                    continue
                if hamming(a["dhash"], b["dhash"]) <= dup_hamming and \
                   hamming(a["ahash"], b["ahash"]) <= dup_hamming:
                    conflicted.add(id(a))
                    conflicted.add(id(b))
                    details[id(a)] = f"{b['cls']}/{b['name']}"
                    details[id(b)] = f"{a['cls']}/{a['name']}"
        for r in survivors:
            if id(r) in conflicted:
                r["drop"] = "crossclass_conflict"
                r["drop_detail"] = f"near-dup of {details[id(r)]} (different label)"
        survivors = [r for r in survivors if r["drop"] is None]

    # 2) exact-pixel duplicates WITHIN a class (keep first, drop rest).
    #    Scoped by class so cross-class exact dups are handled by step 1 above.
    seen_sha = {}
    for r in survivors:
        key = (r["cls"], r["sha"])
        if key in seen_sha:
            r["drop"] = "exact_dup"
            r["drop_detail"] = f"same pixels as {Path(seen_sha[key]).name}"
        else:
            seen_sha[key] = r["path"]
    survivors = [r for r in survivors if r["drop"] is None]

    # 3) within-class near-duplicates (redundant)
    by_cls = defaultdict(list)
    for r in survivors:
        by_cls[r["cls"]].append(r)
    for cls, recs in by_cls.items():
        kept = []  # (dhash, ahash, name)
        for r in recs:
            dup_of = None
            for kdh, kah, kname in kept:
                if hamming(r["dhash"], kdh) <= dup_hamming and \
                   hamming(r["ahash"], kah) <= dup_hamming:
                    dup_of = kname
                    break
            if dup_of is not None:
                r["drop"] = "near_dup"
                r["drop_detail"] = f"near-dup of {cls}/{dup_of}"
            else:
                kept.append((r["dhash"], r["ahash"], r["name"]))


# ─────────────────────────────────────────────────────────────────────────────
# Provenance manifest (staging-side, optional)
# ─────────────────────────────────────────────────────────────────────────────
def load_staging_manifest(staged_root):
    """
    If the fetcher wrote a provenance manifest into the staging dir, load a
    basename -> provenance mapping so we can carry license/attribution/source
    and (critically) the source OBSERVATION id forward into the train pool.

    Supported shapes, in priority order:
      * manifest.jsonl  (one JSON object per line) -- the format BOTH fetchers
        emit (scripts/fetch_inaturalist.py, scripts/fetch_gbif.py). Field names
        differ between the two fetchers and are reconciled here.
      * manifest.json / provenance.json (single JSON blob), either:
          { "entries": [ {"filename": ..., "class": ..., ...} ] }
          OR a flat { "<filename>": { ...provenance... } } mapping.

    The map is keyed on the image BASENAME so a lookup by src.name works even
    though iNat rows store a repo-relative 'path' and GBIF rows a staging-
    relative 'file'. Every entry is normalized to a common shape carrying:
      class, source, license, attribution, url, observation_id.

    Returns {basename: provenance_dict}. Absent/unparseable -> {}.
    """
    def _norm_row(e):
        """Reconcile the divergent iNat vs GBIF field names into one shape."""
        return {
            "class": e.get("class") or e.get("cls"),
            "source": e.get("source"),
            "license": e.get("license"),
            "attribution": e.get("attribution"),
            "url": e.get("url") or e.get("image_url") or e.get("source_url"),
            # iNat: observation_id ; GBIF: occurrence_id. Either is the group id
            # that makes the train/val split truly observation-disjoint.
            "observation_id": (
                e.get("observation_id")
                or e.get("occurrence_id")
                or e.get("obs_id")
            ),
        }

    def _basename_of(e):
        p = e.get("path") or e.get("file") or e.get("filename")
        return Path(p).name if p else None

    out = {}

    # 1) JSONL manifests (what the fetchers actually write). Prefer these.
    for name in ("manifest.jsonl",):
        mp = staged_root / name
        if not mp.exists():
            continue
        try:
            for line in mp.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                bn = _basename_of(row)
                if bn:
                    out[bn] = _norm_row(row)
        except Exception:
            pass  # best-effort; fall through to blob manifests below

    # 2) Single-blob manifests (back-compat with the older 'entries'/flat shapes).
    for name in ("manifest.json", "provenance.json"):
        mp = staged_root / name
        if not mp.exists():
            continue
        try:
            with open(mp) as fh:
                data = json.load(fh)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            for e in data["entries"]:
                bn = _basename_of(e) or e.get("filename")
                if bn:
                    out.setdefault(Path(bn).name, _norm_row(e))
        elif isinstance(data, dict):
            for fn, prov in data.items():
                if isinstance(prov, dict):
                    out.setdefault(Path(fn).name, _norm_row(prov))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Commit
# ─────────────────────────────────────────────────────────────────────────────
def next_start_index(dest_dir, tag):
    """
    Find a safe starting counter for source-tagged filenames so we never
    collide with an existing committed batch from the same source tag.
    Filenames look like  qa_<tag>_<cls>_<NNNN>.<ext>.
    """
    if not dest_dir.is_dir():
        return 0
    mx = -1
    prefix = f"qa_{tag}_"
    for p in dest_dir.iterdir():
        stem = p.stem
        if stem.startswith(prefix):
            tail = stem.rsplit("_", 1)[-1]
            if tail.isdigit():
                mx = max(mx, int(tail))
    return mx + 1


def downsize_copy(src: Path, dest: Path, long_edge: int):
    """Write `src` to `dest` with its long edge capped at `long_edge` px.
    EXIF orientation is baked in first (thumbnails lose the tag); JPEG at
    quality 95, PNG kept as PNG."""
    from PIL import ImageOps
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((long_edge, long_edge), Image.LANCZOS)
        if dest.suffix.lower() == ".png":
            im.save(dest, format="PNG", optimize=True)
        else:
            im.convert("RGB").save(dest, format="JPEG", quality=95, optimize=True)


def commit_survivors(survivors, pool_root, tag, staging_prov, dry_run, downsize_long_edge=None):
    """
    Copy each survivor into  pool_root/<cls>/  with a source-tagged filename.
    NEVER writes into a 'Testing' dir. Returns a list of manifest entries.
    """
    manifest_entries = []
    # Assign per-class running indices, seeded past any existing same-tag batch.
    counters = {}
    for r in survivors:
        cls = r["cls"]
        dest_dir = pool_root / cls
        if cls not in counters:
            counters[cls] = next_start_index(dest_dir, tag)
        idx = counters[cls]
        counters[cls] += 1

        src = Path(r["path"])
        ext = src.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        dest_name = f"qa_{tag}_{cls}_{idx:04d}{ext}"
        dest = dest_dir / dest_name

        prov = staging_prov.get(src.name, {})
        entry = {
            "dest": f"{cls}/{dest_name}",
            "source_path": str(src),
            "source_name": src.name,
            "class": cls,
            "width": r.get("w"),
            "height": r.get("h"),
            "downsized_to_long_edge": downsize_long_edge if (r.get("oversized") and downsize_long_edge) else None,
            "sha256_pixels": r.get("sha"),
            "source": prov.get("source"),
            "license": prov.get("license"),
            "attribution": prov.get("attribution"),
            "source_url": prov.get("url") or prov.get("source_url"),
            # Carry the source observation/occurrence id so the training split
            # can be made truly observation-disjoint (see train_v5.py).
            "observation_id": prov.get("observation_id"),
        }
        manifest_entries.append(entry)

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                # Extremely unlikely given seeding, but never clobber.
                raise FileExistsError(f"Refusing to overwrite existing file: {dest}")
            if r.get("oversized") and downsize_long_edge:
                downsize_copy(src, dest, downsize_long_edge)
            else:
                shutil.copy2(src, dest)
    return manifest_entries


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
DROP_REASONS = [
    "corrupt", "low_resolution", "bad_aspect", "greyscale",
    "leak_heldout", "leak_pool", "exact_dup", "crossclass_conflict", "near_dup",
]

REASON_BLURB = {
    "corrupt": "unreadable / truncated file",
    "low_resolution": "below min resolution",
    "bad_aspect": "extreme aspect ratio",
    "greyscale": "effectively greyscale",
    "leak_heldout": "LEAKAGE vs frozen held-out set",
    "leak_pool": "duplicate of existing train pool",
    "exact_dup": "exact-pixel duplicate within batch",
    "crossclass_conflict": "near-dup across classes (label conflict)",
    "near_dup": "near-duplicate within class",
}


def hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def print_report(records, ref_stats, extras, args):
    per_class_total = defaultdict(int)
    per_class_kept = defaultdict(int)
    per_class_reasons = defaultdict(lambda: defaultdict(int))
    total_reasons = defaultdict(int)

    for r in records:
        per_class_total[r["cls"]] += 1
        if r["drop"] is None:
            per_class_kept[r["cls"]] += 1
        else:
            per_class_reasons[r["cls"]][r["drop"]] += 1
            total_reasons[r["drop"]] += 1

    hr("REFERENCE INDEX (leakage guard baseline)")
    print(f"  held-out images indexed : {ref_stats['heldout']}")
    print(f"  train-pool images indexed: {ref_stats['pool']}")
    if ref_stats["unreadable"]:
        print(f"  (skipped {ref_stats['unreadable']} unreadable reference images)")
    print(f"  dup Hamming <= {args.dup_hamming} (within batch), "
          f"leak Hamming <= {args.leak_hamming} (vs reference)")

    if extras:
        hr("UNRECOGNISED STAGING SUBDIRS (ignored)")
        for name in extras:
            print(f"  ignored: {name}/  (not a canonical class)")

    hr("PER-CLASS KEEP / DROP")
    grand_total = grand_kept = 0
    for cls in CANONICAL_CLASSES:
        tot = per_class_total.get(cls, 0)
        if tot == 0:
            continue
        kept = per_class_kept.get(cls, 0)
        grand_total += tot
        grand_kept += kept
        flag = "[TOXIC]" if cls in TOXIC_CLASSES else "[safe/neg]"
        print(f"\n  {cls:<14} {flag}")
        print(f"    staged={tot:>4}  kept={kept:>4}  dropped={tot - kept:>4}")
        reasons = per_class_reasons.get(cls, {})
        for reason in DROP_REASONS:
            c = reasons.get(reason, 0)
            if c:
                print(f"      - {reason:<20} {c:>4}   ({REASON_BLURB[reason]})")

    hr("TOTALS")
    print(f"  staged : {grand_total}")
    print(f"  kept   : {grand_kept}")
    print(f"  dropped: {grand_total - grand_kept}")
    if total_reasons:
        print("  drop reasons (all classes):")
        for reason in DROP_REASONS:
            c = total_reasons.get(reason, 0)
            if c:
                print(f"    {reason:<22} {c:>4}   ({REASON_BLURB[reason]})")

    # The safety-critical line: shout about any leakage hits.
    leak_total = total_reasons.get("leak_heldout", 0)
    if leak_total:
        print("\n  *** LEAKAGE GUARD TRIPPED ***")
        print(f"  {leak_total} staged image(s) duplicated the FROZEN held-out set "
              f"and were dropped.")
        print("  This is expected and good: it kept the held-out set honest.")
    return {
        "per_class_total": dict(per_class_total),
        "per_class_kept": dict(per_class_kept),
        "per_class_drop_reasons": {c: dict(v) for c, v in per_class_reasons.items()},
        "total_drop_reasons": dict(total_reasons),
        "grand_total": grand_total,
        "grand_kept": grand_kept,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="QA + dedup + leakage-guard for staged LeafAlert images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--staged", required=True,
                    help="Staging root with per-class subdirs of NEW images.")
    ap.add_argument("--pool-dir", default=str(DEFAULT_TRAIN_POOL),
                    help="Train pool root (class subfolders). Commit target.")
    ap.add_argument("--heldout-dir", default=str(DEFAULT_HELDOUT),
                    help="Frozen held-out set. Guard against; NEVER commit into.")
    ap.add_argument("--commit", action="store_true",
                    help="Copy survivors into the train pool. Default: dry-run.")
    ap.add_argument("--tag", default=None,
                    help="Source tag for committed filenames (default: staging dir name).")

    # Quality thresholds
    ap.add_argument("--max-megapixels", type=float, default=20.0,
                    help="Originals above this are downsized on commit (not dropped).")
    ap.add_argument("--downsize-long-edge", type=int, default=2048,
                    help="Long-edge cap applied to oversized originals on commit.")
    ap.add_argument("--min-side", type=int, default=128,
                    help="Minimum length (px) of the shorter image side.")
    ap.add_argument("--min-aspect", type=float, default=0.4,
                    help="Minimum width/height aspect ratio.")
    ap.add_argument("--max-aspect", type=float, default=2.5,
                    help="Maximum width/height aspect ratio.")
    ap.add_argument("--sat-thresh", type=float, default=0.04,
                    help="Min mean HSV saturation (0-1); below => greyscale drop.")

    # Dedup / leakage thresholds
    ap.add_argument("--dup-hamming", type=int, default=5,
                    help="Max Hamming distance for within-batch near-dups.")
    ap.add_argument("--leak-hamming", type=int, default=5,
                    help="Max Hamming distance vs reference (held-out + pool). "
                         "Task contract: <=5.")
    ap.add_argument("--keep-crossclass", action="store_true",
                    help="Keep both sides of across-class near-dup collisions "
                         "(default: drop both as label-ambiguous).")

    ap.add_argument("--manifest", default=None,
                    help="Where to write the commit manifest JSON "
                         "(default: <staged>/qa_commit_manifest.json).")
    ap.add_argument("--json", default=None,
                    help="Optional path to dump the full machine-readable report.")
    args = ap.parse_args()

    staged_root = Path(args.staged)
    pool_root = Path(args.pool_dir)
    heldout_root = Path(args.heldout_dir)

    if not staged_root.is_dir():
        print(f"ERROR: staging dir not found: {staged_root}", file=sys.stderr)
        sys.exit(1)
    if args.min_aspect <= 0 or args.max_aspect <= 0 or args.min_aspect >= args.max_aspect:
        print("ERROR: require 0 < --min-aspect < --max-aspect", file=sys.stderr)
        sys.exit(1)

    tag = args.tag or staged_root.resolve().name
    # Sanitise tag for filenames.
    tag = "".join(ch if (ch.isalnum() or ch in "-") else "_" for ch in tag) or "staged"

    class_dirs, extras = staged_class_dirs(staged_root)
    if not class_dirs:
        print(f"ERROR: no canonical class subdirs found under {staged_root}.\n"
              f"Expected one or more of: {', '.join(CANONICAL_CLASSES)}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Staging root : {staged_root}")
    print(f"Train pool   : {pool_root}   (commit target)")
    print(f"Held-out set : {heldout_root}   (guard only, never write)")
    print(f"Mode         : {'COMMIT' if args.commit else 'DRY-RUN'}")
    print(f"Source tag   : {tag}")

    # If the held-out dir is nested inside the pool root, exclude it from the
    # pool scan so we don't scan those images twice (and so 'pool' vs 'heldout'
    # origin tags stay meaningful).
    exclude_from_pool = set()
    try:
        if heldout_root.resolve().parent == pool_root.resolve():
            exclude_from_pool.add(heldout_root.name)
    except OSError:
        pass

    t0 = time.time()
    print("\nIndexing reference set (held-out + train pool)...")
    ref = build_reference_index(pool_root, heldout_root, exclude_names=exclude_from_pool)
    print(f"  indexed {ref['stats']['heldout']} held-out + {ref['stats']['pool']} pool "
          f"images in {time.time() - t0:.1f}s")

    print("Scanning staged images (quality filters + hashing)...")
    records = scan_staged(class_dirs, args.min_side, args.min_aspect,
                          args.max_aspect, args.sat_thresh,
                          max_pixels=int(args.max_megapixels * 1e6))

    # LEAKAGE GUARD: applied to every image that passed the quality gate.
    for r in records:
        if r["drop"] is not None:
            continue
        m = leak_match(r, ref, args.leak_hamming)
        if m is not None:
            origin, kind = m
            r["drop"] = "leak_heldout" if origin.startswith("heldout:") else "leak_pool"
            r["drop_detail"] = f"{kind} match to {origin}"

    # De-dup among the images that survived quality + leakage.
    dedup_staged(records, args.dup_hamming, args.keep_crossclass)

    summary = print_report(records, ref["stats"], extras, args)

    survivors = [r for r in records if r["drop"] is None]

    # ── Commit ────────────────────────────────────────────────────────────
    staging_prov = load_staging_manifest(staged_root)
    manifest_path = Path(args.manifest) if args.manifest else \
        (staged_root / "qa_commit_manifest.json")

    hr("COMMIT")
    if not survivors:
        print("  Nothing to commit (0 survivors).")
    elif not args.commit:
        print(f"  DRY-RUN: {len(survivors)} image(s) WOULD be copied into "
              f"{pool_root}/<class>/")
        print(f"  Filenames: qa_{tag}_<class>_<NNNN>.<ext>")
        print("  Re-run with --commit to write them (and the manifest).")
        # Still show a preview mapping for the first few.
        for r in survivors[:5]:
            print(f"    {r['cls']}/{Path(r['path']).name}  ->  {pool_root.name}/{r['cls']}/qa_{tag}_{r['cls']}_XXXX")
        if len(survivors) > 5:
            print(f"    ... and {len(survivors) - 5} more")
    else:
        entries = commit_survivors(survivors, pool_root, tag, staging_prov, dry_run=False,
                                       downsize_long_edge=args.downsize_long_edge)
        manifest = {
            "tool": "dataset_qa.py",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "staged_root": str(staged_root),
            "pool_root": str(pool_root),
            "source_tag": tag,
            "thresholds": {
                "min_side": args.min_side,
                "min_aspect": args.min_aspect,
                "max_aspect": args.max_aspect,
                "sat_thresh": args.sat_thresh,
                "dup_hamming": args.dup_hamming,
                "leak_hamming": args.leak_hamming,
                "keep_crossclass": args.keep_crossclass,
            },
            "committed_count": len(entries),
            "entries": entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  Committed {len(entries)} image(s) into {pool_root}/<class>/")
        print(f"  Manifest : {manifest_path}")

        # Propagate the source observation id into a manifest train_v5.py reads
        # (TrainingData/manifest.jsonl). This is what makes the train/val split
        # observation-disjoint for fetched data instead of falling back to
        # filename-hash sharding. One JSONL row per committed image, keyed by the
        # committed destination filename (which train_v5 matches by basename).
        pool_manifest = pool_root / "manifest.jsonl"
        n_obs = 0
        with open(pool_manifest, "a") as pfh:
            for e in entries:
                obs = e.get("observation_id")
                if obs is None:
                    continue  # no provenance -> train_v5 falls back for this file
                pfh.write(json.dumps({
                    "path": e["dest"],            # "<class>/qa_<tag>_<cls>_NNNN.ext"
                    "observation_id": obs,
                    "class": e["class"],
                    "source": e.get("source"),
                }) + "\n")
                n_obs += 1
        if n_obs:
            print(f"  Provenance: wrote {n_obs} observation-id row(s) -> "
                  f"{pool_manifest} (makes train_v5 split observation-disjoint)")
        else:
            print("  Provenance: no observation ids available; train_v5 will "
                  "fall back to filename-hash sharding for these images.")
        missing_lic = sum(1 for e in entries if not e.get("license"))
        if missing_lic:
            print(f"  WARNING: {missing_lic}/{len(entries)} committed images have NO "
                  f"license/attribution recorded (no staging manifest?). "
                  f"Add provenance before publishing/distributing.")

    # ── Optional full JSON report ─────────────────────────────────────────
    if args.json:
        # Strip the big int hashes to keep the JSON portable/small.
        recs_out = []
        for r in records:
            recs_out.append({k: v for k, v in r.items()
                             if k not in ("dhash", "ahash")})
        Path(args.json).write_text(json.dumps({
            "summary": summary,
            "reference_stats": ref["stats"],
            "thresholds": vars(args),
            "records": recs_out,
        }, indent=2, default=str))
        print(f"\nFull report written to {args.json}")

    print()  # trailing newline


if __name__ == "__main__":
    main()
