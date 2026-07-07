#!/usr/bin/env python3
"""
Empirical audit of the LeafAlert image classification dataset.

Audits TrainingData/ (and an optional held-out set such as TrainingData/Testing)
for the things that silently wreck a small vision classifier:

  * exact per-class counts and class imbalance
  * corrupt / unreadable / truncated image files
  * image resolution and aspect-ratio distribution
  * exact-duplicate images (SHA-256 of decoded pixels)
  * near-duplicate images (perceptual dHash + aHash, Hamming distance)
      - WITHIN a class  (redundant / augmentation-leak candidates)
      - ACROSS classes  (label-conflict / contamination candidates)
      - TRAIN <-> TEST   (train/test LEAKAGE -> inflated eval metrics)
  * source/background diversity signal from filename prefixes
  * per-class spot-check manifest (writes a few sample paths per class so a
    human can eyeball label correctness)

No third-party hashing dependency: perceptual hashes are implemented with
PIL + numpy only, so this runs in the project's .venv as-is.

This is intended to be durable, committed tooling. It only READS images;
it never modifies the dataset.

Usage:
    python3 scripts/audit_dataset.py
    python3 scripts/audit_dataset.py --data-dir TrainingData --test-dir TrainingData/Testing
    python3 scripts/audit_dataset.py --hamming 6 --json report.json
    python3 scripts/audit_dataset.py --no-crossclass   # skip the O(n^2) cross-class scan
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

# Be strict: we WANT to detect truncated files rather than silently pad them.
ImageFile.LOAD_TRUNCATED_IMAGES = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical class order (ImageFolder-alphabetical). safe_plants is the negative class.
CANONICAL_CLASSES = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]
TOXIC_CLASSES = {"poison_ivy", "poison_oak", "poison_sumac"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual hashing (dependency-free)
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
# Scanning
# ─────────────────────────────────────────────────────────────────────────────
def list_class_dirs(root, exclude=None):
    """Return {class_name: Path} for immediate subdirectories that look like classes.

    `exclude` is a set of directory names to skip (e.g. a nested held-out split
    like TrainingData/Testing, which must NOT be treated as a class).
    """
    exclude = exclude or set()
    out = {}
    for p in sorted(root.iterdir()):
        if p.is_dir() and not p.name.startswith(".") and p.name not in exclude:
            out[p.name] = p
    return out


def list_images(class_dir):
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    )


def scan_split(root, split_name, compute_hashes=True, exclude=None):
    """
    Walk one split (e.g. TrainingData or TrainingData/Testing).
    Returns dict with per-file records and per-class aggregates.
    """
    class_dirs = list_class_dirs(root, exclude=exclude)
    records = []  # list of dicts
    corrupt = []
    counts = {}
    for cls, cdir in class_dirs.items():
        files = list_images(cdir)
        counts[cls] = len(files)
        for f in files:
            rec = {
                "split": split_name,
                "cls": cls,
                "path": str(f),
                "name": f.name,
            }
            try:
                with Image.open(f) as im:
                    im.load()  # force full decode -> catches truncation
                    im = im.convert("RGB")
                rec["w"], rec["h"] = im.size
                rec["mode_ok"] = True
                if compute_hashes:
                    rec["dhash"] = dhash(im)
                    rec["ahash"] = ahash(im)
                    # exact-content hash of decoded, resized-to-native pixels
                    rec["sha"] = hashlib.sha256(
                        np.asarray(im, dtype=np.uint8).tobytes()
                    ).hexdigest()
                records.append(rec)
            except Exception as e:  # noqa: BLE001 - we want every failure mode
                rec["error"] = f"{type(e).__name__}: {e}"
                corrupt.append(rec)
    return {"root": str(root), "split": split_name, "counts": counts,
            "records": records, "corrupt": corrupt, "classes": list(class_dirs)}


def source_prefixes(records):
    """
    Group filenames by their non-numeric prefix (e.g. blackberry_0007.jpg -> blackberry).
    Signals source/sub-taxon diversity inside a class.
    """
    per_class = defaultdict(lambda: defaultdict(int))
    for r in records:
        stem = Path(r["name"]).stem
        # strip a trailing _<digits> (and an optional _test) to get the source token
        parts = stem.split("_")
        while parts and (parts[-1].isdigit() or parts[-1] == "test"):
            parts.pop()
        prefix = "_".join(parts) if parts else "(none)"
        per_class[r["cls"]][prefix] += 1
    return {c: dict(sorted(v.items(), key=lambda kv: -kv[1])) for c, v in per_class.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate / near-duplicate detection
# ─────────────────────────────────────────────────────────────────────────────
def find_exact_dupes(records):
    """Exact decoded-pixel duplicates (SHA-256). Returns list of groups (>=2)."""
    by_sha = defaultdict(list)
    for r in records:
        if "sha" in r:
            by_sha[r["sha"]].append(r)
    return [grp for grp in by_sha.values() if len(grp) > 1]


def find_near_dupes(records, threshold):
    """
    Brute-force near-duplicate pairs via dHash Hamming distance (confirmed by aHash).
    O(n^2) but n<=~1600 here so it's a few seconds. Returns pairs below threshold.
    A pair is reported if BOTH dhash and ahash distances are <= threshold, which
    strongly cuts false positives from a single hash family.
    """
    recs = [r for r in records if "dhash" in r]
    pairs = []
    n = len(recs)
    for i in range(n):
        di, ai = recs[i]["dhash"], recs[i]["ahash"]
        for j in range(i + 1, n):
            dd = hamming(di, recs[j]["dhash"])
            if dd <= threshold:
                da = hamming(ai, recs[j]["ahash"])
                if da <= threshold:
                    pairs.append((recs[i], recs[j], dd, da))
    return pairs


def summarize_pairs(pairs):
    within = defaultdict(int)
    cross = defaultdict(int)
    cross_pairs = []
    for a, b, dd, da in pairs:
        if a["split"] == b["split"] and a["cls"] == b["cls"]:
            within[a["cls"]] += 1
        elif a["split"] == b["split"] and a["cls"] != b["cls"]:
            key = tuple(sorted((a["cls"], b["cls"])))
            cross[key] += 1
            cross_pairs.append((a, b, dd, da))
    return dict(within), {"/".join(k): v for k, v in cross.items()}, cross_pairs


def find_leakage(train_records, test_records, threshold):
    """
    Cross-split near/exact-dupe detection: any test image that closely matches a
    train image is leakage (inflates held-out metrics). Reports exact + near.
    """
    # exact
    train_sha = defaultdict(list)
    for r in train_records:
        if "sha" in r:
            train_sha[r["sha"]].append(r)
    exact = []
    for r in test_records:
        if r.get("sha") in train_sha:
            exact.append((r, train_sha[r["sha"]][0]))

    # near (only for test files not already exact-matched)
    exact_test_paths = {t["path"] for t, _ in exact}
    tr = [r for r in train_records if "dhash" in r]
    near = []
    for tst in test_records:
        if "dhash" not in tst or tst["path"] in exact_test_paths:
            continue
        for trn in tr:
            dd = hamming(tst["dhash"], trn["dhash"])
            if dd <= threshold and hamming(tst["ahash"], trn["ahash"]) <= threshold:
                near.append((tst, trn, dd))
                break  # one match is enough to flag leakage
    return exact, near


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def pct(a, b):
    return (100.0 * a / b) if b else 0.0


def print_header(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Audit the LeafAlert image dataset.")
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "TrainingData"),
                    help="Root training data dir (class subfolders).")
    ap.add_argument("--test-dir", default=str(PROJECT_ROOT / "TrainingData" / "Testing"),
                    help="Optional held-out/test dir with the same class subfolders.")
    ap.add_argument("--hamming", type=int, default=6,
                    help="Max Hamming distance for near-duplicate (default 6 of 64 bits).")
    ap.add_argument("--no-crossclass", action="store_true",
                    help="Skip the O(n^2) within-split near-dupe scan.")
    ap.add_argument("--spot-samples", type=int, default=5,
                    help="How many sample paths per class to emit for label spot-checks.")
    ap.add_argument("--json", default=None, help="Optional path to dump machine-readable report.")
    args = ap.parse_args()

    data_root = Path(args.data_dir)
    test_root = Path(args.test_dir) if args.test_dir else None
    if not data_root.is_dir():
        print(f"ERROR: data dir not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    # If the held-out dir is nested INSIDE the training root, exclude it from the
    # train scan so it isn't mistaken for a class (and warn: naive ImageFolder
    # ingestion of data_root would otherwise swallow the test split).
    nested_test = None
    if test_root and test_root.is_dir() and test_root.parent.resolve() == data_root.resolve():
        nested_test = test_root.name

    print(f"Auditing: {data_root}")
    train = scan_split(data_root, "train", exclude={nested_test} if nested_test else None)
    if nested_test:
        print(f"  NOTE: held-out '{nested_test}' is NESTED inside the training root; "
              f"excluded from train scan. A naive ImageFolder({data_root.name}) would "
              f"ingest it as a 5th class / contaminate training.")

    test = None
    if test_root and test_root.is_dir():
        print(f"Auditing held-out: {test_root}")
        test = scan_split(test_root, "test")

    report = {"data_dir": str(data_root), "hamming_threshold": args.hamming}

    # ── Counts & imbalance ───────────────────────────────────────────────
    print_header("PER-CLASS COUNTS (readable images)")
    ok_counts = defaultdict(int)
    for r in train["records"]:
        ok_counts[r["cls"]] += 1
    total = sum(ok_counts.values())
    for c in sorted(train["counts"]):
        raw = train["counts"][c]
        okc = ok_counts.get(c, 0)
        flag = "  [TOXIC]" if c in TOXIC_CLASSES else "  [safe/negative]"
        print(f"  {c:<14} readable={okc:>4}  raw_files={raw:>4}  ({pct(okc,total):5.1f}%){flag}")
    if ok_counts:
        mx, mn = max(ok_counts.values()), min(ok_counts.values())
        print(f"  TOTAL readable: {total}")
        print(f"  Imbalance ratio (max/min class): {mx/mn:.2f}  ({mx} / {mn})")
        toxic_n = sum(v for c, v in ok_counts.items() if c in TOXIC_CLASSES)
        safe_n = total - toxic_n
        print(f"  Toxic vs safe (binary framing): {toxic_n} toxic / {safe_n} safe "
              f"= {toxic_n/safe_n:.2f}:1" if safe_n else "")
    report["train_counts"] = dict(ok_counts)

    if test:
        print_header("HELD-OUT (test) PER-CLASS COUNTS")
        tcounts = defaultdict(int)
        for r in test["records"]:
            tcounts[r["cls"]] += 1
        for c in sorted(test["counts"]):
            print(f"  {c:<14} readable={tcounts.get(c,0):>4}  raw_files={test['counts'][c]:>4}")
        print(f"  TOTAL readable held-out: {sum(tcounts.values())}")
        report["test_counts"] = dict(tcounts)

    # ── Corrupt files ────────────────────────────────────────────────────
    print_header("CORRUPT / UNREADABLE FILES")
    all_corrupt = train["corrupt"] + (test["corrupt"] if test else [])
    if not all_corrupt:
        print("  None. All files decoded fully (truncation detection ON).")
    else:
        for r in all_corrupt:
            print(f"  [{r['split']}/{r['cls']}] {r['name']}: {r['error']}")
    report["corrupt"] = [{"path": r["path"], "error": r["error"]} for r in all_corrupt]

    # ── Resolution / aspect ratio ────────────────────────────────────────
    print_header("RESOLUTION & ASPECT-RATIO DISTRIBUTION (train)")
    ws = np.array([r["w"] for r in train["records"]])
    hs = np.array([r["h"] for r in train["records"]])
    ar = ws / np.maximum(hs, 1)
    def stats(a):
        return (f"min={a.min():.0f} p10={np.percentile(a,10):.0f} "
                f"median={np.median(a):.0f} p90={np.percentile(a,90):.0f} max={a.max():.0f}")
    print(f"  width : {stats(ws)}")
    print(f"  height: {stats(hs)}")
    print(f"  aspect(w/h): min={ar.min():.2f} median={np.median(ar):.2f} max={ar.max():.2f}")
    uniq_dims = defaultdict(int)
    for r in train["records"]:
        uniq_dims[(r["w"], r["h"])] += 1
    top_dims = sorted(uniq_dims.items(), key=lambda kv: -kv[1])[:8]
    print(f"  distinct (w,h) sizes: {len(uniq_dims)}")
    print("  most common sizes: " + ", ".join(f"{w}x{h}:{n}" for (w, h), n in top_dims))
    # count very-off-square images (aspect >2 or <0.5) -> heavy squash under .scaleFill
    extreme = int(((ar > 2.0) | (ar < 0.5)).sum())
    print(f"  extreme aspect (>2:1 or <1:2), squashed hard by .scaleFill/Resize(224,224): "
          f"{extreme} ({pct(extreme, len(ar)):.1f}%)")
    report["resolution"] = {
        "width": {"min": int(ws.min()), "median": float(np.median(ws)), "max": int(ws.max())},
        "height": {"min": int(hs.min()), "median": float(np.median(hs)), "max": int(hs.max())},
        "distinct_sizes": len(uniq_dims),
        "extreme_aspect": extreme,
    }

    # ── Source diversity ─────────────────────────────────────────────────
    print_header("SOURCE / SUB-TAXON DIVERSITY (filename-prefix signal, train)")
    prefixes = source_prefixes(train["records"])
    for c in sorted(prefixes):
        subs = prefixes[c]
        shown = ", ".join(f"{k}={v}" for k, v in list(subs.items())[:8])
        print(f"  {c:<14} {len(subs)} prefix group(s): {shown}")
    report["source_prefixes"] = prefixes

    # ── Exact duplicates (within train) ──────────────────────────────────
    print_header("EXACT-DUPLICATE IMAGES (decoded-pixel SHA-256, train)")
    exact_groups = find_exact_dupes(train["records"])
    n_exact_extra = sum(len(g) - 1 for g in exact_groups)
    if not exact_groups:
        print("  None found.")
    else:
        print(f"  {len(exact_groups)} duplicate group(s), {n_exact_extra} redundant copies.")
        cross_cls_exact = 0
        for g in exact_groups[:20]:
            clss = {r["cls"] for r in g}
            tag = "  <-- CROSS-CLASS (label conflict!)" if len(clss) > 1 else ""
            if len(clss) > 1:
                cross_cls_exact += 1
            print(f"    group({len(g)}) {[r['name'] for r in g]} classes={sorted(clss)}{tag}")
        if cross_cls_exact:
            print(f"  >>> {cross_cls_exact} exact-dup group(s) span MULTIPLE classes.")
    report["exact_dupe_groups"] = len(exact_groups)
    report["exact_dupe_redundant"] = n_exact_extra

    # ── Near duplicates (within splits) ──────────────────────────────────
    if not args.no_crossclass:
        print_header(f"NEAR-DUPLICATE IMAGES (dHash & aHash <= {args.hamming} bits, train)")
        pairs = find_near_dupes(train["records"], args.hamming)
        within, cross, cross_pairs = summarize_pairs(pairs)
        print(f"  total near-dupe pairs (excluding exact): {len(pairs)}")
        print("  WITHIN-class near-dupe pairs (redundancy):")
        for c in sorted(within):
            print(f"    {c:<14} {within[c]} pair(s)")
        print("  ACROSS-class near-dupe pairs (possible label contamination):")
        if not cross:
            print("    none")
        for k, v in sorted(cross.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<28} {v} pair(s)")
        for a, b, dd, da in cross_pairs[:15]:
            print(f"      d={dd},a={da}: [{a['cls']}] {a['name']}  <->  [{b['cls']}] {b['name']}")
        report["near_dupe_pairs"] = len(pairs)
        report["near_dupe_within"] = within
        report["near_dupe_cross"] = cross

    # ── Leakage (train <-> test) ─────────────────────────────────────────
    if test:
        print_header(f"TRAIN<->TEST LEAKAGE (exact + near dHash/aHash <= {args.hamming})")
        exact_leak, near_leak = find_leakage(train["records"], test["records"], args.hamming)
        print(f"  EXACT leak (identical pixels in both splits): {len(exact_leak)}")
        for t, tr in exact_leak[:15]:
            same = " (same class)" if t["cls"] == tr["cls"] else " (DIFF class!)"
            print(f"    test[{t['cls']}] {t['name']}  ==  train[{tr['cls']}] {tr['name']}{same}")
        print(f"  NEAR leak (perceptually near-identical across splits): {len(near_leak)}")
        for t, tr, dd in near_leak[:15]:
            same = " (same class)" if t["cls"] == tr["cls"] else " (DIFF class!)"
            print(f"    d={dd}: test[{t['cls']}] {t['name']}  ~  train[{tr['cls']}] {tr['name']}{same}")
        n_test = len(test["records"])
        leaked = len(exact_leak) + len(near_leak)
        print(f"  >>> {leaked}/{n_test} held-out images leak from train "
              f"({pct(leaked, n_test):.1f}%).")
        report["leak_exact"] = len(exact_leak)
        report["leak_near"] = len(near_leak)
        report["test_size"] = n_test

    # ── Spot-check manifest ──────────────────────────────────────────────
    print_header(f"LABEL SPOT-CHECK MANIFEST ({args.spot_samples} samples/class, train)")
    print("  (open these to eyeball label correctness / off-topic images)")
    by_cls = defaultdict(list)
    for r in train["records"]:
        by_cls[r["cls"]].append(r)
    spot = {}
    for c in sorted(by_cls):
        recs = by_cls[c]
        idxs = np.linspace(0, len(recs) - 1, min(args.spot_samples, len(recs))).astype(int)
        sample = [recs[i] for i in idxs]
        spot[c] = [s["path"] for s in sample]
        print(f"  {c}:")
        for s in sample:
            print(f"    {s['w']}x{s['h']}  {s['path']}")
    report["spot_samples"] = spot

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nWrote machine-readable report to {args.json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
