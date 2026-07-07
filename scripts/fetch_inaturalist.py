#!/usr/bin/env python3
"""
Fetch CC-licensed, research-grade iNaturalist photos for LeafAlert's target
taxa into a staging directory, with full provenance/license/attribution in a
manifest. This EXPANDS the training pool with real toxic images and, crucially,
real look-alike HARD NEGATIVES for the safe_plants bucket.

WHY THIS EXISTS
    The model has a severe quality gap (held-out toxic-recall 43.5% @0.65,
    poison_ivy recall 51%, motion-blur flips 58% of toxic->safe) driven partly
    by ~1400 training images with NO look-alike hard negatives. This script
    pulls stratified, license-clean data to fix that. It is the FETCH stage
    only; it stages into data_staging/ and never touches the frozen held-out
    set or TrainingData/ directly.

CONTRACTS HONORED (see project MEMORY / CLAUDE.md)
    * Classes (ImageFolder-alphabetical, immutable):
        poison_ivy, poison_oak, poison_sumac, safe_plants
      Toxic = first three; safe_plants is the negative / look-alike bucket.
    * Taxon -> class mapping:
        Toxicodendron radicans / rydbergii  -> poison_ivy
        Toxicodendron diversilobum / pubescens -> poison_oak
        Toxicodendron vernix                -> poison_sumac
        Parthenocissus quinquefolia, Acer negundo, Rubus (blackberry/raspberry),
        Rhus aromatica, Rhus glabra, Rhus typhina -> safe_plants (hard negatives)
    * HELD-OUT FREEZE: TrainingData/Testing (n=362) is frozen and leakage-free.
      This script NEVER writes there. With --dedup-heldout (default ON) it also
      perceptually de-duplicates every fetched photo against that held-out set
      so no fetched image can leak into it later.
    * Only CC-licensed photos: default --license cc0,cc-by (add cc-by-nc if you
      accept NonCommercial). Attribution + license + photographer recorded per
      image in the manifest so downstream can credit sources.

WHAT IT DOES
    * Resolves each taxon to its iNaturalist taxon id (verified table below;
      re-verify live with --verify-taxa).
    * Pages the observations endpoint (research grade, photos, photo_license
      filter), stratifiable by month(s), place/region.
    * Downloads the LARGE photo variant (swaps /square. -> /large.), preserving
      the real file extension.
    * Appends one JSONL row per image to data_staging/inaturalist/manifest.jsonl
      with: photo id, observation id, taxon name/id, class, license, attribution,
      photographer login/name, observed date, lat/lng, url, saved path, dims.

IDEMPOTENT / RESUMABLE
    Photo ids already in the manifest (or already on disk) are skipped, so you
    can re-run to top up per-taxon counts without re-downloading.

POLITE
    Rate-limit sleeps between API pages and image downloads, retries with
    exponential backoff, sane User-Agent.

DEPENDENCIES
    Stdlib urllib only (no `requests`) plus PIL + numpy (already in the venv).
    Deliberately dependency-free on HTTP, matching audit_dataset.py.

USAGE
    # Smoke test: 3 images from one taxon, dry run of the rest
    python3 scripts/fetch_inaturalist.py --taxon "Toxicodendron radicans" --limit 3

    # Fetch 200 poison_ivy photos (all its taxa), US only, fall months
    python3 scripts/fetch_inaturalist.py --class poison_ivy --per-taxon 100 \
        --place-id 1 --months 9,10,11

    # Fetch everything, 150 per taxon, cc0+cc-by
    python3 scripts/fetch_inaturalist.py --all --per-taxon 150

    # Just re-verify the taxon id table against the live API and exit
    python3 scripts/fetch_inaturalist.py --verify-taxa
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

# Detect truncated downloads rather than silently padding them.
ImageFile.LOAD_TRUNCATED_IMAGES = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical class order (ImageFolder-alphabetical). safe_plants is the negative.
CANONICAL_CLASSES = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]
TOXIC_CLASSES = {"poison_ivy", "poison_oak", "poison_sumac"}

HELDOUT_DIR = PROJECT_ROOT / "TrainingData" / "Testing"
DEFAULT_OUT = PROJECT_ROOT / "data_staging" / "inaturalist"

API_OBS = "https://api.inaturalist.org/v1/observations"
API_TAXA = "https://api.inaturalist.org/v1/taxa"
USER_AGENT = "LeafAlert-fetch/1.0 (safety classifier dataset builder)"

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ─────────────────────────────────────────────────────────────────────────────
# Taxon table. IDs verified live against the iNaturalist taxa endpoint
# (2026-07-07). Each entry maps an iNat taxon id to one of the four classes and
# to a short filename token. `taxon_id` on the observations endpoint matches the
# taxon AND its descendants (subspecies), which is what we want.
# Re-verify any time with:  python3 scripts/fetch_inaturalist.py --verify-taxa
# ─────────────────────────────────────────────────────────────────────────────
TAXA = [
    # token,             taxon_id, scientific name,                cls
    ("poison_ivy_east",  58732, "Toxicodendron radicans",       "poison_ivy"),
    ("poison_ivy_west",  58729, "Toxicodendron rydbergii",      "poison_ivy"),
    ("poison_oak_pac",   51080, "Toxicodendron diversilobum",   "poison_oak"),
    ("poison_oak_atl",   52083, "Toxicodendron pubescens",      "poison_oak"),
    ("poison_sumac",     54767, "Toxicodendron vernix",         "poison_sumac"),
    # ---- hard-negative look-alikes -> safe_plants ----
    ("virginia_creeper", 50278, "Parthenocissus quinquefolia",  "safe_plants"),
    ("box_elder",        47726, "Acer negundo",                 "safe_plants"),
    ("blackberry",       47544, "Rubus (blackberry/raspberry)", "safe_plants"),
    ("fragrant_sumac",   58738, "Rhus aromatica",               "safe_plants"),
    ("smooth_sumac",     54764, "Rhus glabra",                  "safe_plants"),
    ("staghorn_sumac",  167829, "Rhus typhina",                 "safe_plants"),
]
# Note: Rubus 47544 is the plant genus (brambles: blackberries+raspberries);
# using the genus id pulls the whole look-alike bramble group, which is
# intentional for hard negatives. taxon_id matches descendants too.

# Named US regions -> iNaturalist place_id (place_id=1 == United States).
REGIONS = {
    "us": 1,
    "usa": 1,
    "united_states": 1,
    "north_america": 97394,
}


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual hashing (dependency-free) — same dHash used by audit_dataset.py so
# held-out de-dup is consistent with the audit tooling.
# ─────────────────────────────────────────────────────────────────────────────
def _gray_small(img, size):
    return np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)


def dhash(img, size=8):
    px = _gray_small(img, size + 1)
    diff = px[:, 1:] > px[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(b)
    return val


def hamming(a, b):
    return bin(a ^ b).count("1")


# ─────────────────────────────────────────────────────────────────────────────
# HTTP (stdlib urllib, with retries)
# ─────────────────────────────────────────────────────────────────────────────
def _http_get(url, params=None, retries=4, timeout=30, sleep=1.0):
    """GET with query params; retry with exponential backoff. Returns bytes."""
    if params:
        # Drop None-valued params; encode the rest.
        clean = {k: v for k, v in params.items() if v is not None and v != ""}
        url = url + "?" + urllib.parse.urlencode(clean)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001 - network flakiness; back off and retry
            last_err = e
            wait = sleep * (2 ** attempt)
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {retries} tries: {url} :: {last_err}")


def _api_json(url, params, retries=4):
    raw = _http_get(url, params=params, retries=retries)
    return json.loads(raw.decode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Taxon verification
# ─────────────────────────────────────────────────────────────────────────────
def verify_taxa():
    """Query the live taxa endpoint and confirm each hardcoded id resolves to
    the expected scientific name. Prints a table; returns True if all match."""
    print("Verifying taxon ids against the live iNaturalist taxa endpoint...\n")
    print(f"  {'token':<18} {'id':>7}  {'expected':<32} live_name / match")
    all_ok = True
    for token, tid, sci, cls in TAXA:
        try:
            data = _api_json(API_TAXA, {"id": tid})
            results = data.get("results", [])
            live = results[0].get("name") if results else "(not found)"
        except Exception as e:  # noqa: BLE001
            live = f"(error: {e})"
        # For the Rubus genus entry, `sci` is a friendly label; compare loosely.
        expect_base = sci.split(" (")[0]
        ok = live == expect_base or expect_base.startswith(live) or live.startswith(expect_base.split()[0])
        all_ok = all_ok and (ok or "Rubus" in sci)
        flag = "OK" if ok else ("~genus" if "Rubus" in sci else "MISMATCH")
        print(f"  {token:<18} {tid:>7}  {expect_base:<32} {live}  [{flag}] -> {cls}")
        time.sleep(0.5)
    print("\nAll taxa verified." if all_ok else "\nWARNING: some taxa mismatched; update the TAXA table.")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Held-out de-dup index
# ─────────────────────────────────────────────────────────────────────────────
def build_heldout_hashes(heldout_dir):
    """Compute dHash for every image in the frozen held-out set so fetched
    photos that near-match it can be dropped (protects the FREEZE)."""
    hashes = []
    if not heldout_dir.is_dir():
        return hashes
    for cls_dir in sorted(heldout_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() not in IMG_EXTS or f.name.startswith("."):
                continue
            try:
                with Image.open(f) as im:
                    im.load()
                    hashes.append(dhash(im.convert("RGB")))
            except Exception:  # noqa: BLE001 - a bad held-out file shouldn't stop us
                continue
    return hashes


def matches_heldout(img, heldout_hashes, threshold):
    if not heldout_hashes:
        return False
    h = dhash(img)
    for hh in heldout_hashes:
        if hamming(h, hh) <= threshold:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────
def load_manifest(manifest_path):
    """Return set of photo ids already recorded (for resumability)."""
    seen = set()
    if not manifest_path.exists():
        return seen
    with manifest_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                pid = row.get("photo_id")
                if pid is not None:
                    seen.add(int(pid))
            except Exception:  # noqa: BLE001 - skip a corrupt manifest line
                continue
    return seen


def append_manifest(manifest_path, row):
    with manifest_path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Photo URL helpers
# ─────────────────────────────────────────────────────────────────────────────
def large_url(square_url):
    """Swap the iNat 'square' thumbnail for the 'large' (~1024px) variant,
    preserving the real extension (.jpg vs .jpeg vs .png)."""
    for sz in ("/square.", "/small.", "/medium.", "/thumb."):
        if sz in square_url:
            return square_url.replace(sz, "/large.")
    return square_url


def url_ext(url):
    path = urllib.parse.urlparse(url).path
    ext = Path(path).suffix.lower()
    return ext if ext in IMG_EXTS else ".jpg"


# ─────────────────────────────────────────────────────────────────────────────
# Fetch one taxon
# ─────────────────────────────────────────────────────────────────────────────
def fetch_taxon(token, taxon_id, sci_name, cls, args, seen_ids, heldout_hashes,
                out_root, manifest_path):
    """Page observations for one taxon, download up to per-taxon photos.
    Returns (n_saved, n_skipped_seen, n_skipped_heldout, n_failed, n_would)."""
    per_taxon = args.per_taxon
    if args.limit is not None:
        per_taxon = min(per_taxon, args.limit)

    cls_dir = out_root / cls
    n_saved = n_seen = n_held = n_fail = n_would = 0
    page = 1
    max_pages = args.max_pages

    # Idempotency: by default --per-taxon is a TOTAL target. Count how many
    # photos for this taxon already exist on disk (files named <token>_*) and
    # only fetch the remainder, so re-running the same command downloads nothing
    # new. With --top-up-new the target is instead "N new photos each run".
    existing = 0
    if cls_dir.is_dir():
        existing = sum(1 for f in cls_dir.iterdir()
                       if f.is_file() and f.name.startswith(token + "_")
                       and f.suffix.lower() in IMG_EXTS)
    remaining = per_taxon if args.top_up_new else max(0, per_taxon - existing)

    print(f"\n{'='*68}")
    print(f"{token}  (taxon {taxon_id}: {sci_name})  -> class '{cls}'")
    mode = "N-new" if args.top_up_new else "total"
    print(f"  target {per_taxon} ({mode}) | already on disk {existing} | "
          f"fetch up to {remaining}")
    print(f"  license={args.license} | place_id={args.place_id} | "
          f"months={args.months or 'any'}")
    if remaining == 0:
        print("  already at target; nothing to do.")
        return 0, 0, 0, 0, 0

    while n_saved + n_would < remaining and page <= max_pages:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "photo_license": args.license,
            # NOTE: we intentionally do NOT also send the observation-level
            # `license` param. We only download and record the PHOTO license
            # (photo.license_code), so filtering on the observation record's
            # license is redundant AND wrongly drops ~10% of otherwise-usable
            # CC-licensed photos (verified live: e.g. poison ivy US 10,034 with
            # both filters vs 11,133 with photo_license alone).
            "place_id": args.place_id,
            "month": args.months,
            "per_page": min(200, args.per_page),
            "page": page,
            "order": "desc",
            "order_by": "votes",
        }
        try:
            data = _api_json(API_OBS, params)
        except Exception as e:  # noqa: BLE001
            print(f"  API error on page {page}: {e}")
            break

        results = data.get("results", [])
        if page == 1:
            print(f"  API total_results for this filter: {data.get('total_results')}")
        if not results:
            break

        for obs in results:
            if n_saved + n_would >= remaining:
                break
            obs_id = obs.get("id")
            user = obs.get("user") or {}
            observed_on = obs.get("observed_on")
            location = obs.get("location")  # "lat,lng" or None
            obs_taxon = (obs.get("taxon") or {})
            for photo in obs.get("photos", []):
                if n_saved + n_would >= remaining:
                    break
                pid = photo.get("id")
                if pid is None or int(pid) in seen_ids:
                    n_seen += 1
                    continue
                sq = photo.get("url", "")
                if not sq:
                    continue
                url = large_url(sq)
                ext = url_ext(url)
                save_path = cls_dir / f"{token}_{pid}{ext}"

                # Already on disk from a prior run without a manifest entry?
                if save_path.exists():
                    seen_ids.add(int(pid))
                    n_seen += 1
                    continue

                if args.dry_run:
                    n_would += 1
                    if n_would <= 5:
                        print(f"    [DRY] would fetch photo {pid} "
                              f"({photo.get('license_code')}) -> {save_path.name}")
                    continue

                # Download bytes.
                try:
                    img_bytes = _http_get(url, retries=args.retries, timeout=25,
                                          sleep=args.sleep)
                except Exception as e:  # noqa: BLE001
                    print(f"    download failed photo {pid}: {e}")
                    n_fail += 1
                    # Remember this pid so a resumed run does not re-attempt
                    # the same failing photo (idempotent skip).
                    seen_ids.add(int(pid))
                    continue

                # Validate + hash-check against held-out before committing.
                try:
                    from io import BytesIO
                    with Image.open(BytesIO(img_bytes)) as im:
                        im.load()
                        rgb = im.convert("RGB")
                        w, h = rgb.size
                        if args.dedup_heldout and matches_heldout(
                                rgb, heldout_hashes, args.hamming):
                            n_held += 1
                            # Remember held-out matches so a resumed run does not
                            # re-download and re-hash the same photo every time.
                            seen_ids.add(int(pid))
                            continue
                except Exception as e:  # noqa: BLE001
                    print(f"    corrupt image photo {pid}: {e}")
                    n_fail += 1
                    seen_ids.add(int(pid))
                    continue

                cls_dir.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(img_bytes)

                # Record a path relative to the repo when the staging dir lives
                # inside it (the normal case); otherwise fall back to absolute.
                try:
                    rel_path = str(save_path.resolve().relative_to(PROJECT_ROOT))
                except ValueError:
                    rel_path = str(save_path.resolve())

                row = {
                    "photo_id": int(pid),
                    "observation_id": obs_id,
                    "token": token,
                    "taxon_id": taxon_id,
                    "taxon_name": sci_name,
                    "observed_taxon_id": obs_taxon.get("id"),
                    "observed_taxon_name": obs_taxon.get("name"),
                    "cls": cls,
                    "license": photo.get("license_code"),
                    "attribution": photo.get("attribution"),
                    "photographer_login": user.get("login"),
                    "photographer_name": user.get("name"),
                    "observed_on": observed_on,
                    "location": location,
                    "source": "inaturalist",
                    "url": url,
                    "path": rel_path,
                    "width": w,
                    "height": h,
                }
                append_manifest(manifest_path, row)
                seen_ids.add(int(pid))
                n_saved += 1
                if n_saved <= 3 or n_saved % 25 == 0:
                    print(f"    saved {n_saved}/{remaining}: photo {pid} "
                          f"({row['license']}) {w}x{h}")
                time.sleep(args.sleep)  # polite between image downloads

        page += 1
        if n_saved + n_would < remaining and page <= max_pages:
            time.sleep(max(args.sleep, 1.0))  # polite between API pages

    print(f"  done: saved={n_saved} would={n_would} skipped_seen={n_seen} "
          f"skipped_heldout={n_held} failed={n_fail}")
    return n_saved, n_seen, n_held, n_fail, n_would


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def resolve_place_id(args):
    if args.region:
        key = args.region.strip().lower().replace(" ", "_")
        if key in REGIONS:
            return REGIONS[key]
        print(f"WARNING: unknown --region '{args.region}'. Known: "
              f"{', '.join(sorted(REGIONS))}. Ignoring.", file=sys.stderr)
    return args.place_id


def select_taxa(args):
    """Return the list of TAXA rows to process based on --class / --taxon / --all."""
    if args.taxon:
        want = args.taxon.strip().lower()
        rows = [t for t in TAXA
                if want in t[2].lower() or want == t[0].lower()]
        if not rows:
            print(f"ERROR: no taxon matched '{args.taxon}'. Options: "
                  + ", ".join(t[2] for t in TAXA), file=sys.stderr)
            sys.exit(2)
        return rows
    if args.taxon_id is not None:
        rows = [t for t in TAXA if t[1] == args.taxon_id]
        if not rows:
            # allow an ad-hoc id if class is specified
            if args.cls:
                return [(f"taxon_{args.taxon_id}", args.taxon_id,
                         f"(ad-hoc {args.taxon_id})", args.cls)]
            print(f"ERROR: --taxon-id {args.taxon_id} not in table; also pass "
                  f"--class to map it.", file=sys.stderr)
            sys.exit(2)
        return rows
    if args.cls:
        rows = [t for t in TAXA if t[3] == args.cls]
        if not rows:
            print(f"ERROR: no taxa for class '{args.cls}'.", file=sys.stderr)
            sys.exit(2)
        return rows
    if args.all:
        return list(TAXA)
    print("ERROR: choose one of --class, --taxon, --taxon-id, or --all "
          "(or --verify-taxa).", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(
        description="Fetch CC-licensed research-grade iNaturalist photos for "
                    "LeafAlert's target taxa into a staging dir with a provenance "
                    "manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sel = ap.add_argument_group("selection")
    sel.add_argument("--class", dest="cls", choices=CANONICAL_CLASSES,
                     help="Fetch every taxon mapped to this class.")
    sel.add_argument("--all", action="store_true",
                     help="Fetch every taxon in the table.")
    sel.add_argument("--taxon", help="Fetch one taxon by scientific name or token "
                                     "(substring match), e.g. 'Toxicodendron radicans'.")
    sel.add_argument("--taxon-id", type=int, default=None,
                     help="Fetch one taxon by iNat id (use with --class for ad-hoc ids).")

    fetch = ap.add_argument_group("fetch")
    fetch.add_argument("--per-taxon", type=int, default=100,
                       help="TOTAL target photos per taxon (default 100). "
                            "Counts photos already staged, so re-running with "
                            "the same value fetches nothing new.")
    fetch.add_argument("--top-up-new", action="store_true",
                       help="Treat --per-taxon as 'N NEW photos each run' instead "
                            "of a total (fetches N more every invocation).")
    fetch.add_argument("--limit", type=int, default=None,
                       help="Smoke cap: never fetch more than this per taxon.")
    fetch.add_argument("--license", default="cc0,cc-by",
                       help="Comma photo licenses (default cc0,cc-by; "
                            "add cc-by-nc for NonCommercial).")
    fetch.add_argument("--months", default=None,
                       help="Comma month numbers to stratify by season, e.g. 3,4,5.")
    fetch.add_argument("--place-id", type=int, default=1,
                       help="iNat place_id (default 1 = United States).")
    fetch.add_argument("--region", default=None,
                       help=f"Named region -> place_id ({', '.join(sorted(REGIONS))}). "
                            "Overrides --place-id if known.")
    fetch.add_argument("--per-page", type=int, default=200,
                       help="API page size (max 200).")
    fetch.add_argument("--max-pages", type=int, default=30,
                       help="Safety cap on API pages per taxon (default 30).")

    io = ap.add_argument_group("output / dedup")
    io.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"Staging root (default {DEFAULT_OUT}).")
    io.add_argument("--heldout-dir", default=str(HELDOUT_DIR),
                    help="Frozen held-out dir to de-duplicate against.")
    io.add_argument("--no-dedup-heldout", dest="dedup_heldout",
                    action="store_false",
                    help="Disable perceptual de-dup vs the held-out set "
                         "(NOT recommended; risks future leakage).")
    io.add_argument("--hamming", type=int, default=6,
                    help="Max dHash Hamming distance to count as a held-out match.")

    misc = ap.add_argument_group("misc")
    misc.add_argument("--sleep", type=float, default=0.5,
                      help="Base polite sleep seconds between calls (default 0.5).")
    misc.add_argument("--retries", type=int, default=4,
                      help="HTTP retry attempts (default 4).")
    misc.add_argument("--dry-run", action="store_true",
                      help="Resolve + page the API but download nothing.")
    misc.add_argument("--verify-taxa", action="store_true",
                      help="Verify the taxon id table against the live API and exit.")
    ap.set_defaults(dedup_heldout=True)
    args = ap.parse_args()

    if args.verify_taxa:
        ok = verify_taxa()
        sys.exit(0 if ok else 1)

    args.place_id = resolve_place_id(args)
    rows = select_taxa(args)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"

    print("LeafAlert iNaturalist fetcher")
    print(f"  staging root : {out_root}")
    print(f"  manifest     : {manifest_path}")
    print(f"  taxa selected: {len(rows)}  ({', '.join(t[0] for t in rows)})")
    print(f"  dry_run={args.dry_run}  dedup_heldout={args.dedup_heldout}"
          + (f"  limit={args.limit}" if args.limit is not None else ""))

    seen_ids = load_manifest(manifest_path)
    print(f"  resuming: {len(seen_ids)} photo id(s) already in manifest")

    heldout_hashes = []
    if args.dedup_heldout and not args.dry_run:
        hd = Path(args.heldout_dir)
        heldout_hashes = build_heldout_hashes(hd)
        print(f"  held-out de-dup index: {len(heldout_hashes)} hashes from {hd}")
        if hd.resolve() == out_root.resolve():
            print("  REFUSING to stage into the held-out dir.", file=sys.stderr)
            sys.exit(2)

    totals = defaultdict(int)
    per_class = defaultdict(int)
    for token, tid, sci, cls in rows:
        s, sk, hl, fl, wd = fetch_taxon(
            token, tid, sci, cls, args, seen_ids, heldout_hashes,
            out_root, manifest_path)
        totals["saved"] += s
        totals["skipped_seen"] += sk
        totals["skipped_heldout"] += hl
        totals["failed"] += fl
        totals["would"] += wd
        per_class[cls] += s + wd
        time.sleep(max(args.sleep, 1.0))  # polite between taxa

    print(f"\n{'='*68}\nSUMMARY")
    print(f"  saved          : {totals['saved']}")
    if args.dry_run:
        print(f"  would-fetch    : {totals['would']} (dry run)")
    print(f"  skipped (seen) : {totals['skipped_seen']}")
    print(f"  skipped (held) : {totals['skipped_heldout']}")
    print(f"  failed         : {totals['failed']}")
    print(f"  per class (saved+would): "
          + ", ".join(f"{c}={per_class[c]}" for c in CANONICAL_CLASSES))
    print(f"  manifest       : {manifest_path}")
    print("\nNext: audit_dataset.py on the merged pool, then merge into "
          "TrainingData/{class}/ (NEVER into TrainingData/Testing).")


if __name__ == "__main__":
    main()
