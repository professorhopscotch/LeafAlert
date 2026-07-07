#!/usr/bin/env python3
"""
Reproducible GBIF-based image fetch for LeafAlert dataset expansion.

Pulls research-quality, openly-licensed plant occurrence images from GBIF
(Global Biodiversity Information Facility) for the same taxa used by the
iNaturalist fetcher, into the shared staging layout so both feeds can be
de-duplicated and ingested together.

Why GBIF (in addition to iNaturalist):
  - GBIF aggregates iNaturalist AND many other datasets (herbaria, museums,
    citizen-science networks), giving broader taxonomic/geographic/seasonal
    coverage — exactly the diversity LeafAlert lacks (red spring leaves,
    green summer, fall color, leafless winter vines).
  - GBIF records carry a canonical, DOI-citable occurrence `key` and an
    explicit `license` per record, so provenance is reproducible.

This script uses the GBIF Occurrence *search* API (no auth, no key required),
which is resumable and paginate-able. It intentionally does NOT trigger a
GBIF Occurrence *Download* (which is async, email-gated, and can take minutes
to hours to become ready) — the search API gives us the same records with
per-record media + license, which is what we need for a resumable image pull.
To make the pull citable/reproducible anyway, the manifest records the exact
query (taxonKey, license filter, country, date bounds) and the GBIF occurrence
`key` for every image, so the identical set can be re-derived or cited later.

=== CONTRACTS (must match the iNaturalist fetcher) ===
Classes (ImageFolder-alphabetical, DO NOT change):
    poison_ivy, poison_oak, poison_sumac, safe_plants
Toxic = first three. safe_plants is the negative / look-alike bucket.

Staging layout (NOT the frozen held-out set, NOT the live train pool):
    data_staging/gbif/<class>/<image files>
    data_staging/gbif/manifest.jsonl

Manifest schema (one JSON object per line, SAME schema as the iNat fetcher):
    {
      "source":        "gbif",
      "occurrence_id": <int|str>,   # GBIF occurrence key (stable, citable)
      "taxon":         "Toxicodendron radicans",  # scientificName
      "taxon_key":     <int>,       # GBIF usageKey used for the query
      "class":         "poison_ivy",
      "license":       "CC_BY_4_0", # normalized enum
      "attribution":   "© <rightsHolder>, licensed CC-BY-4.0 (via GBIF)",
      "date":          "2021-06-14" | null,   # observation eventDate (date part)
      "lat":           <float|null>,
      "lng":           <float|null>,
      "image_url":     "<source media identifier URL>",
      "file":          "gbif/poison_ivy/gbif_6130342104_0.jpg",  # rel to staging
      "phash":         "<16-hex perceptual hash>" | null
    }

=== HELD-OUT FREEZE ===
Everything lands in data_staging/ ONLY. This script never writes to
TrainingData/ or TrainingData/Testing/. De-duplication against the frozen
held-out set and the train pool is the ingest step's job; this fetcher
computes a perceptual hash (phash) per image and records it in the manifest,
and additionally skips images whose phash matches a --dedupe-against tree
(e.g. TrainingData/Testing) so leaked held-out photos are never staged.

=== SMOKE TEST (safe, tiny) ===
    python scripts/fetch_gbif.py --limit 3 --dry-run
    python scripts/fetch_gbif.py --limit 3            # downloads 3/class

=== FULL PULL (do NOT run inside an agent; it is a bulk download) ===
    python scripts/fetch_gbif.py --limit 400 \
        --dedupe-against TrainingData/Testing

Resumable: re-running skips images already on disk and occurrence/photo ids
already present in the manifest, and appends only new manifest lines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlencode, urlparse

# Prefer `requests` (as the sibling iNat fetcher uses) but fall back to the
# stdlib so this script runs even in a venv where `requests` isn't installed.
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

import urllib.error
import urllib.request

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is a stated dependency
    Image = None  # phash becomes a no-op if PIL is somehow unavailable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGING_ROOT = PROJECT_ROOT / "data_staging" / "gbif"
MANIFEST_PATH = STAGING_ROOT / "manifest.jsonl"

GBIF_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"

# GBIF caps per-page limit at 300 and (offset + limit) at 100000.
GBIF_MAX_LIMIT = 300
GBIF_MAX_OFFSET_PLUS_LIMIT = 100_000

# Only these two licenses are allowed for staged data (task contract).
# Map GBIF's enum filter value <-> the canonical license-URL prefixes GBIF
# returns per record, so we can both filter server-side AND double-check
# client-side (the search license filter is applied at the occurrence level,
# but individual media entries can carry a different, stricter license).
ALLOWED_LICENSES = {
    "CC0_1_0": "creativecommons.org/publicdomain/zero/1.0",
    "CC_BY_4_0": "creativecommons.org/licenses/by/4.0",
}

# --- Taxa -> class mapping (SAME species set as the iNat fetcher) ------------
# Each class maps to a list of scientific names; names are resolved to GBIF
# taxonKeys at runtime via the species/match endpoint (so we never hardcode a
# key that could drift). Multiple accepted names per bucket broaden coverage.
CLASS_TAXA: dict[str, list[str]] = {
    "poison_ivy": [
        "Toxicodendron radicans",   # eastern poison ivy
        "Toxicodendron rydbergii",  # western/northern poison ivy
    ],
    "poison_oak": [
        "Toxicodendron diversilobum",  # Pacific poison oak
        "Toxicodendron pubescens",     # Atlantic poison oak
    ],
    "poison_sumac": [
        "Toxicodendron vernix",  # poison sumac
    ],
    # Hard-negative look-alikes for the safe/negative bucket. These are the
    # exact confusers listed in the research notes; without them the model
    # has no look-alike negatives and over-fires on any trifoliate/pinnate leaf.
    "safe_plants": [
        "Parthenocissus quinquefolia",  # Virginia creeper (ivy look-alike)
        "Acer negundo",                 # boxelder (trifoliate look-alike)
        "Rubus",                        # blackberry/raspberry brambles
        "Rhus aromatica",               # fragrant sumac (oak look-alike)
        "Rhus glabra",                  # smooth sumac (sumac look-alike)
        "Rhus typhina",                 # staghorn sumac (sumac look-alike)
    ],
}

CLASSES = ("poison_ivy", "poison_oak", "poison_sumac", "safe_plants")


# --------------------------------------------------------------------------- #
# HTTP helpers — thin adapter over `requests` OR stdlib urllib.
# --------------------------------------------------------------------------- #
class HttpClient:
    """Minimal GET client with a JSON + bytes API, backend-agnostic."""

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._session = None
        if requests is not None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": user_agent})

    def get_json(self, url: str, params=None, timeout: float = 45) -> dict:
        """GET a URL (params may be a dict or list of (k, v) tuples) -> JSON."""
        if self._session is not None:
            resp = self._session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        full = url
        if params:
            full = f"{url}?{urlencode(list(_as_pairs(params)), doseq=False)}"
        req = urllib.request.Request(full, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as fh:  # noqa: S310
            return json.loads(fh.read().decode("utf-8"))

    def get_bytes(self, url: str, timeout: float = 40) -> tuple[bytes, str]:
        """GET a URL -> (content_bytes, content_type)."""
        if self._session is not None:
            resp = self._session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content, resp.headers.get("content-type", "")
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as fh:  # noqa: S310
            return fh.read(), fh.headers.get("content-type", "")


def _as_pairs(params) -> Iterable[tuple[str, str]]:
    """Normalize a dict or list-of-tuples into (str, str) pairs for urlencode."""
    items = params.items() if isinstance(params, dict) else params
    for k, v in items:
        yield str(k), str(v)


def resolve_taxon_key(session: "HttpClient", name: str) -> Optional[dict]:
    """Resolve a scientific name to a GBIF usageKey via species/match.

    Returns a dict {"taxon_key": int, "scientific_name": str} or None on
    no/low-confidence match.
    """
    try:
        data = session.get_json(GBIF_MATCH_URL, params={"name": name}, timeout=30)
    except Exception as exc:  # network / JSON errors
        print(f"    ! species/match failed for {name!r}: {exc}", file=sys.stderr)
        return None

    usage_key = data.get("usageKey")
    match_type = data.get("matchType", "NONE")
    if not usage_key or match_type == "NONE":
        print(f"    ! no GBIF match for {name!r} (matchType={match_type})")
        return None

    return {
        "taxon_key": int(usage_key),
        # Prefer the canonical name GBIF returns; fall back to the query name.
        "scientific_name": data.get("scientificName") or name,
    }


def search_occurrences(
    session: "HttpClient",
    taxon_key: int,
    *,
    country: Optional[str],
    year_from: Optional[int],
    year_to: Optional[int],
    limit: int,
    offset: int,
) -> Optional[dict]:
    """One page of the GBIF occurrence search, filtered to allowed licenses.

    `limit`/`offset` are the GBIF page params. Returns the parsed JSON dict
    (with keys offset/limit/endOfRecords/count/results) or None on error.
    """
    params: list[tuple[str, object]] = [
        ("taxonKey", taxon_key),
        ("mediaType", "StillImage"),
        ("limit", limit),
        ("offset", offset),
    ]
    # Repeated `license` params act as an OR filter on the GBIF side.
    for lic in ALLOWED_LICENSES:
        params.append(("license", lic))
    if country:
        params.append(("country", country))
    if year_from is not None and year_to is not None:
        params.append(("year", f"{year_from},{year_to}"))
    elif year_from is not None:
        params.append(("year", f"{year_from},{datetime.now().year}"))

    try:
        return session.get_json(GBIF_SEARCH_URL, params=params, timeout=45)
    except Exception as exc:
        print(f"    ! occurrence/search failed (offset={offset}): {exc}",
              file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# License / media parsing
# --------------------------------------------------------------------------- #
def normalize_license(license_url_or_code: Optional[str]) -> Optional[str]:
    """Map a GBIF license URL (or code) to our allowed enum, else None."""
    if not license_url_or_code:
        return None
    val = license_url_or_code.lower()
    for code, marker in ALLOWED_LICENSES.items():
        if marker in val or code.lower() in val:
            return code
    return None


def license_short(code: str) -> str:
    return {"CC0_1_0": "CC0-1.0", "CC_BY_4_0": "CC-BY-4.0"}.get(code, code)


def iter_still_images(occ: dict) -> Iterable[dict]:
    """Yield StillImage media entries (dicts) that have an identifier URL."""
    for media in occ.get("media", []) or []:
        if media.get("type") not in (None, "StillImage"):
            continue
        if media.get("identifier"):
            yield media


def parse_event_date(occ: dict) -> Optional[str]:
    """Return the date portion (YYYY-MM-DD) of the occurrence eventDate."""
    ev = occ.get("eventDate")
    if not ev or not isinstance(ev, str):
        return None
    # GBIF eventDate can be "2021-06-14T11:23:15", "2021-06-14",
    # or a range "2021-06-14/2021-06-15". Take the leading date token.
    token = ev.split("/")[0].split("T")[0].strip()
    return token or None


# --------------------------------------------------------------------------- #
# Perceptual hash (dHash, 64-bit -> 16 hex chars). Matches a common phash the
# ingest/dedupe step can compare with Hamming distance.
# --------------------------------------------------------------------------- #
def compute_phash(path: Path) -> Optional[str]:
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((9, 8), Image.LANCZOS)
            # getdata() is deprecated in Pillow 14; tobytes() is stable and
            # returns one byte per luminance pixel for an 'L' image.
            px = list(im.tobytes())
        bits = 0
        idx = 0
        for row in range(8):
            base = row * 9
            for col in range(8):
                left = px[base + col]
                right = px[base + col + 1]
                bits = (bits << 1) | (1 if left > right else 0)
                idx += 1
        return f"{bits:016x}"
    except Exception:
        return None


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def load_dedupe_hashes(root: Path) -> set[str]:
    """Compute phashes for every image under `root` (e.g. held-out set)."""
    hashes: set[str] = set()
    if not root.exists():
        return hashes
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            h = compute_phash(p)
            if h:
                hashes.add(h)
    return hashes


# --------------------------------------------------------------------------- #
# Manifest state (for resumability + idempotency)
# --------------------------------------------------------------------------- #
def load_manifest_state(path: Path) -> tuple[list[dict], set[str], set[str]]:
    """Return (records, seen_image_urls, seen_phashes) from an existing manifest."""
    records: list[dict] = []
    seen_urls: set[str] = set()
    seen_phashes: set[str] = set()
    if not path.exists():
        return records, seen_urls, seen_phashes
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(rec)
            if rec.get("image_url"):
                seen_urls.add(rec["image_url"])
            if rec.get("phash"):
                seen_phashes.add(rec["phash"])
    return records, seen_urls, seen_phashes


def append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _ext_from_media(media: dict, url: str) -> str:
    fmt = (media.get("format") or "").lower()
    if "png" in fmt:
        return ".png"
    if "webp" in fmt:
        return ".webp"
    if "jpeg" in fmt or "jpg" in fmt:
        return ".jpg"
    # fall back to URL suffix
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png", ".webp"):
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def download_image(session: "HttpClient", url: str, dest: Path) -> bool:
    try:
        content, content_type = session.get_bytes(url, timeout=40)
        if "image" not in content_type:
            return False
        if not content:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Main per-class fetch
# --------------------------------------------------------------------------- #
def fetch_class(
    session: "HttpClient",
    class_name: str,
    taxa: list[str],
    *,
    limit: int,
    country: Optional[str],
    year_from: Optional[int],
    year_to: Optional[int],
    dry_run: bool,
    seen_urls: set[str],
    seen_phashes: set[str],
    dedupe_hashes: set[str],
    phash_threshold: int,
    sleep: float,
) -> int:
    """Fetch up to `limit` NEW images for one class across its taxa.

    Returns the number of images newly staged (or that would be, in dry-run).
    """
    print(f"\n{'=' * 60}")
    print(f"CLASS: {class_name}  (target {limit} new images)")
    class_dir = STAGING_ROOT / class_name
    staged = 0

    for name in taxa:
        if staged >= limit:
            break
        print(f"\n  taxon: {name}")
        resolved = resolve_taxon_key(session, name)
        if not resolved:
            continue
        taxon_key = resolved["taxon_key"]
        canonical = resolved["scientific_name"]
        print(f"    -> taxonKey={taxon_key} ({canonical})")

        offset = 0
        page_size = min(GBIF_MAX_LIMIT, max(limit, 20))
        while staged < limit:
            if offset + page_size > GBIF_MAX_OFFSET_PLUS_LIMIT:
                print("    ! reached GBIF offset ceiling; stopping this taxon")
                break

            page = search_occurrences(
                session,
                taxon_key,
                country=country,
                year_from=year_from,
                year_to=year_to,
                limit=page_size,
                offset=offset,
            )
            if page is None:
                break
            results = page.get("results", []) or []
            if not results:
                break

            for occ in results:
                if staged >= limit:
                    break

                occ_id = occ.get("key")
                occ_license_code = normalize_license(occ.get("license"))
                event_date = parse_event_date(occ)
                lat = occ.get("decimalLatitude")
                lng = occ.get("decimalLongitude")

                for m_idx, media in enumerate(iter_still_images(occ)):
                    if staged >= limit:
                        break
                    url = media.get("identifier")
                    if not url or url in seen_urls:
                        continue

                    # Per-media license may be stricter than the occurrence
                    # license; enforce the allowed set at the media level too.
                    # IMPORTANT: only inherit the occurrence license when the
                    # media entry carries NO license field at all. If the media
                    # has an explicit license that is not in the allowed set,
                    # normalize_license() returns None -- we must REJECT it, not
                    # silently fall back to the (always-permissive, server-
                    # filtered) occurrence license. Otherwise a CC-BY-NC/-SA
                    # media image from an aggregated herbarium/museum dataset
                    # would be staged under the occurrence's permissive label.
                    raw_media_license = media.get("license")
                    media_license = normalize_license(raw_media_license)
                    if raw_media_license and media_license is None:
                        continue  # explicit, disallowed media license -> reject
                    lic_code = media_license or occ_license_code
                    if lic_code not in ALLOWED_LICENSES:
                        continue

                    rights = (
                        media.get("rightsHolder")
                        or occ.get("rightsHolder")
                        or media.get("creator")
                        or occ.get("recordedBy")
                        or "unknown"
                    )
                    attribution = (
                        f"© {rights}, licensed {license_short(lic_code)} "
                        f"(via GBIF, occurrence {occ_id})"
                    )

                    ext = _ext_from_media(media, url)
                    fname = f"gbif_{occ_id}_{m_idx}{ext}"
                    dest = class_dir / fname
                    rel = f"gbif/{class_name}/{fname}"

                    seen_urls.add(url)  # avoid re-attempting within this run

                    if dry_run:
                        print(
                            f"      [DRY] {class_name}: {rel}  "
                            f"lic={lic_code} date={event_date} "
                            f"loc=({lat},{lng})"
                        )
                        staged += 1
                        continue

                    if dest.exists():
                        # Already downloaded in a prior run; ensure it's
                        # represented in this run's counters but don't re-add
                        # to the manifest (manifest load already has it).
                        staged += 1
                        continue

                    if not download_image(session, url, dest):
                        print(f"      x download failed: {url}")
                        continue

                    phash = compute_phash(dest)

                    # An undecodable image (content-type OK but PIL can't decode)
                    # yields phash=None. It would otherwise bypass the held-out
                    # leakage guard entirely and land a null-phash row in the
                    # manifest, so treat it as unusable and drop it.
                    if phash is None:
                        dest.unlink(missing_ok=True)
                        print(f"      ~ skip (undecodable): {rel}")
                        continue

                    # De-dupe: against held-out/train tree AND already-staged.
                    if phash:
                        leaked = any(
                            hamming_hex(phash, h) <= phash_threshold
                            for h in dedupe_hashes
                        )
                        if leaked:
                            dest.unlink(missing_ok=True)
                            print(f"      ~ skip (matches dedupe set): {rel}")
                            continue
                        dup = any(
                            hamming_hex(phash, h) <= phash_threshold
                            for h in seen_phashes
                        )
                        if dup:
                            dest.unlink(missing_ok=True)
                            print(f"      ~ skip (near-dup of staged): {rel}")
                            continue
                        seen_phashes.add(phash)

                    record = {
                        "source": "gbif",
                        "occurrence_id": occ_id,
                        "taxon": canonical,
                        "taxon_key": taxon_key,
                        "class": class_name,
                        "license": lic_code,
                        "attribution": attribution,
                        "date": event_date,
                        "lat": lat,
                        "lng": lng,
                        "image_url": url,
                        "file": rel,
                        "phash": phash,
                    }
                    append_manifest(MANIFEST_PATH, record)
                    staged += 1
                    print(f"      + {rel}  ({license_short(lic_code)})")

            if page.get("endOfRecords"):
                break
            offset += page_size
            if sleep:
                time.sleep(sleep)

    print(f"\n  -> {class_name}: {staged} new image(s) "
          f"{'(dry-run)' if dry_run else 'staged'}")
    return staged


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch openly-licensed plant images from GBIF into "
            "data_staging/gbif/<class>/ with a provenance manifest."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max NEW images per class (default: 200). Use --limit 3 to smoke test.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=CLASSES,
        default=list(CLASSES),
        help="Subset of classes to fetch (default: all four).",
    )
    parser.add_argument(
        "--country",
        default="US",
        help="ISO 3166-1 alpha-2 country filter (default: US). Pass '' to disable.",
    )
    parser.add_argument(
        "--year-from",
        type=int,
        default=None,
        help="Earliest observation year (inclusive).",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        default=None,
        help="Latest observation year (inclusive). Defaults to current year "
             "if only --year-from is given.",
    )
    parser.add_argument(
        "--dedupe-against",
        default=None,
        help=(
            "Path (relative to repo root or absolute) to an image tree whose "
            "perceptual hashes must be excluded from staged results — pass "
            "TrainingData/Testing to guarantee no held-out leakage."
        ),
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=5,
        help="Max Hamming distance (0-64) to treat two images as duplicates "
             "(default: 5).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between GBIF search pages (be polite).",
    )
    parser.add_argument(
        "--user-agent",
        default="LeafAlert-dataset-builder/1.0 (+https://github.com/; research use)",
        help="HTTP User-Agent for GBIF requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve taxa and query GBIF but do NOT download or write files.",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be >= 1")

    country = args.country or None
    session = HttpClient(args.user_agent)
    if requests is None:
        print("  (note: 'requests' not installed; using stdlib urllib)")

    print("LeafAlert GBIF fetcher")
    print(f"  staging root : {STAGING_ROOT}")
    print(f"  manifest     : {MANIFEST_PATH}")
    print(f"  classes      : {', '.join(args.classes)}")
    print(f"  limit/class  : {args.limit}")
    print(f"  country      : {country or '(any)'}")
    print(f"  license filt : {', '.join(ALLOWED_LICENSES)}")
    print(f"  dry_run      : {args.dry_run}")

    # Load prior manifest state for resumability/idempotency.
    _records, seen_urls, seen_phashes = load_manifest_state(MANIFEST_PATH)
    if _records:
        print(f"  existing manifest rows: {len(_records)}")

    # Load held-out / train dedupe hashes.
    dedupe_hashes: set[str] = set()
    if args.dedupe_against:
        dpath = Path(args.dedupe_against)
        if not dpath.is_absolute():
            dpath = PROJECT_ROOT / dpath
        print(f"  dedupe tree  : {dpath}")
        dedupe_hashes = load_dedupe_hashes(dpath)
        print(f"    computed {len(dedupe_hashes)} phashes to exclude")

    if not args.dry_run:
        STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    total = 0
    for class_name in args.classes:
        total += fetch_class(
            session,
            class_name,
            CLASS_TAXA[class_name],
            limit=args.limit,
            country=country,
            year_from=args.year_from,
            year_to=args.year_to,
            dry_run=args.dry_run,
            seen_urls=seen_urls,
            seen_phashes=seen_phashes,
            dedupe_hashes=dedupe_hashes,
            phash_threshold=args.phash_threshold,
            sleep=args.sleep,
        )

    print(f"\n{'=' * 60}")
    print(f"DONE. {total} image(s) "
          f"{'would be staged (dry-run)' if args.dry_run else 'staged'}.")
    if not args.dry_run:
        print(f"Manifest: {MANIFEST_PATH}")
        print("Next: de-dupe + ingest staged images into the train pool "
              "(never into TrainingData/Testing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
