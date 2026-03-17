#!/usr/bin/env python3
"""
Download hard negative images for the safe_plants class.

These are visually similar but non-toxic plants that the model currently
confuses with poison ivy, poison oak, or poison sumac. Adding them as
hard negatives to the safe_plants class should reduce false positives.

Downloads 75 images per species (10 species = 750 images total):
  - 80% -> TrainingData_split/train/safe_plants/
  - 20% -> TrainingData_split/test/safe_plants/

Usage:
    python3 scripts/download_hard_negatives.py [--count 75] [--dry-run]
"""

import os
import sys
import time
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Hard negative species: visually similar to toxic plants but safe
# Grouped by which toxic species they resemble
HARD_NEGATIVES = {
    # Poison Ivy look-alikes (trifoliate leaves)
    "hog_peanut": 76399,          # Amphicarpaea bracteata
    "jack_in_the_pulpit": 49984,  # Arisaema triphyllum
    "raspberry": 47585,           # Rubus (brambles)
    "clematis": 76667,            # Clematis virginiana

    # Poison Oak look-alikes (lobed leaves)
    "antelope_brush": 58805,      # Purshia tridentata
    "skunkbush_sumac": 58750,     # Rhus trilobata
    "white_oak": 49223,           # Quercus alba seedlings

    # Poison Sumac look-alikes (pinnate compound leaves)
    "staghorn_sumac": 49248,      # Rhus typhina
    "smooth_sumac": 49249,        # Rhus glabra
    "tree_of_heaven": 57278,      # Ailanthus altissima
}

BASE_URL = "https://api.inaturalist.org/v1/observations"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "TrainingData_split" / "train" / "safe_plants"
TEST_DIR = PROJECT_ROOT / "TrainingData_split" / "test" / "safe_plants"


def fetch_observation_photos(taxon_id: int, count: int) -> list[str]:
    """Fetch photo URLs from iNaturalist for a given taxon.

    May need multiple pages if observations have few photos.
    """
    urls = []
    page = 1
    max_pages = 5  # Safety limit

    while len(urls) < count and page <= max_pages:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 200,
            "page": page,
            "order": "desc",
            "order_by": "votes",
        }

        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    API error (page {page}): {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            for photo in obs.get("photos", []):
                url = photo.get("url", "")
                if url:
                    # Switch from square thumbnail to medium (640px)
                    url = url.replace("/square.", "/medium.")
                    urls.append(url)
                if len(urls) >= count:
                    break
            if len(urls) >= count:
                break

        page += 1
        if page <= max_pages and len(urls) < count:
            time.sleep(1)  # Be polite between pages

    return urls[:count]


def download_image(url: str, save_path: Path) -> bool:
    """Download a single image to disk."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        save_path.write_bytes(resp.content)
        return True
    except Exception:
        return False


def download_species(
    name: str, taxon_id: int, count: int, dry_run: bool = False
) -> tuple[int, int]:
    """Download images for one species into safe_plants train/test dirs."""
    print(f"\n  {name} (taxon {taxon_id}):")
    print(f"    Fetching {count} photo URLs...")

    urls = fetch_observation_photos(taxon_id, count)
    if not urls:
        print(f"    No images found!")
        return 0, 0

    print(f"    Found {len(urls)} URLs")

    # 80/20 train/test split
    split_idx = max(1, int(len(urls) * 0.8))
    train_urls = urls[:split_idx]
    test_urls = urls[split_idx:]

    if dry_run:
        print(f"    [DRY RUN] Would download {len(train_urls)} train + {len(test_urls)} test")
        return len(train_urls), len(test_urls)

    # Download training images
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    train_count = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for i, url in enumerate(train_urls):
            save_path = TRAIN_DIR / f"{name}_{i:04d}.jpg"
            if save_path.exists():
                train_count += 1
                continue
            futures[executor.submit(download_image, url, save_path)] = save_path

        for future in as_completed(futures):
            if future.result():
                train_count += 1

    # Download testing images
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    test_count = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for i, url in enumerate(test_urls):
            save_path = TEST_DIR / f"{name}_test_{i:04d}.jpg"
            if save_path.exists():
                test_count += 1
                continue
            futures[executor.submit(download_image, url, save_path)] = save_path

        for future in as_completed(futures):
            if future.result():
                test_count += 1

    print(f"    Downloaded: {train_count} train + {test_count} test")
    return train_count, test_count


def main():
    parser = argparse.ArgumentParser(
        description="Download hard negative images for safe_plants class"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=75,
        help="Images per species (default: 75)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch URLs but don't download images",
    )
    args = parser.parse_args()

    print("LeafAlert Hard Negative Downloader")
    print(f"Target: {args.count} images per species, {len(HARD_NEGATIVES)} species")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Test dir:  {TEST_DIR}")

    # Count existing images
    existing_train = len(list(TRAIN_DIR.glob("*.jpg"))) if TRAIN_DIR.exists() else 0
    existing_test = len(list(TEST_DIR.glob("*.jpg"))) if TEST_DIR.exists() else 0
    print(f"\nExisting safe_plants: {existing_train} train, {existing_test} test")

    total_train = 0
    total_test = 0

    print(f"\nDownloading hard negatives:")
    for name, taxon_id in HARD_NEGATIVES.items():
        train, test = download_species(name, taxon_id, args.count, args.dry_run)
        total_train += train
        total_test += test
        time.sleep(1)  # Be polite to the API between species

    print(f"\n{'='*50}")
    print(f"Hard negatives downloaded:")
    print(f"  New training images: {total_train}")
    print(f"  New testing images:  {total_test}")

    # Recount totals
    final_train = len(list(TRAIN_DIR.glob("*.jpg"))) if TRAIN_DIR.exists() else 0
    final_test = len(list(TEST_DIR.glob("*.jpg"))) if TEST_DIR.exists() else 0
    print(f"\nFinal safe_plants totals:")
    print(f"  Training: {existing_train} existing + {total_train} new = {final_train} total")
    print(f"  Testing:  {existing_test} existing + {total_test} new = {final_test} total")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
