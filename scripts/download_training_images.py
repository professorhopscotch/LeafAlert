#!/usr/bin/env python3
"""
Download training images for LeafAlert from iNaturalist.

Uses the iNaturalist API (no auth required) to fetch research-grade
observations of poison ivy, poison oak, and poison sumac, plus a
selection of safe look-alike plants.

Images are saved into the Create ML folder structure:
  TrainingData/
    poison_ivy/
    poison_oak/
    poison_sumac/
    safe_plants/
  TrainingData/Testing/
    poison_ivy/
    poison_oak/
    poison_sumac/
    safe_plants/

Usage:
    python3 scripts/download_training_images.py [--count 50]
"""

import os
import sys
import time
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# iNaturalist taxon IDs for our target species
TAXA = {
    "poison_ivy": 58735,       # Toxicodendron radicans
    "poison_oak": 52858,       # Toxicodendron diversilobum
    "poison_sumac": 54767,     # Toxicodendron vernix
}

# Safe look-alike plants (common non-toxic species hikers encounter)
SAFE_TAXA = {
    "virginia_creeper": 49911,  # Parthenocissus quinquefolia (looks like poison ivy)
    "box_elder": 49758,         # Acer negundo (looks like poison ivy)
    "fragrant_sumac": 56600,    # Rhus aromatica (looks like poison oak)
    "blackberry": 47578,        # Rubus spp. (common trail plant)
}

BASE_URL = "https://api.inaturalist.org/v1/observations"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "TrainingData"
TESTING_DIR = TRAINING_DIR / "Testing"


def fetch_observation_photos(taxon_id: int, count: int, page: int = 1) -> list[str]:
    """Fetch photo URLs from iNaturalist for a given taxon."""
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",  # Only verified observations
        "photos": "true",
        "per_page": min(count, 200),
        "page": page,
        "order": "desc",
        "order_by": "votes",  # Best quality first
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  API error: {e}")
        return []

    urls = []
    for obs in data.get("results", []):
        for photo in obs.get("photos", []):
            # Get medium-sized image (640px) — good for training
            url = photo.get("url", "")
            if url:
                # iNaturalist URLs use "square" by default; switch to "medium"
                url = url.replace("/square.", "/medium.")
                urls.append(url)
            if len(urls) >= count:
                break
        if len(urls) >= count:
            break

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
    name: str, taxon_id: int, count: int, test_split: float = 0.2
) -> tuple[int, int]:
    """Download images for one species, splitting into train/test sets."""
    print(f"\n{'='*50}")
    print(f"Downloading: {name} (taxon {taxon_id})")
    print(f"  Requesting {count} images...")

    urls = fetch_observation_photos(taxon_id, count)
    if not urls:
        print(f"  No images found!")
        return 0, 0

    print(f"  Found {len(urls)} photo URLs")

    # Split into training and testing
    split_idx = max(1, int(len(urls) * (1 - test_split)))
    train_urls = urls[:split_idx]
    test_urls = urls[split_idx:]

    train_count = 0
    test_count = 0

    # Download training images
    train_dir = TRAINING_DIR / name
    train_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for i, url in enumerate(train_urls):
            ext = ".jpg"
            save_path = train_dir / f"{name}_{i:04d}{ext}"
            if save_path.exists():
                train_count += 1
                continue
            futures[executor.submit(download_image, url, save_path)] = save_path

        for future in as_completed(futures):
            if future.result():
                train_count += 1

    # Download testing images
    test_dir = TESTING_DIR / name
    test_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for i, url in enumerate(test_urls):
            ext = ".jpg"
            save_path = test_dir / f"{name}_test_{i:04d}{ext}"
            if save_path.exists():
                test_count += 1
                continue
            futures[executor.submit(download_image, url, save_path)] = save_path

        for future in as_completed(futures):
            if future.result():
                test_count += 1

    print(f"  Downloaded: {train_count} training + {test_count} testing")
    return train_count, test_count


def main():
    parser = argparse.ArgumentParser(description="Download LeafAlert training images")
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of images per species (default: 50)",
    )
    parser.add_argument(
        "--toxic-only",
        action="store_true",
        help="Only download toxic plant images (skip safe plants)",
    )
    args = parser.parse_args()

    print("LeafAlert Training Image Downloader")
    print(f"Target: {args.count} images per species")
    print(f"Output: {TRAINING_DIR}")

    total_train = 0
    total_test = 0

    # Download toxic species
    for name, taxon_id in TAXA.items():
        train, test = download_species(name, taxon_id, args.count)
        total_train += train
        total_test += test
        time.sleep(1)  # Be polite to the API

    # Download safe look-alikes (combined into safe_plants folder)
    if not args.toxic_only:
        safe_count_per = max(10, args.count // len(SAFE_TAXA))
        safe_train = 0
        safe_test = 0

        print(f"\n{'='*50}")
        print(f"Downloading safe look-alike plants...")

        for safe_name, taxon_id in SAFE_TAXA.items():
            print(f"\n  Fetching {safe_name} (taxon {taxon_id})...")
            urls = fetch_observation_photos(taxon_id, safe_count_per)
            if not urls:
                continue

            split_idx = max(1, int(len(urls) * 0.8))
            train_urls = urls[:split_idx]
            test_urls = urls[split_idx:]

            train_dir = TRAINING_DIR / "safe_plants"
            train_dir.mkdir(parents=True, exist_ok=True)

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                for i, url in enumerate(train_urls):
                    save_path = train_dir / f"{safe_name}_{i:04d}.jpg"
                    if not save_path.exists():
                        futures[executor.submit(download_image, url, save_path)] = 1

                for f in as_completed(futures):
                    if f.result():
                        safe_train += 1

            test_dir = TESTING_DIR / "safe_plants"
            test_dir.mkdir(parents=True, exist_ok=True)

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                for i, url in enumerate(test_urls):
                    save_path = test_dir / f"{safe_name}_test_{i:04d}.jpg"
                    if not save_path.exists():
                        futures[executor.submit(download_image, url, save_path)] = 1

                for f in as_completed(futures):
                    if f.result():
                        safe_test += 1

            time.sleep(1)

        print(f"  Safe plants: {safe_train} training + {safe_test} testing")
        total_train += safe_train
        total_test += safe_test

    print(f"\n{'='*50}")
    print(f"DONE!")
    print(f"  Total training images: {total_train}")
    print(f"  Total testing images:  {total_test}")
    print(f"\nNext steps:")
    print(f"  1. Open Xcode → File → New → Project → Create ML")
    print(f"  2. Choose 'Image Classifier'")
    print(f"  3. Drag '{TRAINING_DIR}' folder into Training Data")
    print(f"  4. Drag '{TESTING_DIR}' folder into Testing Data")
    print(f"  5. Set Transfer Learning to 'MobileNetV2' (or Full Network)")
    print(f"  6. Click Train")
    print(f"  7. Export the .mlmodel and rename to PlantDetector.mlmodel")
    print(f"  8. Drop into LeafAlert/Resources/MLModels/")


if __name__ == "__main__":
    main()
