#!/usr/bin/env python3
"""
Expand the LeafAlert training dataset by downloading more images from iNaturalist.

Downloads directly into TrainingData_split/train/ and TrainingData_split/test/
with an 80/20 split. Uses pagination to fetch many more images than the original
download script, and includes subspecies/varieties for poison ivy and oak.

Numbering starts at offset 0500 to avoid colliding with existing files.

Usage:
    python3 scripts/expand_dataset.py [--dry-run]
"""

import os
import sys
import time
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "TrainingData_split" / "train"
TEST_DIR = PROJECT_ROOT / "TrainingData_split" / "test"

BASE_URL = "https://api.inaturalist.org/v1/observations"

# ── Target downloads per class ──────────────────────────────────────────
# These are NEW images to add on top of existing ones.
TARGETS = {
    "poison_ivy": 500,    # weakest class
    "poison_oak": 500,    # second weakest
    "poison_sumac": 300,
    "safe_plants": 300,
}

# ── Taxa with subspecies/varieties ──────────────────────────────────────
# Each entry: (taxon_id, description, weight)
# Weight controls what fraction of the target this taxon gets.
POISON_IVY_TAXA = [
    (58732,  "Toxicodendron radicans (eastern poison ivy)", 0.6),
    (854783, "Toxicodendron rydbergii (western poison ivy)", 0.4),
]

POISON_OAK_TAXA = [
    (51080, "Toxicodendron diversilobum (Pacific poison oak)", 0.6),
    (82790, "Toxicodendron pubescens (Atlantic poison oak)", 0.4),
]

POISON_SUMAC_TAXA = [
    (54767, "Toxicodendron vernix (poison sumac)", 1.0),
]

SAFE_TAXA = [
    (50278, "virginia_creeper — Parthenocissus quinquefolia", 0.25),
    (47726, "box_elder — Acer negundo", 0.25),
    (58738, "fragrant_sumac — Rhus aromatica", 0.25),
    (82110, "blackberry — Rubus allegheniensis", 0.25),
]

# Starting file number offset to avoid collisions with existing images.
FILE_OFFSET = 500


def fetch_photo_urls(taxon_id: int, count: int) -> list[str]:
    """Fetch up to `count` unique photo URLs using pagination."""
    urls = []
    seen = set()
    page = 1
    max_pages = (count // 200) + 3  # extra pages in case some obs have no usable photos

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
            print(f"    API error on page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break  # no more pages

        for obs in results:
            for photo in obs.get("photos", []):
                url = photo.get("url", "")
                if not url:
                    continue
                url = url.replace("/square.", "/medium.")
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
                if len(urls) >= count:
                    break
            if len(urls) >= count:
                break

        page += 1
        time.sleep(1)  # polite delay between API calls

    return urls[:count]


def download_image(url: str, save_path: Path) -> bool:
    """Download a single image to disk."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        if len(resp.content) < 1000:
            return False  # skip tiny/broken images
        save_path.write_bytes(resp.content)
        return True
    except Exception:
        return False


def download_batch(urls: list[str], save_dir: Path, prefix: str, start_idx: int) -> int:
    """Download a batch of images with 5 concurrent workers. Returns count downloaded."""
    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for i, url in enumerate(urls):
            save_path = save_dir / f"{prefix}_{start_idx + i:04d}.jpg"
            if save_path.exists():
                downloaded += 1
                continue
            futures[executor.submit(download_image, url, save_path)] = save_path

        for future in as_completed(futures):
            if future.result():
                downloaded += 1

    return downloaded


def expand_class(class_name: str, taxa: list[tuple], target: int, dry_run: bool = False) -> tuple[int, int]:
    """Download images for a class, splitting 80/20 into train/test."""
    print(f"\n{'=' * 60}")
    print(f"Expanding: {class_name} (target: {target} new images)")

    all_urls = []
    global_seen = set()

    for taxon_id, description, weight in taxa:
        taxon_target = int(target * weight) + 10  # fetch a few extra
        print(f"  Fetching {description} (taxon {taxon_id}, ~{int(target * weight)} images)...")
        urls = fetch_photo_urls(taxon_id, taxon_target)
        # Deduplicate across taxa
        new_urls = [u for u in urls if u not in global_seen]
        global_seen.update(new_urls)
        all_urls.extend(new_urls)
        print(f"    Got {len(new_urls)} unique URLs")

    # Trim to target
    all_urls = all_urls[:target]
    print(f"  Total unique URLs: {len(all_urls)}")

    if dry_run:
        train_count = int(len(all_urls) * 0.8)
        test_count = len(all_urls) - train_count
        print(f"  [DRY RUN] Would download {train_count} train + {test_count} test")
        return train_count, test_count

    # 80/20 split
    split_idx = int(len(all_urls) * 0.8)
    train_urls = all_urls[:split_idx]
    test_urls = all_urls[split_idx:]

    # Download training images
    train_dir = TRAIN_DIR / class_name
    train_prefix = f"{class_name}_exp"
    print(f"  Downloading {len(train_urls)} training images...")
    train_count = download_batch(train_urls, train_dir, train_prefix, FILE_OFFSET)

    # Download test images
    test_dir = TEST_DIR / class_name
    test_prefix = f"{class_name}_exp_test"
    print(f"  Downloading {len(test_urls)} test images...")
    test_count = download_batch(test_urls, test_dir, test_prefix, FILE_OFFSET)

    print(f"  Result: {train_count} train + {test_count} test = {train_count + test_count} total")
    return train_count, test_count


def count_existing():
    """Count existing images per class."""
    print("Current dataset counts:")
    for split_name, split_dir in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        if not split_dir.is_dir():
            print(f"  {split_name}: (none — directory not found)")
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = split_dir / class_name
            if class_dir.is_dir():
                count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {split_name}/{class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Expand LeafAlert training dataset")
    parser.add_argument("--dry-run", action="store_true", help="Only fetch URLs, don't download")
    args = parser.parse_args()

    print("LeafAlert Dataset Expansion")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Test dir:  {TEST_DIR}")
    print()

    count_existing()

    total_train = 0
    total_test = 0

    # Poison ivy — weakest class, gets subspecies boost
    t, e = expand_class("poison_ivy", POISON_IVY_TAXA, TARGETS["poison_ivy"], args.dry_run)
    total_train += t
    total_test += e

    # Poison oak — second weakest, also gets subspecies
    t, e = expand_class("poison_oak", POISON_OAK_TAXA, TARGETS["poison_oak"], args.dry_run)
    total_train += t
    total_test += e

    # Poison sumac
    t, e = expand_class("poison_sumac", POISON_SUMAC_TAXA, TARGETS["poison_sumac"], args.dry_run)
    total_train += t
    total_test += e

    # Safe plants
    t, e = expand_class("safe_plants", SAFE_TAXA, TARGETS["safe_plants"], args.dry_run)
    total_train += t
    total_test += e

    print(f"\n{'=' * 60}")
    print(f"EXPANSION COMPLETE")
    print(f"  New training images: {total_train}")
    print(f"  New test images:     {total_test}")
    print(f"  Total new:           {total_train + total_test}")
    print()

    count_existing()


if __name__ == "__main__":
    main()
