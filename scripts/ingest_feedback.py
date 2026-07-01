#!/usr/bin/env python3
"""
Ingest user feedback into the training dataset.

Reads manifest.json from a feedback folder, copies confirmed/corrected images
into TrainingData_split/train/<label>/, and skips discarded or not_a_plant entries.

Usage:
    python3 scripts/ingest_feedback.py [--feedback-dir PATH]

The feedback folder can come from:
  - iCloud Drive sync (auto-detected from ~/Library/Mobile Documents/...)
  - AirDrop from the iPhone
  - Local project root (default fallback)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "TrainingData_split" / "train"

# Default: look in project root for an AirDropped feedback folder
DEFAULT_FEEDBACK_DIR = PROJECT_ROOT / "feedback"

# iCloud Drive paths to search for feedback (user may have picked any of these)
ICLOUD_SEARCH_PATHS = [
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "LeafAlert" / "feedback",
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "feedback",
]

# Labels that map to training classes
VALID_LABELS = {"poison_ivy", "poison_oak", "poison_sumac", "safe_plants"}


def ingest_feedback(feedback_dir: Path, train_dir: Path, dry_run: bool = False):
    """Read manifest.json and copy images into training directories."""
    manifest_path = feedback_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"No manifest.json found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    print(f"Found {len(entries)} feedback entries")

    copied = 0
    skipped = 0
    errors = 0

    for entry in entries:
        status = entry.get("feedbackStatus", "none")
        filename = entry.get("filename", "")
        original = entry.get("originalPrediction", "")

        # Determine the correct label
        if status == "confirmed":
            label = original
        elif status == "corrected":
            label = entry.get("correctedLabel", "")
        else:
            # Skip "none", "discarded", etc.
            skipped += 1
            continue

        # A "not_a_plant" correction is a strong false-positive signal: the image
        # is a valid negative/safe example, so route it to the safe_plants class.
        if label == "not_a_plant":
            print(f"  Remap 'not_a_plant' → safe_plants: {filename}")
            label = "safe_plants"

        # Skip non-trainable labels
        if label not in VALID_LABELS:
            print(f"  Skip (non-trainable label '{label}'): {filename}")
            skipped += 1
            continue

        # Source image
        src = feedback_dir / filename
        if not src.exists():
            print(f"  Missing image: {filename}")
            errors += 1
            continue

        # Destination
        dest_dir = train_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Use feedback_ prefix to distinguish from iNaturalist images
        dest_name = f"feedback_{filename}"
        dest = dest_dir / dest_name

        if dest.exists():
            skipped += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] Would copy {filename} → {label}/{dest_name}")
        else:
            shutil.copy2(src, dest)
            print(f"  Copied {filename} → {label}/{dest_name}")
        copied += 1

    print(f"\nSummary:")
    print(f"  Copied:  {copied}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")

    if copied > 0 and not dry_run:
        print(f"\nNew images added to {train_dir}")
        print("Run `python3 scripts/train_model.py` to retrain with feedback data.")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest user feedback from iCloud Drive into training dataset"
    )
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        default=DEFAULT_FEEDBACK_DIR,
        help=f"Path to feedback directory (default: {DEFAULT_FEEDBACK_DIR})",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=TRAIN_DIR,
        help=f"Path to training data directory (default: {TRAIN_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LeafAlert — Feedback Ingestion")
    print("=" * 60)
    print(f"Feedback dir: {args.feedback_dir}")
    print(f"Train dir:    {args.train_dir}")
    print()

    feedback_dir = args.feedback_dir

    # Auto-discover feedback from iCloud Drive if default path doesn't exist
    if not feedback_dir.exists():
        for icloud_path in ICLOUD_SEARCH_PATHS:
            if icloud_path.exists() and (icloud_path / "manifest.json").exists():
                feedback_dir = icloud_path
                print(f"Found iCloud feedback at: {icloud_path}")
                break
        else:
            print(f"Feedback directory not found at: {args.feedback_dir}")
            print("Searched iCloud Drive paths:")
            for p in ICLOUD_SEARCH_PATHS:
                print(f"  {p}")
            print("\nPick a folder in iCloud Drive from the app's Settings > Feedback Sync.")
            sys.exit(1)

    ingest_feedback(feedback_dir, args.train_dir, args.dry_run)


if __name__ == "__main__":
    main()
