#!/usr/bin/env python3
"""
Re-export the distilled student checkpoint to a CORRECTED Core ML package.

The previous export baked a WRONG uniform-std (0.226) normalization into the
CoreML ImageType. This tool reloads the trained student weights and re-exports
with exact per-channel ImageNet normalization baked into the graph (via
NormalizeWrapper / export_coreml), without retraining.

Usage:
    python3 scripts/reexport_coreml.py
    python3 scripts/reexport_coreml.py --checkpoint checkpoints/student_distilled.pth

It does NOT retrain — it only re-traces and re-converts existing weights.
"""

import argparse
import sys
from pathlib import Path

import torch

# Make sibling modules importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the existing student architecture — do NOT duplicate it.
from distill_model import PlantDetectorNet
from coreml_export import export_coreml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "student_distilled.pth"
OUTPUT_PATH = PROJECT_ROOT / "LeafAlert" / "Resources" / "MLModels" / "PlantDetector.mlpackage"
IMAGE_SIZE = 224

# Canonical sorted class order — must match ImageFolder's alphabetical ordering
# used during training so label indices line up exactly.
CLASS_LABELS = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]


def main():
    parser = argparse.ArgumentParser(
        description="Re-export the distilled student to a corrected Core ML package.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
        help="Path to the distilled student .pth checkpoint",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help="Output .mlpackage path",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        print("Run scripts/distill_model.py first to produce student_distilled.pth.")
        sys.exit(1)

    print("=" * 60)
    print("LeafAlert PlantDetector — Core ML re-export (corrected normalization)")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output:     {output_path}")
    print(f"  Labels:     {CLASS_LABELS}")

    num_classes = len(CLASS_LABELS)
    model = PlantDetectorNet(num_classes)
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    short_description = (
        "Detects poison ivy, poison oak, and poison sumac from camera images. "
        "EfficientNet-B0 with spatial attention (v4-distilled), re-exported with "
        "exact per-channel ImageNet normalization baked into the graph. "
        f"Classes: {', '.join(CLASS_LABELS)}"
    )

    mlmodel = export_coreml(
        model, CLASS_LABELS, output_path, IMAGE_SIZE,
        short_description=short_description,
        version="4.1.0",
    )

    # Print the model spec so the caller can confirm the corrected I/O.
    print("\n--- Core ML model spec ---")
    print(mlmodel.get_spec())

    print("\nDone. Rebuild the app to include the corrected model.")


if __name__ == "__main__":
    main()
