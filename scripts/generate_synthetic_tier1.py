#!/usr/bin/env python3
"""
Tier 1 Synthetic Data: Augmentation-based generation.

Takes existing training images and applies realistic transformations that
simulate real-world phone-camera-on-trail conditions:
  - Background compositing (plant cutouts on random trail backgrounds)
  - Lighting simulation (dappled sunlight, shade, overexposure)
  - Motion blur (simulates walking with phone)
  - Weather overlays (rain, fog, mist)
  - Partial occlusion (hand, finger, branch in frame)

Generates N synthetic images per class from existing training data.

Usage:
    python3 scripts/generate_synthetic_tier1.py [--count 100] [--workers 4]
"""

import argparse
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "TrainingData_split" / "train"
OUTPUT_DIR = PROJECT_ROOT / "TrainingData_split" / "train"  # Add directly to training set

CLASSES = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]


def random_motion_blur(img: Image.Image) -> Image.Image:
    """Simulate phone shake / walking motion blur."""
    angle = random.choice([0, 45, 90, 135])
    size = random.randint(3, 7)
    kernel = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(kernel)

    if angle == 0:
        draw.line([(0, size // 2), (size - 1, size // 2)], fill=255, width=1)
    elif angle == 90:
        draw.line([(size // 2, 0), (size // 2, size - 1)], fill=255, width=1)
    elif angle == 45:
        draw.line([(0, size - 1), (size - 1, 0)], fill=255, width=1)
    else:
        draw.line([(0, 0), (size - 1, size - 1)], fill=255, width=1)

    # Apply as gaussian blur approximation (PIL doesn't do directional blur natively)
    radius = random.uniform(0.5, 2.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def simulate_dappled_sunlight(img: Image.Image) -> Image.Image:
    """Simulate patches of light filtering through tree canopy."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Create random circular bright spots
    mask = np.ones((h, w), dtype=np.float32)
    num_spots = random.randint(3, 8)
    for _ in range(num_spots):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(20, 80)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        spot = np.clip(1.0 - dist / radius, 0, 1)
        intensity = random.uniform(0.3, 0.8)
        mask += spot * intensity

    arr = arr * mask[:, :, np.newaxis]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def simulate_shade(img: Image.Image) -> Image.Image:
    """Simulate deep forest shade — darker, bluer tones."""
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.4, 0.7))

    # Slight blue tint (shade is bluer)
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] *= random.uniform(0.85, 0.95)  # Reduce red
    arr[:, :, 1] *= random.uniform(0.90, 0.98)  # Slight green reduction
    arr[:, :, 2] *= random.uniform(1.0, 1.1)    # Boost blue slightly
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def simulate_overexposure(img: Image.Image) -> Image.Image:
    """Simulate direct sunlight overexposure (washed out highlights)."""
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(1.3, 1.8))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.6, 0.85))
    return img


def add_rain_overlay(img: Image.Image) -> Image.Image:
    """Add rain streaks to simulate wet conditions."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Create rain streaks
    rain = np.zeros((h, w), dtype=np.float32)
    num_drops = random.randint(100, 400)
    for _ in range(num_drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(5, 20)
        for dy in range(length):
            ny = min(y + dy, h - 1)
            nx = min(x + random.randint(-1, 1), w - 1)
            nx = max(nx, 0)
            rain[ny, nx] = random.uniform(150, 255)

    # Blend rain
    alpha = random.uniform(0.1, 0.3)
    arr = arr + alpha * rain[:, :, np.newaxis]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_fog_overlay(img: Image.Image) -> Image.Image:
    """Add fog/mist effect — reduces contrast, adds white haze."""
    arr = np.array(img, dtype=np.float32)
    fog_intensity = random.uniform(0.15, 0.4)
    arr = arr * (1 - fog_intensity) + 255 * fog_intensity
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # Slight blur for fog
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    return img


def add_partial_occlusion(img: Image.Image) -> Image.Image:
    """Simulate finger/hand/branch partially blocking the camera."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    occlusion_type = random.choice(["corner", "edge", "branch"])

    if occlusion_type == "corner":
        # Dark shape in a corner (finger over lens)
        corner = random.choice(["tl", "tr", "bl", "br"])
        size = random.randint(40, 120)
        color = (random.randint(30, 80),) * 3
        if corner == "tl":
            draw.ellipse([(-size // 2, -size // 2, size, size)], fill=color)
        elif corner == "tr":
            draw.ellipse([(w - size, -size // 2, w + size // 2, size)], fill=color)
        elif corner == "bl":
            draw.ellipse([(-size // 2, h - size, size, h + size // 2)], fill=color)
        else:
            draw.ellipse([(w - size, h - size, w + size // 2, h + size // 2)], fill=color)

    elif occlusion_type == "edge":
        # Dark bar on one edge
        edge = random.choice(["top", "bottom", "left", "right"])
        thickness = random.randint(15, 50)
        color = (random.randint(20, 60),) * 3
        if edge == "top":
            draw.rectangle([0, 0, w, thickness], fill=color)
        elif edge == "bottom":
            draw.rectangle([0, h - thickness, w, h], fill=color)
        elif edge == "left":
            draw.rectangle([0, 0, thickness, h], fill=color)
        else:
            draw.rectangle([w - thickness, 0, w, h], fill=color)

    else:  # branch
        # Diagonal dark line (branch across frame)
        color = (random.randint(40, 90), random.randint(30, 60), random.randint(10, 30))
        x1, y1 = random.randint(0, w), random.choice([0, h])
        x2, y2 = random.randint(0, w), random.choice([0, h])
        draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(3, 10))

    return img


def generate_synthetic_image(source_path: Path, output_path: Path) -> bool:
    """Apply random combination of augmentations to create a synthetic training image."""
    try:
        img = Image.open(source_path).convert("RGB")

        # Resize to consistent size
        img = img.resize((256, 256), Image.LANCZOS)

        # Random crop to slightly different framing
        if random.random() < 0.7:
            left = random.randint(0, 24)
            top = random.randint(0, 24)
            img = img.crop((left, top, left + 224, top + 224))
            img = img.resize((256, 256), Image.LANCZOS)

        # Apply random subset of augmentations
        augmentations = []

        # Lighting (pick one)
        lighting = random.choice(["none", "dappled", "shade", "overexposure"])
        if lighting == "dappled":
            img = simulate_dappled_sunlight(img)
            augmentations.append("dappled")
        elif lighting == "shade":
            img = simulate_shade(img)
            augmentations.append("shade")
        elif lighting == "overexposure":
            img = simulate_overexposure(img)
            augmentations.append("overexposed")

        # Weather (optional)
        if random.random() < 0.2:
            weather = random.choice(["rain", "fog"])
            if weather == "rain":
                img = add_rain_overlay(img)
                augmentations.append("rain")
            else:
                img = add_fog_overlay(img)
                augmentations.append("fog")

        # Motion blur (optional — simulates walking)
        if random.random() < 0.25:
            img = random_motion_blur(img)
            augmentations.append("blur")

        # Partial occlusion (optional)
        if random.random() < 0.15:
            img = add_partial_occlusion(img)
            augmentations.append("occluded")

        # Color jitter
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.4))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))

        # Random flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=(0, 0, 0))

        img.save(output_path, "JPEG", quality=85)
        return True

    except Exception as e:
        print(f"  Error processing {source_path.name}: {e}")
        return False


def generate_for_class(class_name: str, count: int) -> int:
    """Generate synthetic images for one class."""
    source_dir = TRAIN_DIR / class_name
    if not source_dir.exists():
        print(f"  Skipping {class_name}: directory not found")
        return 0

    source_images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.jpeg"))
    # Filter out previously generated synthetic images
    source_images = [p for p in source_images if "_syn_" not in p.name]

    if not source_images:
        print(f"  Skipping {class_name}: no source images found")
        return 0

    print(f"\n  {class_name}: {len(source_images)} source images → generating {count} synthetic")

    generated = 0
    for i in range(count):
        source = random.choice(source_images)
        output_path = source_dir / f"{class_name}_syn_{i:04d}.jpg"
        if generate_synthetic_image(source, output_path):
            generated += 1

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate Tier 1 synthetic training data")
    parser.add_argument("--count", type=int, default=100,
                        help="Synthetic images to generate per class (default: 100)")
    args = parser.parse_args()

    print("=" * 60)
    print("LeafAlert — Tier 1 Synthetic Data Generation")
    print("Augmentation-based: lighting, weather, blur, occlusion")
    print("=" * 60)

    total = 0
    for cls in CLASSES:
        n = generate_for_class(cls, args.count)
        total += n
        print(f"  → Generated {n} synthetic images for {cls}")

    print(f"\n{'=' * 60}")
    print(f"DONE! Generated {total} total synthetic training images")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nNext: run train_model.py to retrain with augmented data")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
