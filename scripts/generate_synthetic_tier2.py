#!/usr/bin/env python3
"""
Tier 2 Synthetic Data: AI-generated images via Stable Diffusion.

Runs Stable Diffusion XL (SDXL) locally on Apple Silicon to generate
photorealistic training images of toxic plants in realistic hiking scenarios.

Optimized for M4 with 32GB RAM using the diffusers library with MPS backend.

Prerequisites:
    pip3 install diffusers transformers accelerate safetensors

Usage:
    python3 scripts/generate_synthetic_tier2.py [--count 50] [--model sdxl-turbo]

Models:
    sdxl-turbo  — Fast (~2s/image), good quality, 6GB VRAM (default)
    sdxl        — Slower (~30s/image), best quality, 10GB VRAM
"""

import argparse
import random
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "TrainingData_split" / "train"

# ─── Prompt templates ────────────────────────────────────────────────
# Each template is designed to produce images resembling what a hiker
# would see through their iPhone camera on a trail.

PLANT_PROMPTS = {
    "poison_ivy": [
        "close-up photograph of poison ivy plant with three shiny leaves, growing along a forest trail, natural lighting, iPhone photo, high detail",
        "poison ivy vine climbing a tree trunk in a deciduous forest, clusters of three pointed leaflets, summer, natural sunlight filtering through canopy",
        "poison ivy leaves turning red in autumn, three-leaf clusters on forest floor, hiking trail visible, mobile phone camera quality",
        "young poison ivy sprouts with reddish new growth, three leaflets per leaf, growing near trail edge, spring season, naturalistic photo",
        "dense patch of poison ivy groundcover along woodland path, characteristic three-leaf pattern, varied leaf sizes, outdoor natural lighting",
        "poison ivy growing on a fallen log beside hiking trail, glossy green trifoliate leaves, forest undergrowth background",
        "poison ivy plant with white berries in late summer, three shiny leaves, woodland setting, realistic photo",
        "hairy poison ivy vine on tree bark with aerial rootlets, three-leaf clusters growing upward, dappled forest light",
    ],
    "poison_oak": [
        "close-up photograph of poison oak with rounded oak-shaped leaves in groups of three, Pacific coast woodland, iPhone photo quality",
        "poison oak shrub growing along California hiking trail, scalloped three-leaf clusters, golden hillside background, natural sunlight",
        "poison oak leaves turning red in fall, rounded lobed leaflets in threes, dry grass background, West Coast trail setting",
        "young poison oak with green glossy leaves, three rounded leaflets per leaf, chaparral landscape, realistic mobile phone photo",
        "dense poison oak thicket along narrow trail, characteristic three rounded leaves, mixed with other shrubs, natural forest lighting",
        "poison oak growing near rocky outcrop on hillside trail, three lobed shiny leaves, California native plants around",
        "poison oak branch with small white flowers, three oak-shaped leaflets, spring season woodland, naturalistic photograph",
        "poison oak in mixed vegetation along fire road, identifying three-leaf clusters with rounded edges, outdoor lighting",
    ],
    "poison_sumac": [
        "close-up photograph of poison sumac with pinnate compound leaves, growing in wetland area, 7-13 smooth-edged leaflets per leaf, iPhone photo",
        "poison sumac shrub in swampy woodland, compound leaves with paired smooth leaflets, standing water visible, natural lighting",
        "poison sumac leaves turning brilliant red in autumn, compound leaf structure with smooth margins, boggy forest setting",
        "young poison sumac tree near pond edge, elongated smooth leaflets arranged on central stem, wetland vegetation, realistic photo",
        "poison sumac branch with drooping white berry clusters, compound leaves with untoothed leaflets, swamp background",
        "poison sumac growing along boardwalk in marshy area, smooth-margined leaflets in pairs plus terminal leaflet, natural light",
        "poison sumac in summer with bright green pinnate leaves, red central stem, growing in wet lowland forest",
        "tall poison sumac shrub in flooded forest area, distinctive smooth compound leaves, naturalistic mobile photograph",
    ],
    "safe_plants": [
        "close-up of virginia creeper vine with five leaflets, growing on fence, harmless look-alike plant, forest setting, iPhone photo",
        "box elder tree seedling with three compound leaves, NOT poison ivy, woodland trail edge, natural lighting",
        "blackberry bush along hiking trail with thorny stems and compound leaves, safe edible plant, outdoor photo",
        "fragrant sumac shrub with three small lobed leaves, safe sumac species, woodland edge, natural sunlight",
        "wild grape vine leaves along forest trail, large heart-shaped leaves, common trail plant, realistic photo",
        "boston ivy climbing wall with three-pointed leaves, common ornamental vine, urban park setting",
        "jack-in-the-pulpit plant on forest floor, three large leaflets, spring wildflower, shaded woodland",
        "common hog peanut vine with three oval leaflets, woodland groundcover, dappled forest light, naturalistic photo",
    ],
}

# Negative prompts to avoid unrealistic outputs
NEGATIVE_PROMPT = (
    "cartoon, illustration, drawing, painting, art, sketch, anime, "
    "watercolor, digital art, render, 3d, cgi, text, watermark, logo, "
    "blurry, low quality, deformed, ugly, oversaturated, studio lighting, "
    "indoor, artificial background, plastic, fake plant"
)

# Additional variety modifiers appended randomly
LIGHTING_MODIFIERS = [
    "morning golden hour light",
    "midday harsh sunlight with shadows",
    "overcast cloudy day diffused light",
    "late afternoon warm light",
    "dappled sunlight through tree canopy",
    "shaded forest understory lighting",
    "bright direct sunlight",
    "after rain with wet leaves glistening",
]

CAMERA_MODIFIERS = [
    "shot on iPhone 15",
    "smartphone camera quality",
    "mobile phone photograph",
    "casual handheld photo",
    "macro close-up lens",
    "slightly out of focus background bokeh",
]

SEASON_MODIFIERS = [
    "spring season with new growth",
    "summer with full green foliage",
    "early fall with some color change",
    "late autumn with red and orange leaves",
]


def build_prompt(plant_class: str) -> str:
    """Build a varied prompt by combining templates with random modifiers."""
    base = random.choice(PLANT_PROMPTS[plant_class])
    modifiers = [
        random.choice(LIGHTING_MODIFIERS),
        random.choice(CAMERA_MODIFIERS),
    ]
    if random.random() < 0.4:
        modifiers.append(random.choice(SEASON_MODIFIERS))

    return f"{base}, {', '.join(modifiers)}"


def generate_images_sdxl_turbo(plant_class: str, count: int, output_dir: Path):
    """Generate images using SDXL-Turbo (fast, 1-4 step inference)."""
    from diffusers import AutoPipelineForText2Image
    import torch

    print(f"\n  Loading SDXL-Turbo model...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("mps")

    class_dir = output_dir / plant_class
    class_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for i in range(count):
        prompt = build_prompt(plant_class)
        output_path = class_dir / f"{plant_class}_sdxl_{i:04d}.jpg"

        if output_path.exists():
            generated += 1
            continue

        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=4,  # SDXL-Turbo is designed for 1-4 steps
                guidance_scale=0.0,     # Turbo doesn't use guidance
                width=512,
                height=512,
            )
            image = result.images[0]
            image.save(output_path, "JPEG", quality=90)
            generated += 1

            if (i + 1) % 10 == 0:
                print(f"    {plant_class}: {i + 1}/{count} generated")
                gc.collect()

        except Exception as e:
            print(f"    Error generating {output_path.name}: {e}")

    return generated


def generate_images_sdxl(plant_class: str, count: int, output_dir: Path):
    """Generate images using full SDXL (slower, higher quality)."""
    from diffusers import DiffusionPipeline
    import torch

    print(f"\n  Loading SDXL model (this may take a minute)...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe = pipe.to("mps")
    # Enable memory-efficient attention for M4
    pipe.enable_attention_slicing()

    class_dir = output_dir / plant_class
    class_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for i in range(count):
        prompt = build_prompt(plant_class)
        output_path = class_dir / f"{plant_class}_sdxl_{i:04d}.jpg"

        if output_path.exists():
            generated += 1
            continue

        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=512,
                height=512,
            )
            image = result.images[0]
            image.save(output_path, "JPEG", quality=90)
            generated += 1

            if (i + 1) % 5 == 0:
                print(f"    {plant_class}: {i + 1}/{count} generated")
                gc.collect()

        except Exception as e:
            print(f"    Error generating {output_path.name}: {e}")

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate Tier 2 synthetic training data with AI")
    parser.add_argument("--count", type=int, default=50,
                        help="Images to generate per class (default: 50)")
    parser.add_argument("--model", choices=["sdxl-turbo", "sdxl"], default="sdxl-turbo",
                        help="Model to use (default: sdxl-turbo, faster)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Specific classes to generate (default: all)")
    args = parser.parse_args()

    classes = args.classes or list(PLANT_PROMPTS.keys())

    print("=" * 60)
    print("LeafAlert — Tier 2 Synthetic Data Generation")
    print(f"Model: {args.model} | Count: {args.count}/class | Classes: {classes}")
    print("=" * 60)

    generate_fn = generate_images_sdxl_turbo if args.model == "sdxl-turbo" else generate_images_sdxl

    total = 0
    for cls in classes:
        if cls not in PLANT_PROMPTS:
            print(f"\n  Skipping unknown class: {cls}")
            continue

        n = generate_fn(cls, args.count, OUTPUT_DIR)
        total += n
        print(f"  → Generated {n} AI images for {cls}")

        # Free GPU memory between classes
        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"DONE! Generated {total} total AI-generated training images")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Review images for quality: open {OUTPUT_DIR}")
    print(f"  2. Delete any obviously bad generations")
    print(f"  3. Run train_model.py to retrain with synthetic data")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
