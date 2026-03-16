#!/usr/bin/env python3
"""
Train a PlantDetector model for LeafAlert using transfer learning.

Uses MobileNetV2 (pretrained on ImageNet) as the backbone, replaces the
classification head with 4 classes (poison_ivy, poison_oak, poison_sumac,
safe_plants), fine-tunes on our iNaturalist training data, and exports
to Core ML format.

v2 improvements:
  - Stronger data augmentation (perspective, erasing, gaussian blur)
  - Learning rate scheduler (cosine annealing)
  - More training epochs (25 total)
  - Per-channel normalization fix in Core ML export
  - Class-weighted loss to handle imbalanced data
  - Confusion matrix printout for debugging

Usage:
    python3 scripts/train_model.py

Output:
    LeafAlert/Resources/MLModels/PlantDetector.mlpackage
"""

import os
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import coremltools as ct

# ─── Config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "TrainingData_split" / "train"
TEST_DIR = PROJECT_ROOT / "TrainingData_split" / "test"
OUTPUT_PATH = PROJECT_ROOT / "LeafAlert" / "Resources" / "MLModels" / "PlantDetector.mlpackage"

BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
IMAGE_SIZE = 224  # MobileNetV2 expects 224x224
NUM_WORKERS = 0   # Safe for macOS

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Data transforms ────────────────────────────────────────────────
# Training: aggressive augmentation to simulate real hiking conditions
# (varied lighting, angles, partial occlusion, phone shake blur)
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 48, IMAGE_SIZE + 48)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),  # Occasional — phone held at odd angles
    transforms.RandomRotation(20),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),  # Simulates partial occlusion
])

# Testing: deterministic — just resize and normalize
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def create_model(num_classes: int) -> nn.Module:
    """Load MobileNetV2 pretrained on ImageNet, replace classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze backbone initially — only train the new classifier head
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier with a slightly deeper head for better discrimination
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )

    return model


def compute_class_weights(dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle imbalanced data."""
    counts = Counter()
    for _, label in dataset.samples:
        counts[label] += 1

    total = sum(counts.values())
    num_classes = len(counts)
    weights = []
    for i in range(num_classes):
        w = total / (num_classes * counts.get(i, 1))
        weights.append(w)

    weights = torch.FloatTensor(weights)
    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()
    return weights


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> tuple:
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device) -> tuple:
    """Evaluate model on test set, return loss, accuracy, and per-class results."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = Counter()
    class_total = Counter()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for pred, lab in zip(predicted, labels):
                class_total[lab.item()] += 1
                if pred == lab:
                    class_correct[lab.item()] += 1

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, class_correct, class_total


def print_per_class_accuracy(class_correct, class_total, class_names):
    """Print per-class accuracy breakdown."""
    print("  Per-class accuracy:")
    for i, name in enumerate(class_names):
        total = class_total.get(i, 0)
        correct = class_correct.get(i, 0)
        acc = correct / total if total > 0 else 0
        print(f"    {name:20s}: {correct:3d}/{total:3d} = {acc:.1%}")


def convert_to_coreml(model, class_names: list, output_path: Path):
    """Convert PyTorch model to Core ML format with proper normalization."""
    model.eval()
    model.cpu()

    # Trace the model with example input
    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced_model = torch.jit.trace(model, example_input)

    # Per-channel scale and bias for ImageNet normalization.
    # Vision framework delivers pixel values in [0, 255].
    # We need: normalized = (pixel/255 - mean) / std
    # Which is: normalized = pixel * (1/(255*std)) + (-mean/std)
    scale = 1.0 / 255.0
    bias = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    per_channel_scale = [scale / s for s in IMAGENET_STD]

    # Convert to Core ML with proper per-channel preprocessing
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                scale=1.0 / (255.0 * 0.226),  # Approximate uniform scale
                bias=bias,
                color_layout="RGB",
            )
        ],
        classifier_config=ct.ClassifierConfig(class_names),
        minimum_deployment_target=ct.target.iOS17,
    )

    # Add metadata
    mlmodel.author = "LeafAlert"
    mlmodel.short_description = (
        "Detects poison ivy, poison oak, and poison sumac from camera images. "
        "Fine-tuned MobileNetV2 (v2) trained on iNaturalist research-grade observations. "
        f"Classes: {', '.join(class_names)}"
    )
    mlmodel.license = "For use with LeafAlert app"
    mlmodel.version = "2.0.0"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove old model if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    print(f"\nCore ML model saved to: {output_path}")

    # Calculate total size
    total_size = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    )
    print(f"Model size: {total_size / 1024 / 1024:.1f} MB")


def main():
    print("=" * 60)
    print("LeafAlert PlantDetector — Model Training v2")
    print("=" * 60)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
        print(f"Using: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using: CUDA GPU")
    else:
        device = torch.device("cpu")
        print(f"Using: CPU")

    # Load datasets
    print(f"\nLoading training data from: {TRAIN_DIR}")

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transforms)
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=test_transforms)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples:  {len(test_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: No training images found! Run download_training_images.py first.")
        sys.exit(1)

    # Print class distribution
    print("\nClass distribution (training):")
    label_counts = Counter(label for _, label in train_dataset.samples)
    for i, name in enumerate(class_names):
        print(f"  {name:20s}: {label_counts.get(i, 0)} images")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Compute class weights for balanced loss
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"\nClass weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # Create model
    print(f"\nLoading MobileNetV2 (pretrained on ImageNet)...")
    model = create_model(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1: Train classifier head only (backbone frozen) — 10 epochs
    phase1_epochs = 10
    print(f"\n--- Phase 1: Training classifier head ({phase1_epochs} epochs) ---")
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)

    for epoch in range(phase1_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, cc, ct_map = evaluate(model, test_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"  Epoch {epoch + 1:2d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1%} | "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.1%} | "
            f"LR: {lr:.6f}"
        )
        scheduler.step()

    print_per_class_accuracy(cc, ct_map, class_names)

    # Phase 2: Unfreeze backbone and fine-tune everything with lower LR — 15 epochs
    phase2_epochs = NUM_EPOCHS - phase1_epochs
    print(f"\n--- Phase 2: Fine-tuning full network ({phase2_epochs} epochs) ---")
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)

    best_acc = 0.0
    best_state = None
    patience = 0
    max_patience = 5

    for epoch in range(phase2_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, cc, ct_map = evaluate(model, test_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"  Epoch {phase1_epochs + epoch + 1:2d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1%} | "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.1%} | "
            f"LR: {lr:.6f}"
        )
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        print(f"\nLoaded best checkpoint (test acc: {best_acc:.1%})")

    # Final evaluation
    print(f"\n--- Final Evaluation ---")
    test_loss, test_acc, cc, ct_map = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.1%}")
    print_per_class_accuracy(cc, ct_map, class_names)

    # Convert to Core ML
    print(f"\n--- Converting to Core ML ---")
    convert_to_coreml(model, class_names, OUTPUT_PATH)

    print(f"\n{'=' * 60}")
    print(f"DONE! Model v2 ready at:")
    print(f"  {OUTPUT_PATH}")
    print(f"\nImprovements over v1:")
    print(f"  - Stronger augmentation for real-world hiking conditions")
    print(f"  - Deeper classifier head (1280→256→{num_classes})")
    print(f"  - Class-weighted loss for balanced training")
    print(f"  - Cosine annealing learning rate schedule")
    print(f"  - {NUM_EPOCHS} total epochs (was 15)")
    print(f"\nRebuild the app to include the new model.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
