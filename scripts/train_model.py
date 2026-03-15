#!/usr/bin/env python3
"""
Train a PlantDetector model for LeafAlert using transfer learning.

Uses MobileNetV2 (pretrained on ImageNet) as the backbone, replaces the
classification head with 4 classes (poison_ivy, poison_oak, poison_sumac,
safe_plants), fine-tunes on our iNaturalist training data, and exports
to Core ML format.

Usage:
    python3 scripts/train_model.py

Output:
    LeafAlert/Resources/MLModels/PlantDetector.mlmodel
"""

import os
import sys
from pathlib import Path

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
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
IMAGE_SIZE = 224  # MobileNetV2 expects 224x224
NUM_WORKERS = 0   # Safe for macOS

# ─── Data transforms ────────────────────────────────────────────────
# Training: augment to improve generalization (random flips, rotations,
# color jitter — simulates varied lighting on hiking trails)
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Testing: just resize and normalize (no augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def create_model(num_classes: int) -> nn.Module:
    """Load MobileNetV2 pretrained on ImageNet, replace classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze backbone initially — only train the new classifier head
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier: original is Linear(1280, 1000)
    # New: Linear(1280, num_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, num_classes),
    )

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> float:
    """Train for one epoch, return average loss."""
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
    """Evaluate model on test set, return loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def convert_to_coreml(model, class_names: list, output_path: Path):
    """Convert PyTorch model to Core ML format."""
    model.eval()
    model.cpu()

    # Trace the model with example input
    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced_model = torch.jit.trace(model, example_input)

    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                scale=1.0 / (0.226 * 255.0),  # Approximate combined scale
                bias=[
                    -0.485 / 0.229,
                    -0.456 / 0.224,
                    -0.406 / 0.225,
                ],
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
        "Fine-tuned MobileNetV2 trained on iNaturalist research-grade observations."
    )
    mlmodel.license = "For use with LeafAlert app"
    mlmodel.version = "1.0.0"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"\nCore ML model saved to: {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    print("=" * 60)
    print("LeafAlert PlantDetector — Model Training")
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

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Create model
    print(f"\nLoading MobileNetV2 (pretrained on ImageNet)...")
    model = create_model(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Phase 1: Train classifier head only (backbone frozen)
    print(f"\n--- Phase 1: Training classifier head ({NUM_EPOCHS // 2} epochs) ---")
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS // 2):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"  Epoch {epoch + 1:2d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1%} | "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.1%}"
        )

    # Phase 2: Unfreeze backbone and fine-tune everything with lower LR
    print(f"\n--- Phase 2: Fine-tuning full network ({NUM_EPOCHS - NUM_EPOCHS // 2} epochs) ---")
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)

    best_acc = 0.0
    for epoch in range(NUM_EPOCHS // 2, NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"  Epoch {epoch + 1:2d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1%} | "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.1%}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = model.state_dict().copy()

    # Load best model
    if best_acc > 0:
        model.load_state_dict(best_state)
        print(f"\nBest test accuracy: {best_acc:.1%}")

    # Final evaluation
    print(f"\n--- Final Evaluation ---")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.1%}")

    # Convert to Core ML
    print(f"\n--- Converting to Core ML ---")
    convert_to_coreml(model, class_names, OUTPUT_PATH)

    print(f"\n{'=' * 60}")
    print(f"DONE! Model ready at:")
    print(f"  {OUTPUT_PATH}")
    print(f"\nRebuild the app to include the new model.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
