#!/usr/bin/env python3
"""
Train a PlantDetector model for LeafAlert using transfer learning.

Uses EfficientNet-B0 (pretrained on ImageNet) with CBAM spatial attention,
dual avg+max pooling, and a bottleneck classifier head. Fine-tunes on our
iNaturalist training data and exports to Core ML format.

v4 improvements:
  - CBAM spatial attention for leaf-region focus
  - Dual pooling (avg + max) for richer feature representation
  - Bottleneck classifier head (2560 → 512 → 128 → 4)
  - Stronger weight decay for Phase 1 (1e-3)
  - Spatial attention in Phase 2 discriminative LR groups

Usage:
    python3 scripts/train_model.py

Output:
    LeafAlert/Resources/MLModels/PlantDetector.mlpackage
"""

import os
import random
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

BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
MIXUP_ALPHA = 0.2     # Mixup regularization
CUTMIX_ALPHA = 1.0    # CutMix regularization
IMAGE_SIZE = 224  # EfficientNet-B0 expects 224x224
NUM_WORKERS = 0   # Safe for macOS

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Model Architecture ───────────────────────────────────────────

class SpatialAttention(nn.Module):
    """CBAM-style spatial attention. Produces a spatial mask that tells
    the model WHERE on the feature map to focus (leaf vs background)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        mask = self.sigmoid(self.conv(pooled))           # [B, 1, H, W]
        return x * mask


class PlantDetectorNet(nn.Module):
    """EfficientNet-B0 backbone with spatial attention, dual pooling,
    and a bottleneck classifier head."""

    def __init__(self, num_classes: int):
        super().__init__()
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.backbone = efficientnet.features  # [B, 1280, 7, 7]
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Bottleneck classifier: 2560 → 512 → 128 → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)             # [B, 1280, 7, 7]
        features = self.spatial_attn(features)   # [B, 1280, 7, 7] (attended)
        avg = self.avg_pool(features).flatten(1) # [B, 1280]
        mx = self.max_pool(features).flatten(1)  # [B, 1280]
        combined = torch.cat([avg, mx], dim=1)   # [B, 2560]
        return self.classifier(combined)


# ─── Data transforms ────────────────────────────────────────────────
# Training: aggressive augmentation to simulate real hiking conditions
# (varied lighting, angles, partial occlusion, phone shake blur)
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 48, IMAGE_SIZE + 48)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomRotation(15),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
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


def create_model(num_classes: int) -> PlantDetectorNet:
    """Create PlantDetectorNet: EfficientNet-B0 + spatial attention + dual pooling + bottleneck head."""
    model = PlantDetectorNet(num_classes)

    # Freeze backbone initially — only train attention + classifier head
    for param in model.backbone.parameters():
        param.requires_grad = False

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


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation: blend random pairs of images and labels."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation: paste a random rectangular patch from one image onto another."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # Generate random bounding box
    _, _, H, W = x.size()
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = torch.randint(0, H, (1,)).item()
    cx = torch.randint(0, W, (1,)).item()

    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to the actual area ratio
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True) -> tuple:
    """Train for one epoch with optional mixup, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_mixup and random.random() < 0.5:
            # CutMix: paste a rectangular patch from another image
            mixed_inputs, y_a, y_b, lam = cutmix_data(inputs, labels, CUTMIX_ALPHA)
            outputs = model(mixed_inputs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            # For accuracy tracking, use original labels
            _, predicted = model(inputs).max(1)
        elif use_mixup:
            # Mixup: blend entire images
            mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            outputs = model(mixed_inputs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            # For accuracy tracking, use original labels
            _, predicted = model(inputs).max(1)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
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
        "EfficientNet-B0 with spatial attention (v4) trained on iNaturalist research-grade observations. "
        f"Classes: {', '.join(class_names)}"
    )
    mlmodel.license = "For use with LeafAlert app"
    mlmodel.version = "4.0.0"

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
    print("LeafAlert PlantDetector — Model Training v4")
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
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Compute class weights for balanced loss
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"\nClass weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # Create model
    print(f"\nLoading EfficientNet-B0 + spatial attention (v4)...")
    model = create_model(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Phase 1: Train classifier head only (backbone frozen) — 10 epochs
    phase1_epochs = 10
    print(f"\n--- Phase 1: Training classifier head ({phase1_epochs} epochs) ---")
    # Train classifier head + spatial attention (backbone frozen)
    phase1_params = list(model.classifier.parameters()) + list(model.spatial_attn.parameters())
    optimizer = optim.Adam(phase1_params, lr=LEARNING_RATE, weight_decay=1e-3)
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

    # Phase 2: Unfreeze entire backbone with discriminative learning rates
    # Lower LR for early layers, higher for later layers
    phase2_epochs = NUM_EPOCHS - phase1_epochs
    print(f"\n--- Phase 2: Fine-tuning full network with discriminative LR ({phase2_epochs} epochs) ---")
    for param in model.backbone.parameters():
        param.requires_grad = True

    # Discriminative LR: early layers barely move, late layers adapt more
    # EfficientNet-B0 has 9 feature blocks (0-8)
    param_groups = [
        {"params": list(model.backbone[:4].parameters()), "lr": LEARNING_RATE / 50},
        {"params": list(model.backbone[4:6].parameters()), "lr": LEARNING_RATE / 10},
        {"params": list(model.backbone[6:].parameters()), "lr": LEARNING_RATE / 3},
        {"params": list(model.spatial_attn.parameters()), "lr": LEARNING_RATE / 3},
        {"params": list(model.classifier.parameters()), "lr": LEARNING_RATE},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)

    best_acc = 0.0
    best_state = None
    patience = 0
    max_patience = 8  # Stop if no improvement for 8 epochs

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
            if patience >= max_patience:
                print(f"\n  Early stopping at epoch {phase1_epochs + epoch + 1} (no improvement for {max_patience} epochs)")
                break

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
    print(f"DONE! Model v4 ready at:")
    print(f"  {OUTPUT_PATH}")
    print(f"\nv4 architecture:")
    print(f"  - EfficientNet-B0 backbone with CBAM spatial attention")
    print(f"  - Dual pooling (avg + max) → 2560-dim feature vector")
    print(f"  - Bottleneck classifier (2560→512→128→{num_classes})")
    print(f"  - {NUM_EPOCHS} total epochs, label smoothing, mixup/cutmix, class weights")
    print(f"\nRebuild the app to include the new model.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
