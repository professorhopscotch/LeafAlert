#!/usr/bin/env python3
"""
Knowledge Distillation training for LeafAlert PlantDetector.

Uses a heavyweight teacher model (ViT-Large or ConvNeXt-Large pretrained on
ImageNet-21k) to generate soft labels, then trains the lightweight student
(EfficientNet-B0 + CBAM spatial attention) to match both the soft teacher
outputs and the hard ground-truth labels.

Pipeline:
  1. Fine-tune teacher on our dataset (or load existing checkpoint)
  2. Generate and cache soft labels from teacher
  3. Train student with distillation loss
  4. Export distilled student to CoreML

Usage:
    python3 scripts/distill_model.py
    python3 scripts/distill_model.py --temperature 6 --alpha 0.9
    python3 scripts/distill_model.py --skip-teacher  # reuse cached teacher/soft labels
    python3 scripts/distill_model.py --teacher-model convnextv2_large.fcmae_ft_in22k_in1k

Output:
    LeafAlert/Resources/MLModels/PlantDetector.mlpackage
"""

import argparse
import json
import os
import pickle
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

try:
    import timm
except ImportError:
    print("ERROR: timm is required for the teacher model.")
    print("Install it with: pip install timm")
    sys.exit(1)

import coremltools as ct

# ─── Config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "TrainingData_split" / "train"
TEST_DIR = PROJECT_ROOT / "TrainingData_split" / "test"
OUTPUT_PATH = PROJECT_ROOT / "LeafAlert" / "Resources" / "MLModels" / "PlantDetector.mlpackage"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SOFT_LABELS_PATH = CHECKPOINT_DIR / "soft_labels.pkl"

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 0  # Safe for macOS

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data augmentation params (matching train_model.py)
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0


# ─── Student Model (from train_model.py) ─────────────────────────────

class SpatialAttention(nn.Module):
    """CBAM-style spatial attention."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        mask = self.sigmoid(self.conv(pooled))
        return x * mask


class PlantDetectorNet(nn.Module):
    """EfficientNet-B0 backbone with spatial attention, dual pooling,
    and a bottleneck classifier head."""

    def __init__(self, num_classes: int):
        super().__init__()
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.backbone = efficientnet.features
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

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
        features = self.backbone(x)
        features = self.spatial_attn(features)
        avg = self.avg_pool(features).flatten(1)
        mx = self.max_pool(features).flatten(1)
        combined = torch.cat([avg, mx], dim=1)
        return self.classifier(combined)


# ─── Teacher Model ───────────────────────────────────────────────────

class TeacherModel(nn.Module):
    """Wrapper around a timm pretrained model, adapted to our num_classes."""

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
        )
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─── Dataset with Soft Labels ────────────────────────────────────────

class SoftLabelDataset(Dataset):
    """Wraps an ImageFolder dataset and pairs each sample with its
    precomputed soft label from the teacher."""

    def __init__(self, image_folder_dataset, soft_labels: dict):
        self.dataset = image_folder_dataset
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, hard_label = self.dataset[idx]
        path = self.dataset.samples[idx][0]
        soft_label = self.soft_labels[path]
        return img, hard_label, torch.tensor(soft_label, dtype=torch.float32)


# ─── Data Transforms ─────────────────────────────────────────────────

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
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
])

# Deterministic transform for teacher inference (soft label generation)
inference_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─── Augmentation helpers (from train_model.py) ─────────────────────

def mixup_data(x, y_hard, y_soft, alpha=0.2):
    """Mixup that handles both hard labels and soft label tensors."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_hard_a, y_hard_b = y_hard, y_hard[index]
    y_soft_a, y_soft_b = y_soft, y_soft[index]
    return mixed_x, y_hard_a, y_hard_b, y_soft_a, y_soft_b, lam


def cutmix_data(x, y_hard, y_soft, alpha=1.0):
    """CutMix that handles both hard labels and soft label tensors."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

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

    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)

    y_hard_a, y_hard_b = y_hard, y_hard[index]
    y_soft_a, y_soft_b = y_soft, y_soft[index]
    return mixed_x, y_hard_a, y_hard_b, y_soft_a, y_soft_b, lam


# ─── Utility ─────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using: CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using: CPU")
    return device


def compute_class_weights(dataset) -> torch.Tensor:
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
    weights = weights / weights.mean()
    return weights


def evaluate(model, dataloader, device) -> tuple:
    """Evaluate model, return accuracy and per-class stats."""
    model.eval()
    correct = 0
    total = 0
    class_correct = Counter()
    class_total = Counter()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for pred, lab in zip(predicted, labels):
                class_total[lab.item()] += 1
                if pred == lab:
                    class_correct[lab.item()] += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, class_correct, class_total


def print_per_class_accuracy(class_correct, class_total, class_names):
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

    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced_model = torch.jit.trace(model, example_input)

    scale = 1.0 / 255.0
    bias = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                scale=1.0 / (255.0 * 0.226),
                bias=bias,
                color_layout="RGB",
            )
        ],
        classifier_config=ct.ClassifierConfig(class_names),
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.author = "LeafAlert"
    mlmodel.short_description = (
        "Detects poison ivy, poison oak, and poison sumac from camera images. "
        "EfficientNet-B0 with spatial attention (v4-distilled) trained via knowledge "
        "distillation from a large teacher model on iNaturalist research-grade observations. "
        f"Classes: {', '.join(class_names)}"
    )
    mlmodel.license = "For use with LeafAlert app"
    mlmodel.version = "4.1.0"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    print(f"\nCore ML model saved to: {output_path}")

    total_size = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    )
    print(f"Model size: {total_size / 1024 / 1024:.1f} MB")


# ─── Phase 1: Fine-tune Teacher ─────────────────────────────────────

def train_teacher(model_name: str, num_classes: int, class_names: list,
                  train_loader, test_loader, device, epochs: int = 20) -> TeacherModel:
    """Fine-tune a large pretrained teacher on our dataset."""
    checkpoint_path = CHECKPOINT_DIR / f"teacher_{model_name.replace('/', '_')}.pth"

    if checkpoint_path.exists():
        print(f"\nFound existing teacher checkpoint: {checkpoint_path}")
        teacher = TeacherModel(model_name, num_classes)
        teacher.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        teacher = teacher.to(device)
        acc, cc, ct_map = evaluate(teacher, test_loader, device)
        print(f"Teacher accuracy (loaded): {acc:.1%}")
        print_per_class_accuracy(cc, ct_map, class_names)
        return teacher

    print(f"\n{'=' * 60}")
    print(f"Phase 1: Fine-tuning teacher model")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"{'=' * 60}")

    teacher = TeacherModel(model_name, num_classes)
    teacher = teacher.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in teacher.parameters())
    trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"  Teacher parameters: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable")

    # Phase 1a: freeze backbone, train head only (5 epochs)
    head_epochs = min(5, epochs // 2)
    finetune_epochs = epochs - head_epochs

    # Freeze everything except the classifier head
    for name, param in teacher.model.named_parameters():
        if "head" not in name and "fc" not in name and "classifier" not in name:
            param.requires_grad = False

    trainable_now = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"\n--- Teacher Phase 1a: Head only ({head_epochs} epochs, {trainable_now} params) ---")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, teacher.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=head_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(head_epochs):
        teacher.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        test_acc, _, _ = evaluate(teacher, test_loader, device)
        print(f"  Epoch {epoch + 1:2d} | Train Loss: {train_loss:.4f}  Acc: {train_acc:.1%} | Test Acc: {test_acc:.1%}")
        scheduler.step()

    # Phase 1b: unfreeze and fine-tune full model
    print(f"\n--- Teacher Phase 1b: Full fine-tune ({finetune_epochs} epochs) ---")
    for param in teacher.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(teacher.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(finetune_epochs):
        teacher.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        test_acc, cc, ct_map = evaluate(teacher, test_loader, device)
        print(f"  Epoch {head_epochs + epoch + 1:2d} | Train Loss: {train_loss:.4f}  Acc: {train_acc:.1%} | Test Acc: {test_acc:.1%}")
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}

    if best_state is not None:
        teacher.load_state_dict(best_state)
        teacher = teacher.to(device)

    print(f"\nBest teacher accuracy: {best_acc:.1%}")
    acc, cc, ct_map = evaluate(teacher, test_loader, device)
    print_per_class_accuracy(cc, ct_map, class_names)

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(teacher.state_dict(), checkpoint_path)
    print(f"Teacher checkpoint saved: {checkpoint_path}")

    return teacher


# ─── Phase 2: Generate Soft Labels ──────────────────────────────────

def generate_soft_labels(teacher: TeacherModel, dataset_path: Path,
                         device, temperature: float) -> dict:
    """Run teacher inference on all training images and cache logits."""
    if SOFT_LABELS_PATH.exists():
        print(f"\nLoading cached soft labels from: {SOFT_LABELS_PATH}")
        with open(SOFT_LABELS_PATH, "rb") as f:
            cached = pickle.load(f)
        # Verify temperature matches
        if cached.get("temperature") == temperature:
            print(f"  Cached labels: {len(cached['labels'])} images, T={cached['temperature']}")
            return cached["labels"]
        else:
            print(f"  Temperature mismatch (cached={cached.get('temperature')}, requested={temperature}). Regenerating.")

    print(f"\n{'=' * 60}")
    print(f"Phase 2: Generating soft labels from teacher")
    print(f"  Temperature: {temperature}")
    print(f"{'=' * 60}")

    # Use deterministic transforms for consistent soft labels
    dataset = datasets.ImageFolder(str(dataset_path), transform=inference_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    teacher.eval()
    soft_labels = {}
    processed = 0

    with torch.no_grad():
        sample_idx = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = teacher(inputs)
            # Store temperature-scaled softmax probabilities
            soft_probs = F.softmax(logits / temperature, dim=1)

            for i in range(inputs.size(0)):
                path = dataset.samples[sample_idx][0]
                soft_labels[path] = soft_probs[i].cpu().tolist()
                sample_idx += 1

            processed += inputs.size(0)
            if processed % 200 == 0 or processed == len(dataset):
                print(f"  Processed {processed}/{len(dataset)} images")

    # Cache to disk
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cache_data = {"temperature": temperature, "labels": soft_labels}
    with open(SOFT_LABELS_PATH, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"  Soft labels cached to: {SOFT_LABELS_PATH}")

    return soft_labels


# ─── Phase 3: Distillation Training ─────────────────────────────────

def distillation_loss(student_logits, soft_targets, hard_targets,
                      temperature: float, alpha: float,
                      class_weights=None):
    """
    Combined distillation loss:
      L = alpha * T^2 * KL(student_soft || teacher_soft)
        + (1 - alpha) * CE(student, hard_labels)

    The T^2 scaling compensates for the reduced gradient magnitude
    when using temperature-scaled softmax.
    """
    # Soft loss: KL divergence between temperature-scaled distributions
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    # soft_targets are already softmax probabilities from generate_soft_labels
    kl_loss = F.kl_div(student_soft, soft_targets, reduction="batchmean")
    soft_loss = (temperature ** 2) * kl_loss

    # Hard loss: standard cross-entropy with class weights
    if class_weights is not None:
        hard_loss = F.cross_entropy(student_logits, hard_targets,
                                    weight=class_weights, label_smoothing=0.1)
    else:
        hard_loss = F.cross_entropy(student_logits, hard_targets, label_smoothing=0.1)

    return alpha * soft_loss + (1 - alpha) * hard_loss


def train_distill_epoch(student, dataloader, optimizer, device,
                        temperature: float, alpha: float,
                        class_weights=None, use_augmix=True) -> tuple:
    """Train student for one epoch with distillation loss + augmentation."""
    student.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, hard_labels, soft_targets in dataloader:
        inputs = inputs.to(device)
        hard_labels = hard_labels.to(device)
        soft_targets = soft_targets.to(device)

        optimizer.zero_grad()

        if use_augmix and random.random() < 0.5:
            # CutMix with soft labels
            mixed_inputs, y_hard_a, y_hard_b, y_soft_a, y_soft_b, lam = \
                cutmix_data(inputs, hard_labels, soft_targets, CUTMIX_ALPHA)
            outputs = student(mixed_inputs)
            loss_a = distillation_loss(outputs, y_soft_a, y_hard_a,
                                       temperature, alpha, class_weights)
            loss_b = distillation_loss(outputs, y_soft_b, y_hard_b,
                                       temperature, alpha, class_weights)
            loss = lam * loss_a + (1 - lam) * loss_b
            # Track accuracy on unaugmented inputs
            with torch.no_grad():
                _, predicted = student(inputs).max(1)
        elif use_augmix:
            # Mixup with soft labels
            mixed_inputs, y_hard_a, y_hard_b, y_soft_a, y_soft_b, lam = \
                mixup_data(inputs, hard_labels, soft_targets, MIXUP_ALPHA)
            outputs = student(mixed_inputs)
            loss_a = distillation_loss(outputs, y_soft_a, y_hard_a,
                                       temperature, alpha, class_weights)
            loss_b = distillation_loss(outputs, y_soft_b, y_hard_b,
                                       temperature, alpha, class_weights)
            loss = lam * loss_a + (1 - lam) * loss_b
            with torch.no_grad():
                _, predicted = student(inputs).max(1)
        else:
            outputs = student(inputs)
            loss = distillation_loss(outputs, soft_targets, hard_labels,
                                     temperature, alpha, class_weights)
            _, predicted = outputs.max(1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += hard_labels.size(0)
        correct += predicted.eq(hard_labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_student_distilled(student, soft_labels: dict, train_dataset,
                            test_loader, device, class_names: list,
                            temperature: float, alpha: float,
                            phase1_epochs: int, phase2_epochs: int):
    """Two-phase distillation training of the student model."""
    print(f"\n{'=' * 60}")
    print(f"Phase 3: Distillation training of student")
    print(f"  Temperature: {temperature}")
    print(f"  Alpha (soft label weight): {alpha}")
    print(f"  Phase 1 (head only): {phase1_epochs} epochs")
    print(f"  Phase 2 (full fine-tune): {phase2_epochs} epochs")
    print(f"{'=' * 60}")

    # Wrap training dataset with soft labels
    distill_dataset = SoftLabelDataset(train_dataset, soft_labels)
    distill_loader = DataLoader(
        distill_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True,
    )

    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # Freeze backbone for Phase 1
    for param in student.backbone.parameters():
        param.requires_grad = False

    # ─── Phase 1: Train head + attention only ─────────────────────
    print(f"\n--- Student Phase 1: Head + attention ({phase1_epochs} epochs) ---")
    phase1_params = list(student.classifier.parameters()) + list(student.spatial_attn.parameters())
    optimizer = optim.Adam(phase1_params, lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)

    for epoch in range(phase1_epochs):
        train_loss, train_acc = train_distill_epoch(
            student, distill_loader, optimizer, device,
            temperature, alpha, class_weights,
        )
        test_acc, cc, ct_map = evaluate(student, test_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch + 1:2d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1%} | "
            f"Test Acc: {test_acc:.1%} | LR: {lr:.6f}"
        )
        scheduler.step()

    print_per_class_accuracy(cc, ct_map, class_names)

    # ─── Phase 2: Full fine-tune with discriminative LR ───────────
    print(f"\n--- Student Phase 2: Full fine-tune ({phase2_epochs} epochs) ---")
    for param in student.backbone.parameters():
        param.requires_grad = True

    base_lr = 1e-3
    param_groups = [
        {"params": list(student.backbone[:4].parameters()), "lr": base_lr / 50},
        {"params": list(student.backbone[4:6].parameters()), "lr": base_lr / 10},
        {"params": list(student.backbone[6:].parameters()), "lr": base_lr / 3},
        {"params": list(student.spatial_attn.parameters()), "lr": base_lr / 3},
        {"params": list(student.classifier.parameters()), "lr": base_lr},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)

    best_acc = 0.0
    best_state = None
    patience = 0
    max_patience = 8

    for epoch in range(phase2_epochs):
        train_loss, train_acc = train_distill_epoch(
            student, distill_loader, optimizer, device,
            temperature, alpha, class_weights,
        )
        test_acc, cc, ct_map = evaluate(student, test_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {phase1_epochs + epoch + 1:2d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1%} | "
            f"Test Acc: {test_acc:.1%} | LR: {lr:.6f}"
        )
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n  Early stopping at epoch {phase1_epochs + epoch + 1} "
                      f"(no improvement for {max_patience} epochs)")
                break

    if best_state is not None:
        student.load_state_dict(best_state)
        student = student.to(device)
        print(f"\nLoaded best student checkpoint (test acc: {best_acc:.1%})")

    return student, best_acc


# ─── Phase 4: Evaluation Comparison ─────────────────────────────────

def evaluate_comparison(student, test_loader, device, class_names,
                        baseline_checkpoint=None):
    """Evaluate distilled student and optionally compare to baseline."""
    print(f"\n{'=' * 60}")
    print(f"Phase 4: Evaluation")
    print(f"{'=' * 60}")

    # Distilled student accuracy
    acc, cc, ct_map = evaluate(student, test_loader, device)
    print(f"\nDistilled student accuracy: {acc:.1%}")
    print_per_class_accuracy(cc, ct_map, class_names)

    # Compare to baseline if available
    if baseline_checkpoint and Path(baseline_checkpoint).exists():
        print(f"\nLoading baseline student from: {baseline_checkpoint}")
        num_classes = len(class_names)
        baseline = PlantDetectorNet(num_classes)
        baseline.load_state_dict(torch.load(baseline_checkpoint, map_location="cpu", weights_only=True))
        baseline = baseline.to(device)
        base_acc, base_cc, base_ct = evaluate(baseline, test_loader, device)
        print(f"Baseline student accuracy: {base_acc:.1%}")
        print_per_class_accuracy(base_cc, base_ct, class_names)

        diff = acc - base_acc
        sign = "+" if diff >= 0 else ""
        print(f"\nDistillation improvement: {sign}{diff:.1%}")
    else:
        print("\nNo baseline checkpoint found for comparison.")
        print("To compare, save a baseline with: --save-baseline")


# ─── Main ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge distillation training for LeafAlert PlantDetector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--teacher-model", type=str,
        default="vit_large_patch16_224.augreg_in21k_ft_in1k",
        help="timm model name for the teacher",
    )
    parser.add_argument(
        "--temperature", type=float, default=4.0,
        help="Temperature for soft label generation and distillation loss",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.8,
        help="Weight for soft label loss (1-alpha for hard label CE loss)",
    )
    parser.add_argument(
        "--teacher-epochs", type=int, default=20,
        help="Number of epochs to fine-tune the teacher",
    )
    parser.add_argument(
        "--student-phase1-epochs", type=int, default=10,
        help="Student Phase 1 epochs (head + attention only)",
    )
    parser.add_argument(
        "--student-phase2-epochs", type=int, default=30,
        help="Student Phase 2 epochs (full fine-tune)",
    )
    parser.add_argument(
        "--skip-teacher", action="store_true",
        help="Skip teacher training, use existing checkpoint and cached soft labels",
    )
    parser.add_argument(
        "--regenerate-soft-labels", action="store_true",
        help="Force regeneration of soft labels even if cached",
    )
    parser.add_argument(
        "--baseline-checkpoint", type=str, default=None,
        help="Path to baseline student checkpoint for accuracy comparison",
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Save current student (before distillation) as baseline checkpoint",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip CoreML export",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LeafAlert PlantDetector — Knowledge Distillation Training")
    print("=" * 60)
    print(f"  Teacher model:    {args.teacher_model}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  Alpha:            {args.alpha}")
    print(f"  Teacher epochs:   {args.teacher_epochs}")
    print(f"  Student epochs:   {args.student_phase1_epochs} + {args.student_phase2_epochs}")

    device = get_device()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Load datasets ────────────────────────────────────────────
    print(f"\nLoading data from: {TRAIN_DIR}")
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

    # Teacher uses augmented training data too
    train_loader_teacher = DataLoader(
        datasets.ImageFolder(str(TRAIN_DIR), transform=train_transforms),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ─── Phase 1: Teacher training ────────────────────────────────
    if not args.skip_teacher:
        teacher = train_teacher(
            args.teacher_model, num_classes, class_names,
            train_loader_teacher, test_loader, device,
            epochs=args.teacher_epochs,
        )
    else:
        # Load existing teacher checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"teacher_{args.teacher_model.replace('/', '_')}.pth"
        if not checkpoint_path.exists():
            print(f"\nERROR: No teacher checkpoint found at {checkpoint_path}")
            print("Run without --skip-teacher first to train the teacher.")
            sys.exit(1)
        teacher = TeacherModel(args.teacher_model, num_classes)
        teacher.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        teacher = teacher.to(device)
        acc, cc, ct_map = evaluate(teacher, test_loader, device)
        print(f"\nLoaded teacher checkpoint: {acc:.1%} accuracy")

    # ─── Phase 2: Generate soft labels ────────────────────────────
    if args.regenerate_soft_labels and SOFT_LABELS_PATH.exists():
        SOFT_LABELS_PATH.unlink()

    soft_labels = generate_soft_labels(
        teacher, TRAIN_DIR, device, args.temperature,
    )

    # Free teacher from GPU memory
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ─── Optionally save baseline ─────────────────────────────────
    baseline_path = args.baseline_checkpoint
    if args.save_baseline:
        baseline_path = str(CHECKPOINT_DIR / "student_baseline.pth")
        print(f"\nSaving baseline student checkpoint...")
        baseline_student = PlantDetectorNet(num_classes)
        # Train baseline briefly with standard CE to have a fair comparison
        # (or just save the untrained weights as a reference)
        torch.save(baseline_student.state_dict(), baseline_path)
        print(f"  Saved to: {baseline_path}")

    # ─── Phase 3: Student distillation ────────────────────────────
    student = PlantDetectorNet(num_classes)
    student = student.to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"\nStudent parameters: {student_params / 1e6:.1f}M")

    student, best_acc = train_student_distilled(
        student, soft_labels, train_dataset, test_loader, device,
        class_names, args.temperature, args.alpha,
        args.student_phase1_epochs, args.student_phase2_epochs,
    )

    # Save distilled student checkpoint
    distilled_path = CHECKPOINT_DIR / "student_distilled.pth"
    torch.save(student.state_dict(), distilled_path)
    print(f"Distilled student checkpoint saved: {distilled_path}")

    # ─── Phase 4: Evaluation ──────────────────────────────────────
    evaluate_comparison(student, test_loader, device, class_names,
                        baseline_checkpoint=baseline_path)

    # ─── CoreML Export ────────────────────────────────────────────
    if not args.no_export:
        print(f"\n--- Converting to Core ML ---")
        convert_to_coreml(student, class_names, OUTPUT_PATH)

    # ─── Summary ──────────────────────────────────────────────────
    total_epochs = args.student_phase1_epochs + args.student_phase2_epochs
    print(f"\n{'=' * 60}")
    print(f"DONE! Distilled model ready.")
    print(f"\nDistillation config:")
    print(f"  Teacher:          {args.teacher_model}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  Alpha:            {args.alpha}")
    print(f"  Student epochs:   {total_epochs}")
    print(f"  Best test acc:    {best_acc:.1%}")
    print(f"\nArchitecture (unchanged):")
    print(f"  - EfficientNet-B0 backbone with CBAM spatial attention")
    print(f"  - Dual pooling (avg + max) -> 2560-dim feature vector")
    print(f"  - Bottleneck classifier (2560->512->128->{num_classes})")
    if not args.no_export:
        print(f"\nCoreML model: {OUTPUT_PATH}")
    print(f"Checkpoint:   {distilled_path}")
    print(f"\nRebuild the app to include the new model.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
