# Model Architecture v4: Attention + Rich Head

## Problem

LeafAlert's EfficientNet-B0 classifier achieves 70% overall test accuracy, with poison ivy (55%) and poison oak (56%) significantly underperforming. The current classifier head is a single `Dropout(0.4) → Linear(1280, 4)` layer — too simple to capture fine-grained leaf distinctions. The model also lacks spatial attention, meaning it gives equal weight to background clutter and leaf regions.

## Goal

Improve classification accuracy — especially on poison ivy/oak — by adding spatial attention, richer classification head, and test-time augmentation. Stay within iOS deployment constraints (~10MB model, <50ms inference on iPhone).

## Constraints

- Must export cleanly to Core ML (iOS 17+)
- Model size budget: <10MB (current: 7.8MB)
- Training dataset: 1,737 images (small — must avoid overfitting)
- Inference latency: <50ms per frame on iPhone (with TTA)

## Architecture Changes

### 1. Spatial Attention (CBAM-style)

Applied after the EfficientNet-B0 backbone features, before pooling.

EfficientNet-B0 already contains SE (Squeeze-and-Excitation) channel attention in its blocks, so we only add the **spatial** component:

```
Input: feature map [B, 1280, 7, 7]
→ Channel-wise AvgPool + MaxPool → [B, 2, 7, 7]
→ Conv2d(2, 1, kernel=7, padding=3) → [B, 1, 7, 7]
→ Sigmoid → spatial attention mask
→ Element-wise multiply with input feature map
Output: attended feature map [B, 1280, 7, 7]
```

Parameters added: ~99 (negligible). Purpose: force the model to focus on leaf regions rather than background.

### 2. Dual Pooling (Avg + Max)

Replace the backbone's single `AdaptiveAvgPool2d(1)` with both average and max pooling, concatenated:

```
Attended features [B, 1280, 7, 7]
→ AdaptiveAvgPool2d(1) → [B, 1280]  (overall texture)
→ AdaptiveMaxPool2d(1) → [B, 1280]  (strongest activations: edges, veins)
→ Concatenate → [B, 2560]
```

This gives the classifier two complementary views of the feature map.

### 3. Richer Classifier Head

Replace the single linear layer with a bottleneck:

```
Linear(2560, 512) → BatchNorm1d(512) → ReLU → Dropout(0.4)
Linear(512, 128)  → BatchNorm1d(128) → ReLU → Dropout(0.3)
Linear(128, 4)
```

The bottleneck forces learning a compressed discriminative representation. BatchNorm stabilizes training with the small dataset. Two dropout stages provide regularization without being too aggressive.

Note: DataLoader must use `drop_last=True` to avoid batch_size=1 edge case that crashes BatchNorm1d.

Additional parameters: ~1.4M. Estimated size increase: ~1.5MB.

### 4. Test-Time Augmentation (TTA) in InferenceEngine.swift

Run two inference passes per frame:
1. Original image (normal orientation)
2. Horizontally flipped image (using `CGImagePropertyOrientation.upMirrored` on the `VNImageRequestHandler` — zero-copy, no pixel buffer manipulation)

Averaging logic: collect all `VNClassificationObservation` objects from both passes, group by `identifier`, average their `confidence` values, then apply the existing safe_plants comparison logic.

Both passes run within a single `isProcessing` guard window — the second request fires immediately after the first completes. Frame drops may increase slightly but the capture engine's duty-cycled approach (one frame per stride) means this is acceptable.

Expected latency increase: ~15ms (total ~30ms, well within 50ms budget).

### 5. Unchanged Components

- EfficientNet-B0 backbone (pretrained ImageNet weights)
- Two-phase training: Phase 1 (10 epochs, classifier only) → Phase 2 (30 epochs, full fine-tune with discriminative LR)
- Mixup augmentation (alpha=0.2)
- Label smoothing (0.1)
- Class-weighted CrossEntropyLoss
- Cosine annealing LR scheduler
- Early stopping (patience=8)
- All training/test data augmentation transforms
- Alert engine thresholds, cooldowns, sensitivity settings
- CaptureEngine apogee detection and duty cycling

## Custom Model Wrapper

The architectural changes require wrapping EfficientNet-B0 in a custom `nn.Module` (`PlantDetectorNet`) rather than modifying the model in-place. This wrapper:

1. Uses `model.features` as the frozen/unfrozen backbone
2. Applies spatial attention after features
3. Applies dual pooling
4. Passes through the new classifier head

The two-phase training adjusts `requires_grad` on `self.backbone` parameters, same as before.

### Training Details

**Phase 1 (classifier only):** Increase weight decay to `1e-3` (from `1e-4`) for the larger head to prevent overfitting.

**Phase 2 (full fine-tune):** Discriminative LR parameter groups:
- Early backbone (features[:4]): LR/50
- Mid backbone (features[4:6]): LR/10
- Late backbone (features[6:]): LR/3
- Spatial attention: LR/3 (same as late backbone — operates on final feature map)
- Classifier head: LR

Early stopping patience remains at 8 epochs — the richer head converges within the same window due to the frozen-then-unfrozen strategy.

## Core ML Export

The custom wrapper traces cleanly with `torch.jit.trace` since all operations (Conv2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, Linear, BatchNorm1d, ReLU, Dropout, sigmoid, cat, multiply) are standard ops supported by `coremltools`.

Note: use `values, _ = torch.max(x, dim=1, keepdim=True)` to avoid trace warnings from unused indices tensor.

Update model metadata: version to "4.0.0", description to reference EfficientNet-B0 with spatial attention (v4). Fix stale MobileNetV2 references in script docstring.

The existing approximate per-channel scale (`1.0 / (255.0 * 0.226)`) is a known `coremltools` limitation — `ImageType` does not support per-channel scale, only per-channel bias. This is unchanged from v3 and is acceptable.

## Estimated Size

| Component | Parameters | Size |
|-----------|-----------|------|
| EfficientNet-B0 backbone | 4.0M | ~7.5MB |
| Spatial attention (Conv2d) | 99 | <1KB |
| Classifier head | ~1.4M | ~1.5MB |
| **Total** | **~5.4M** | **~9.0MB** |

## Expected Impact

- Spatial attention: reduces false positives from background clutter
- Dual pooling: captures both texture (avg) and discriminative features (max)
- Richer head: better separation of visually similar classes (ivy vs oak)
- TTA: smooths orientation-dependent errors at inference time
- Combined expected improvement: 5-15% accuracy lift, primarily on poison ivy/oak

## Files Modified

1. `scripts/train_model.py` — New `SpatialAttention`, `PlantDetectorNet` classes; updated `create_model()`, training loop parameter group logic
2. `LeafAlert/Engines/InferenceEngine.swift` — Add TTA (flip + average predictions)
