# Model Architecture v4 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CBAM spatial attention, dual pooling, richer classifier head, and test-time augmentation to improve poison ivy/oak classification accuracy and reduce false positives.

**Architecture:** Wrap EfficientNet-B0 backbone in a custom `PlantDetectorNet` that applies spatial attention on the final feature map, dual avg+max pooling, and a 3-layer bottleneck classifier. Swift-side TTA flips the image and averages predictions from both orientations.

**Tech Stack:** PyTorch, torchvision, coremltools, Swift/Vision framework

---

## Chunk 1: Training Script — New Architecture

### Task 1: Add SpatialAttention module

**Files:**
- Modify: `scripts/train_model.py:80-99` (replace `create_model` section)

- [ ] **Step 1: Add SpatialAttention class after imports**

Insert after the `IMAGENET_STD` line (line 52), before the data transforms section:

```python
# ─── Model Architecture ───────────────────────────────────────────

class SpatialAttention(nn.Module):
    """CBAM-style spatial attention. Produces a spatial mask that tells
    the model WHERE on the feature map to focus (leaf vs background)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        mask = self.sigmoid(self.conv(pooled))           # [B, 1, H, W]
        return x * mask
```

- [ ] **Step 2: Add PlantDetectorNet class**

Insert immediately after `SpatialAttention`:

```python
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
```

- [ ] **Step 3: Replace `create_model` function**

Replace the existing `create_model` function (lines 80-99) with:

```python
def create_model(num_classes: int) -> PlantDetectorNet:
    """Create PlantDetectorNet: EfficientNet-B0 + spatial attention + dual pooling + bottleneck head."""
    model = PlantDetectorNet(num_classes)

    # Freeze backbone initially — only train attention + classifier head
    for param in model.backbone.parameters():
        param.requires_grad = False

    return model
```

- [ ] **Step 4: Verify the model instantiates and forward-passes**

Run:
```bash
cd /Users/burkley/Documents/claude/LeafAlert && python3 -c "
import torch
from scripts.train_model import create_model
model = create_model(4)
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f'Output shape: {out.shape}')
assert out.shape == (2, 4), f'Expected (2, 4) got {out.shape}'
print('OK: Model forward pass works')
total_params = sum(p.numel() for p in model.parameters())
print(f'Total params: {total_params:,}')
"
```
Expected: `Output shape: torch.Size([2, 4])`, `OK: Model forward pass works`, total params ~5.4M

- [ ] **Step 5: Commit**

```bash
git add scripts/train_model.py
git commit -m "feat: add SpatialAttention and PlantDetectorNet architecture v4"
```

---

### Task 2: Update training loop for new architecture

**Files:**
- Modify: `scripts/train_model.py:267-415` (main function)

- [ ] **Step 1: Update docstring and version references**

Replace the file docstring (lines 2-17) with:

```python
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
```

- [ ] **Step 2: Update IMAGE_SIZE comment**

Change line 47 from:
```python
IMAGE_SIZE = 224  # MobileNetV2 expects 224x224
```
to:
```python
IMAGE_SIZE = 224  # EfficientNet-B0 expects 224x224
```

- [ ] **Step 3: Update DataLoader to use drop_last=True**

Change the train_loader creation (line 306-308) from:
```python
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
```
to:
```python
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True,
    )
```

- [ ] **Step 4: Update model loading print message**

Change line 318 from:
```python
    print(f"\nLoading MobileNetV2 (pretrained on ImageNet)...")
```
to:
```python
    print(f"\nLoading EfficientNet-B0 + spatial attention (v4)...")
```

- [ ] **Step 5: Update Phase 1 optimizer weight decay**

Change line 327 from:
```python
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
```
to:
```python
    # Train classifier head + spatial attention (backbone frozen)
    phase1_params = list(model.classifier.parameters()) + list(model.spatial_attn.parameters())
    optimizer = optim.Adam(phase1_params, lr=LEARNING_RATE, weight_decay=1e-3)
```

- [ ] **Step 6: Update Phase 2 backbone unfreeze and parameter groups**

Replace lines 348-358 (the backbone unfreeze + param_groups block):

From:
```python
    for param in model.features.parameters():
        param.requires_grad = True

    # Discriminative LR: early layers barely move, late layers adapt more
    # EfficientNet-B0 has 9 feature blocks (0-8)
    param_groups = [
        {"params": list(model.features[:4].parameters()), "lr": LEARNING_RATE / 50},
        {"params": list(model.features[4:6].parameters()), "lr": LEARNING_RATE / 10},
        {"params": list(model.features[6:].parameters()), "lr": LEARNING_RATE / 3},
        {"params": list(model.classifier.parameters()), "lr": LEARNING_RATE},
    ]
```

To:
```python
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
```

- [ ] **Step 7: Update Core ML export metadata**

In `convert_to_coreml`, change the metadata (lines 243-250) from:
```python
    mlmodel.short_description = (
        "Detects poison ivy, poison oak, and poison sumac from camera images. "
        "Fine-tuned MobileNetV2 (v2) trained on iNaturalist research-grade observations. "
        f"Classes: {', '.join(class_names)}"
    )
    mlmodel.license = "For use with LeafAlert app"
    mlmodel.version = "2.0.0"
```
to:
```python
    mlmodel.short_description = (
        "Detects poison ivy, poison oak, and poison sumac from camera images. "
        "EfficientNet-B0 with spatial attention (v4) trained on iNaturalist research-grade observations. "
        f"Classes: {', '.join(class_names)}"
    )
    mlmodel.license = "For use with LeafAlert app"
    mlmodel.version = "4.0.0"
```

- [ ] **Step 8: Update final output messages**

Replace the final print block (lines 405-415) with:
```python
    print(f"\n{'=' * 60}")
    print(f"DONE! Model v4 ready at:")
    print(f"  {OUTPUT_PATH}")
    print(f"\nv4 architecture:")
    print(f"  - EfficientNet-B0 backbone with CBAM spatial attention")
    print(f"  - Dual pooling (avg + max) → 2560-dim feature vector")
    print(f"  - Bottleneck classifier (2560→512→128→{num_classes})")
    print(f"  - {NUM_EPOCHS} total epochs, label smoothing, mixup, class weights")
    print(f"\nRebuild the app to include the new model.")
    print(f"{'=' * 60}")
```

Also change line 269 from:
```python
    print("LeafAlert PlantDetector — Model Training v2")
```
to:
```python
    print("LeafAlert PlantDetector — Model Training v4")
```

- [ ] **Step 9: Verify the full script parses without errors**

Run:
```bash
cd /Users/burkley/Documents/claude/LeafAlert && python3 -c "import scripts.train_model; print('Script parses OK')"
```
Expected: `Script parses OK`

- [ ] **Step 10: Commit**

```bash
git add scripts/train_model.py
git commit -m "feat: update training loop for v4 architecture

Phase 1 trains classifier + spatial attention with weight_decay=1e-3.
Phase 2 adds spatial_attn to discriminative LR groups at LR/3.
Update metadata, drop_last=True, fix stale MobileNetV2 references."
```

---

## Chunk 2: Swift TTA + Training Run

### Task 3: Add test-time augmentation to InferenceEngine.swift

**Files:**
- Modify: `LeafAlert/Engines/InferenceEngine.swift:74-159` (classify method)

- [ ] **Step 1: Replace the classify method with TTA version**

Replace the entire `classify` method (lines 74-159) with the following. The key changes are:
1. Run inference on original image
2. Run inference on horizontally flipped image (via `.upMirrored` orientation)
3. Average the confidence values by identifier
4. Apply the existing toxic-vs-safe comparison on averaged results

```swift
    /// Runs inference on a pixel buffer with test-time augmentation (TTA).
    /// Averages predictions from the original and horizontally flipped image
    /// to smooth orientation-dependent errors.
    /// - Parameters:
    ///   - pixelBuffer: A CVPixelBuffer from the camera capture pipeline.
    ///   - completion: Called on a background queue with the result, or nil on failure.
    func classify(
        pixelBuffer: CVPixelBuffer,
        completion: @escaping (DetectionResult?) -> Void
    ) {
        guard let vnModel else {
            completion(nil)
            return
        }

        inferenceQueue.async { [weak self] in
            guard let self else {
                completion(nil)
                return
            }

            // Drop this request if a previous inference is still in progress.
            guard !self.isProcessing else {
                completion(nil)
                return
            }
            self.isProcessing = true

            let startTime = CFAbsoluteTimeGetCurrent()

            let finishInference = {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                DispatchQueue.main.async {
                    self.lastInferenceTime = elapsed
                }
                self.inferenceQueue.async {
                    self.isProcessing = false
                }
            }

            // --- Pass 1: Original orientation ---
            let originalHandler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer, options: [:]
            )
            let originalRequest = VNCoreMLRequest(model: vnModel)
            originalRequest.imageCropAndScaleOption = .centerCrop

            // --- Pass 2: Horizontally flipped (zero-copy via orientation) ---
            let flippedHandler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: .upMirrored,
                options: [:]
            )
            let flippedRequest = VNCoreMLRequest(model: vnModel)
            flippedRequest.imageCropAndScaleOption = .centerCrop

            do {
                try originalHandler.perform([originalRequest])
                try flippedHandler.perform([flippedRequest])
            } catch {
                print("[InferenceEngine] TTA inference failed: \(error)")
                finishInference()
                completion(nil)
                return
            }

            // Collect observations from both passes
            guard let originalObs = originalRequest.results as? [VNClassificationObservation],
                  let flippedObs = flippedRequest.results as? [VNClassificationObservation],
                  !originalObs.isEmpty
            else {
                finishInference()
                completion(nil)
                return
            }

            // Average confidence values by class identifier
            let averaged = Self.averageObservations(originalObs, flippedObs)

            finishInference()

            // Find the highest-confidence *toxic* plant class.
            guard let topToxic = averaged
                .filter({ Self.toxicLabels.contains($0.key) })
                .max(by: { $0.value < $1.value })
            else {
                completion(nil)
                return
            }

            // Only report if the toxic class beats safe_plants.
            let safeConfidence = averaged["safe_plants"] ?? 0.0
            guard topToxic.value > safeConfidence else {
                completion(nil)
                return
            }

            let clampedConfidence = min(max(topToxic.value, 0.0), 1.0)
            let result = DetectionResult(
                plantType: topToxic.key,
                confidence: clampedConfidence,
                boundingBox: .zero
            )
            completion(result)
        }
    }
```

- [ ] **Step 2: Add the averageObservations helper method**

Add this private static method at the end of the `InferenceEngine` class (before the closing `}`):

```swift
    // MARK: - TTA Helpers

    /// Averages confidence values from two sets of classification observations.
    private static func averageObservations(
        _ a: [VNClassificationObservation],
        _ b: [VNClassificationObservation]
    ) -> [String: Float] {
        var sums: [String: Float] = [:]
        var counts: [String: Int] = [:]

        for obs in a {
            sums[obs.identifier, default: 0] += obs.confidence
            counts[obs.identifier, default: 0] += 1
        }
        for obs in b {
            sums[obs.identifier, default: 0] += obs.confidence
            counts[obs.identifier, default: 0] += 1
        }

        var result: [String: Float] = [:]
        for (key, sum) in sums {
            result[key] = sum / Float(counts[key] ?? 1)
        }
        return result
    }
```

- [ ] **Step 3: Verify the Swift file compiles**

Run:
```bash
cd /Users/burkley/Documents/claude/LeafAlert && xcodebuild -scheme LeafAlert -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 4: Commit**

```bash
git add LeafAlert/Engines/InferenceEngine.swift
git commit -m "feat: add test-time augmentation to InferenceEngine

Run inference on original + horizontally flipped image (via
.upMirrored orientation, zero-copy), average confidence values
by class, then apply toxic-vs-safe comparison on averaged results."
```

---

### Task 4: Train the model and verify improvement

**Files:**
- Uses: `scripts/train_model.py` (no modifications)
- Output: `LeafAlert/Resources/MLModels/PlantDetector.mlpackage`

- [ ] **Step 1: Run the v4 training**

Run:
```bash
cd /Users/burkley/Documents/claude/LeafAlert && python3 scripts/train_model.py 2>&1 | tee training_v4_log.txt
```
Expected: Training completes with ~40 epochs (or early stopping). Final test accuracy should be reported in the output. Watch for per-class accuracy on poison_ivy and poison_oak — these were 55% and 56% in v3.

- [ ] **Step 2: Review training results**

Check the training log for:
- Final test accuracy (target: >70%, ideally >75%)
- Per-class accuracy for poison_ivy and poison_oak (target: >60%)
- No NaN losses or training instabilities
- Model size in the output (target: <10MB)

```bash
tail -20 training_v4_log.txt
```

- [ ] **Step 3: Commit the trained model**

```bash
rm training_v4_log.txt
git add LeafAlert/Resources/MLModels/PlantDetector.mlpackage
git commit -m "feat: retrain model v4 with spatial attention and rich head

Architecture: EfficientNet-B0 + CBAM spatial attention + dual pooling
+ bottleneck classifier (2560→512→128→4). Trained with stronger
weight decay, spatial attention in discriminative LR groups."
```
