#!/usr/bin/env python3
"""
Shared Core ML export helper for LeafAlert PlantDetector.

PARITY FIX (P0): coremltools' ImageType cannot express a per-channel scale —
its `scale` argument is a single scalar. The old export approximated ImageNet
normalization with a uniform std of 0.226 (scale = 1/(255*0.226)), which is
wrong per channel and silently degrades on-device accuracy.

The fix is to BAKE exact per-channel ImageNet normalization into the model
graph via `NormalizeWrapper`. The wrapper consumes raw 0-255 RGB pixels (what
CoreML's ImageType passes through with scale=1.0, bias=[0,0,0]) and applies
`(x / 255 - mean) / std` per channel before calling the backbone. The CoreML
ImageType therefore does NO normalization — it just hands raw pixels in.
"""

from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct

# ImageNet normalization constants (must match the training transforms)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NormalizeWrapper(nn.Module):
    """Wraps a backbone and bakes exact per-channel ImageNet normalization
    into the traced graph.

    Consumes raw 0-255 RGB pixels (what CoreML's ImageType passes through when
    configured with scale=1.0, bias=[0,0,0]) and computes
    `(x / 255 - mean) / std` per channel before delegating to the backbone.

    mean/std are registered as buffers shaped [1, 3, 1, 1] so they trace
    correctly and broadcast over the NCHW input.
    """

    def __init__(self, backbone: nn.Module,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.backbone = backbone
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives as raw 0-255 RGB pixels from the CoreML ImageType.
        x = (x / 255.0 - self.mean) / self.std
        return self.backbone(x)


class _ClassifierExportModel(nn.Module):
    """Export-time wrapper: NormalizeWrapper + an optional softmax.

    The backbone emits raw logits. Baking a softmax into the exported graph
    makes the Core ML classifier output true probabilities in [0, 1] that sum
    to 1, so Vision's `VNClassificationObservation.confidence` is a real
    probability. Without this, on-device confidence clamping, TTA averaging,
    and the user alert threshold all operate on unbounded logits and are
    meaningless. Kept separate from NormalizeWrapper so the normalization-math
    parity test (which uses an Identity backbone) stays unaffected.
    """

    def __init__(self, backbone: nn.Module, apply_softmax: bool = True,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.norm = NormalizeWrapper(backbone, mean, std)
        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        if self.apply_softmax:
            out = torch.softmax(out, dim=1)
        return out


def export_coreml(model: nn.Module, labels: list, out_path: Path,
                  image_size: int,
                  author: str = "LeafAlert",
                  short_description: str = "",
                  version: str = "4.1.0",
                  apply_softmax: bool = True):
    """Trace `model` (wrapped with per-channel normalization and a softmax)
    and convert to a Core ML classifier .mlpackage.

    The CoreML ImageType is configured with scale=1.0 / bias=[0,0,0] so it
    passes raw 0-255 RGB pixels straight into NormalizeWrapper, which performs
    the real per-channel normalization in-graph. A softmax is baked on top
    (apply_softmax=True) so the classifier emits true probabilities.
    """
    wrapped = _ClassifierExportModel(model, apply_softmax=apply_softmax)
    wrapped.eval()
    wrapped.cpu()

    example_input = torch.randn(1, 3, image_size, image_size)
    traced_model = torch.jit.trace(wrapped, example_input)

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, image_size, image_size),
                scale=1.0,
                bias=[0.0, 0.0, 0.0],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        classifier_config=ct.ClassifierConfig(labels),
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.author = author
    if short_description:
        mlmodel.short_description = short_description
    mlmodel.license = "For use with LeafAlert app"
    mlmodel.version = version

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    print(f"\nCore ML model saved to: {out_path}")

    total_size = sum(
        f.stat().st_size for f in out_path.rglob("*") if f.is_file()
    )
    print(f"Model size: {total_size / 1024 / 1024:.1f} MB")

    return mlmodel
