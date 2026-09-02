"""
Preprocessing-parity tests for the Core ML normalization fix.

These verify that NormalizeWrapper applies EXACT per-channel ImageNet
normalization — `(x/255 - mean)/std` per channel — and that the old buggy
uniform-std (0.226) approximation it replaces produces a measurably different
result. No model checkpoint is loaded; we only check the normalization math.

Run with:
    .venv/bin/python -m pytest tests/test_preprocessing_parity.py
"""

import pytest
pytest.importorskip("torch")  # heavy ML dep: skipped in the light CI job
pytest.importorskip("coremltools")  # heavy ML dep: skipped in the light CI job

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the scripts/ directory importable so we can pull in the real wrapper.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from coreml_export import IMAGENET_MEAN, IMAGENET_STD, NormalizeWrapper  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


MEAN = np.array(IMAGENET_MEAN, dtype=np.float64)
STD = np.array(IMAGENET_STD, dtype=np.float64)

# A few representative raw 0-255 RGB pixel triples to test against.
PIXEL_CASES = [
    [0.0, 0.0, 0.0],        # black
    [255.0, 255.0, 255.0],  # white
    [128.0, 64.0, 200.0],   # arbitrary mid-tones
    [10.0, 250.0, 33.0],    # channel extremes
]


def reference_normalize(pixel_rgb):
    """Exact per-channel reference: (x/255 - mean) / std."""
    x = np.asarray(pixel_rgb, dtype=np.float64)
    return (x / 255.0 - MEAN) / STD


def buggy_uniform_normalize(pixel_rgb):
    """The bug being replaced: a single uniform std of 0.226.

    The old CoreML ImageType used scale = 1/(255*0.226) with bias = -mean/std,
    i.e. (x * 1/(255*0.226)) + (-mean/std). That divides every channel by the
    same 0.226 instead of its true per-channel std."""
    x = np.asarray(pixel_rgb, dtype=np.float64)
    scale = 1.0 / (255.0 * 0.226)
    bias = -MEAN / STD
    return x * scale + bias


@pytest.mark.parametrize("pixel", PIXEL_CASES)
def test_normalize_wrapper_matches_reference(pixel):
    """NormalizeWrapper must reproduce exact per-channel normalization."""
    # NormalizeWrapper wraps a backbone; use Identity so its output is exactly
    # the normalized tensor (no model weights involved).
    wrapper = NormalizeWrapper(nn.Identity())
    wrapper.eval()

    # Shape [1, 3, 1, 1]: one pixel, 3 channels.
    x = torch.tensor(pixel, dtype=torch.float32).view(1, 3, 1, 1)
    with torch.no_grad():
        out = wrapper(x).view(3).numpy().astype(np.float64)

    expected = reference_normalize(pixel)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("pixel", PIXEL_CASES)
def test_reference_math_is_self_consistent(pixel):
    """Pure-numpy sanity: the reference is (x/255 - mean)/std per channel."""
    x = np.asarray(pixel, dtype=np.float64)
    manual = np.array([
        (x[c] / 255.0 - MEAN[c]) / STD[c] for c in range(3)
    ])
    np.testing.assert_allclose(reference_normalize(pixel), manual, rtol=0, atol=0)


def test_buggy_uniform_scale_differs_from_correct():
    """The replaced uniform-0.226 approximation must differ from the exact
    per-channel normalization on at least one channel for typical pixels.

    If these were equal, the bug would have been harmless — assert it is not."""
    any_meaningful_diff = False
    for pixel in PIXEL_CASES:
        correct = reference_normalize(pixel)
        buggy = buggy_uniform_normalize(pixel)
        if np.max(np.abs(correct - buggy)) > 1e-3:
            any_meaningful_diff = True
    assert any_meaningful_diff, (
        "Uniform-0.226 approximation unexpectedly matched exact per-channel "
        "normalization — the parity fix would be a no-op."
    )


def test_buggy_differs_per_channel_for_white_pixel():
    """For a white pixel (255), the red channel std (0.229) vs the uniform
    0.226 must yield a visibly different normalized value."""
    correct = reference_normalize([255.0, 255.0, 255.0])
    buggy = buggy_uniform_normalize([255.0, 255.0, 255.0])
    # Red channel: true std 0.229 != 0.226, so the values must differ.
    assert abs(correct[0] - buggy[0]) > 1e-3
