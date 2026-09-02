"""dataset_qa: oversized originals are downsized on commit, small ones copied byte-for-byte."""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dataset_qa as qa  # noqa: E402


def _jpeg(path: Path, w: int, h: int, seed: int):
    rng = np.random.default_rng(seed)
    # Noisy colour image: passes the saturation/greyscale filter and hashes stably.
    a = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    a[..., 1] //= 3
    Image.fromarray(a).save(path, "JPEG", quality=90)


def test_oversized_originals_are_flagged_and_downsized_on_commit(tmp_path):
    staged = tmp_path / "staged" / "poison_ivy"; staged.mkdir(parents=True)
    _jpeg(staged / "big.jpg", 5200, 4300, 1)     # 22.4 MP
    _jpeg(staged / "small.jpg", 640, 480, 2)
    records = qa.scan_staged({"poison_ivy": staged}, min_side=128, min_aspect=0.4, max_aspect=2.5,
                             sat_thresh=0.04, max_pixels=int(20e6))
    by = {r["name"]: r for r in records}
    assert by["big.jpg"]["drop"] is None and by["big.jpg"]["oversized"] is True
    assert by["small.jpg"]["drop"] is None and by["small.jpg"]["oversized"] is False

    pool = tmp_path / "pool"
    entries = qa.commit_survivors([r for r in records if r["drop"] is None], pool, "t", {}, dry_run=False,
                                  downsize_long_edge=2048)
    assert len(entries) == 2
    for e in entries:
        dest = pool / e["dest"]
        with Image.open(dest) as im:
            w, h = im.size
        if e["source_name"] == "big.jpg":
            assert max(w, h) == 2048 and e["downsized_to_long_edge"] == 2048
            assert abs((w / h) - (5200 / 4300)) < 0.01, "aspect must be preserved"
        else:
            assert (w, h) == (640, 480) and e["downsized_to_long_edge"] is None
            assert dest.read_bytes() == (staged / "small.jpg").read_bytes(), "small files are copied verbatim"


def test_no_cap_means_no_flag(tmp_path):
    staged = tmp_path / "s" / "safe_plants"; staged.mkdir(parents=True)
    _jpeg(staged / "x.jpg", 3000, 2000, 3)
    (r,) = qa.scan_staged({"safe_plants": staged}, 128, 0.4, 2.5, 0.04)
    assert r["oversized"] is False
