import io

import numpy as np
from PIL import Image

from ivai.data.duplicates import find_duplicates, phash_bits, pixel_correlation
from ivai.data.index import pixel_sha256


def texture(seed: int, size: int = 200) -> np.ndarray:
    """Random smooth texture (low-pass filtered noise) - resembles a metal surface."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(size // 8, size // 8))
    img = Image.fromarray(((noise - noise.min()) / np.ptp(noise) * 255).astype(np.uint8))
    return np.asarray(img.resize((size, size), Image.Resampling.BICUBIC))


def jpeg_roundtrip(arr: np.ndarray, quality: int = 70) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L"))


def run(images):
    keys = [pixel_sha256(i) for i in images]
    return find_duplicates(images, keys, phash_max_distance=6, min_pixel_correlation=0.95)


def test_phash_has_64_bits():
    assert phash_bits(texture(0)).shape == (64,)


def test_exact_duplicate_detected():
    a = texture(0)
    groups, pairs, _ = run([a, a.copy(), texture(1)])
    assert groups[0] == groups[1] != groups[2]
    assert pairs[0]["kind"] == "exact"


def test_recompressed_copy_detected():
    a = texture(0)
    groups, pairs, _ = run([a, jpeg_roundtrip(a), texture(1)])
    assert groups[0] == groups[1]
    assert any(p["kind"] == "near" for p in pairs)


def test_flipped_copy_detected():
    a = texture(3)
    groups, _, _ = run([a, np.ascontiguousarray(np.fliplr(a)), texture(4)])
    assert groups[0] == groups[1]


def test_distinct_images_not_grouped():
    images = [texture(s) for s in range(10)]
    groups, pairs, nearest = run(images)
    assert len(set(groups)) == 10
    assert pairs == []
    assert (nearest > 6).all()


def test_pixel_correlation_bounds():
    a = texture(0)
    assert pixel_correlation(a, a) == 1.0
    assert abs(pixel_correlation(a, texture(9))) < 0.5
