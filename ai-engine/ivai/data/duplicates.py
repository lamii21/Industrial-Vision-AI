"""Exact and near-duplicate detection.

Why it matters: if the same (or almost the same) image ends up both in train and
in test, the test score measures memorisation instead of generalisation. This is
a classic source of data leakage.

Method
------
1. **Exact duplicates**: identical decoded pixels (SHA-256 of the pixel array).
2. **Near-duplicates** (re-encoded, slightly cropped/brightened, flipped or rotated
   copies), in two steps:

   a. *Candidates*: perceptual hash (pHash, 64 bits from the low-frequency DCT
      coefficients). The minimum Hamming distance over the 8 flips/rotations of
      the second image is used, so mirrored copies are also found.
   b. *Confirmation*: Pearson correlation of the pixels (after applying the best
      flip/rotation) must be high. Step b removes false positives on regular
      textures that can share a similar low-frequency signature.

3. Confirmed pairs are merged into groups (union-find). The split then keeps each
   group inside a single subset.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image
from scipy.fft import dctn

# The 8 elements of the dihedral group D4 (identity, rotations, flips).
TRANSFORMS = (
    lambda a: a,
    lambda a: np.rot90(a, 1),
    lambda a: np.rot90(a, 2),
    lambda a: np.rot90(a, 3),
    lambda a: np.fliplr(a),
    lambda a: np.flipud(a),
    lambda a: np.rot90(np.fliplr(a), 1),
    lambda a: np.rot90(np.flipud(a), 1),
)


def phash_bits(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """Perceptual hash as a boolean vector of ``hash_size**2`` bits."""
    size = hash_size * highfreq_factor
    small = np.asarray(
        Image.fromarray(gray).resize((size, size), Image.Resampling.LANCZOS), dtype=np.float64
    )
    low = dctn(small, norm="ortho")[:hash_size, :hash_size]
    return (low > np.median(low)).ravel()


def _to_uint64(bits: np.ndarray) -> np.ndarray:
    """(N, 64) booleans -> (N,) uint64, one integer per hash."""
    return np.packbits(bits, axis=1).view(">u8").astype(np.uint64).ravel()


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two sets of 64-bit hashes (N,64) x (M,64).

    XOR of the packed integers followed by a population count (number of 1 bits).
    """
    xa, xb = _to_uint64(a), _to_uint64(b)
    return np.bitwise_count(xa[:, None] ^ xb[None, :]).astype(np.int32)


def pixel_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two same-shaped images (1.0 = identical up to gain/offset)."""
    if a.shape != b.shape:
        b = np.asarray(Image.fromarray(b).resize(a.shape[::-1], Image.Resampling.BILINEAR))
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    x, y = x - x.mean(), y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else float(np.array_equal(a, b))


def compute_hashes(images: Sequence[np.ndarray]) -> np.ndarray:
    """Return an array (N, 8, 64): pHash of each image under the 8 transforms."""
    return np.stack([np.stack([phash_bits(np.ascontiguousarray(t(img))) for t in TRANSFORMS]) for img in images])


def min_hamming_all_pairs(hashes: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """For every pair (i, j), the minimum distance between hash(i) and any transform of j.

    Returns (distance matrix N x N, index of the best transform N x N).
    """
    n = hashes.shape[0]
    base = hashes[:, 0, :]
    dist = np.full((n, n), 10**6, dtype=np.int32)
    best_t = np.zeros((n, n), dtype=np.int8)
    for t in range(hashes.shape[1]):
        other = hashes[:, t, :]
        for start in range(0, n, chunk):
            d = hamming_matrix(base[start : start + chunk], other)
            block = dist[start : start + chunk]
            better = d < block
            block[better] = d[better]
            best_t[start : start + chunk][better] = t
    np.fill_diagonal(dist, 0)
    return dist, best_t


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def find_duplicates(
    images: Sequence[np.ndarray],
    exact_keys: Sequence[str],
    phash_max_distance: int,
    min_pixel_correlation: float,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Detect duplicates and return ``(group_ids, confirmed_pairs, nearest_distance)``.

    * ``group_ids[i]``: duplicate-group id of image i (singletons get their own id).
    * ``confirmed_pairs``: list of dicts describing every confirmed pair.
    * ``nearest_distance[i]``: Hamming distance to the closest *other* image
      (useful in the EDA to justify the threshold).
    """
    n = len(images)
    uf = UnionFind(n)
    pairs: list[dict] = []

    # 1. Exact duplicates
    first_seen: dict[str, int] = {}
    for i, key in enumerate(exact_keys):
        if key in first_seen:
            j = first_seen[key]
            uf.union(i, j)
            pairs.append({"i": j, "j": i, "kind": "exact", "hamming": 0, "correlation": 1.0, "transform": 0})
        else:
            first_seen[key] = i

    # 2. Near duplicates
    hashes = compute_hashes(images)
    dist, best_t = min_hamming_all_pairs(hashes)
    sym = np.minimum(dist, dist.T)
    nearest = np.where(np.eye(n, dtype=bool), 10**6, sym).min(axis=1)

    cand_i, cand_j = np.nonzero(np.triu(sym <= phash_max_distance, k=1))
    for i, j in zip(cand_i.tolist(), cand_j.tolist()):
        if exact_keys[i] == exact_keys[j]:
            continue  # already recorded as exact
        t = int(best_t[i, j]) if dist[i, j] <= dist[j, i] else int(best_t[j, i])
        a, b = (images[i], TRANSFORMS[t](images[j])) if dist[i, j] <= dist[j, i] else (images[j], TRANSFORMS[t](images[i]))
        corr = pixel_correlation(a, np.ascontiguousarray(b))
        if corr >= min_pixel_correlation:
            uf.union(i, j)
            pairs.append({"i": i, "j": j, "kind": "near", "hamming": int(sym[i, j]), "correlation": corr, "transform": t})

    roots = [uf.find(i) for i in range(n)]
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids, pairs, nearest
