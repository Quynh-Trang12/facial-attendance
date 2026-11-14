"""Content fingerprints for duplicate discovery, never for liveness labeling."""

from hashlib import sha256

import cv2
import numpy as np
from PIL import Image


def pixel_sha256(image: Image.Image) -> str:
    image = image.convert("RGB")
    digest = sha256(f"RGB:{image.width}:{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def perceptual_hash(image: Image.Image) -> str:
    """64 low-frequency DCT bits; collisions require visual adjudication."""
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    frequencies = cv2.dct(np.asarray(gray, dtype=np.float32))[:8, :8].ravel()
    bits = frequencies > np.median(frequencies[1:])
    return np.packbits(bits).tobytes().hex()


def near_candidates(signatures, *, radius=6, neighbors=16):
    """Yield approximate LSH neighbors among distinct hashes.

    Identical hashes are grouped separately, so they cannot fill every neighbor
    slot. Saturated queries are reported, not interpreted as an exhaustive scan.
    This index finds candidates; absence of a candidate is not independence proof.
    """
    signatures = sorted(set(signatures))
    if len(signatures) < 2:
        return [], 0
    values = np.array([list(bytes.fromhex(s)) for s in signatures], dtype=np.uint8)
    cv2.setRNGSeed(42)
    index = cv2.flann_Index(values, dict(algorithm=6, table_number=12,
                                       key_size=16, multi_probe_level=2))
    k = min(neighbors + 1, len(values))
    pairs, saturated = set(), 0
    # Bound query output memory independently of dataset size.
    for start in range(0, len(values), 4096):
        matches, distances = index.knnSearch(values[start:start + 4096], k, params={})
        for offset, (indices, dists) in enumerate(zip(matches, distances)):
            left = start + offset
            if k < len(values) and indices[-1] >= 0 and dists[-1] <= radius:
                saturated += 1
            for right, distance in zip(indices, dists):
                if right >= 0 and right != left and distance <= radius:
                    pairs.add(tuple(sorted((left, int(right)))))
    return [(signatures[a], signatures[b],
             (int(signatures[a], 16) ^ int(signatures[b], 16)).bit_count())
            for a, b in sorted(pairs)], saturated


def complete_hamming_pairs(signatures, *, radius=6):
    """Yield every distinct 64-bit signature pair within radius, exactly once.

    Four 16-bit blocks suffice for radii up to seven: a matching pair must
    differ by at most one bit in at least one block. Probe all such blocks,
    verify the full distance, and emit only the first matching block. This
    proves coverage in hash space, not recall of visually similar images.
    Identical signatures must be grouped by the caller.
    """
    if type(radius) is not int or not 0 <= radius <= 7:
        raise ValueError('Complete search supports integer radii from 0 to 7')
    signatures = sorted(set(signatures))
    if any(len(s) != 16 or any(c not in '0123456789abcdef' for c in s) for s in signatures):
        raise ValueError('Expected lowercase 64-bit hexadecimal signatures')
    values = np.array([int(s, 16) for s in signatures], dtype=np.uint64)
    masks = np.array([0] + [1 << bit for bit in range(16)], dtype=np.uint16)
    for block in range(4):
        keys = ((values >> (16 * block)) & 65535).astype(np.uint16)
        order = np.argsort(keys, kind='stable')
        ordered = keys[order]
        start = 0
        while start < len(values):
            size = min(1024, len(values) - start)
            # Dense buckets can be large even with a bounded query count.
            # Bound intermediate candidate arrays, without dropping candidates.
            while True:
                probes = (keys[start:start + size, None] ^ masks).ravel()
                lo = np.searchsorted(ordered, probes, side='left')
                hi = np.searchsorted(ordered, probes, side='right')
                counts = hi - lo
                total = int(counts.sum())
                if total <= 1000000 or size == 1:
                    break
                size = max(1, size // 2)
            query = np.repeat(np.arange(size * len(masks)), counts)
            offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            right = order[lo[query] + offsets]
            left = start + query // len(masks)
            unique = right > left
            left, right = left[unique], right[unique]
            differences = values[left] ^ values[right]
            distances = np.bitwise_count(differences)
            keep = distances <= radius
            for previous in range(block):
                keep &= np.bitwise_count((differences >> (16 * previous)) & 65535) > 1
            for a, b, distance in zip(left[keep], right[keep], distances[keep]):
                yield signatures[a], signatures[b], int(distance)
            start += size
