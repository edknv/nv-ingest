# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blur + pHash + entropy frame dedup for the video extraction pipeline.

Ports ``Deduplicator.py`` from the reference.  Uses ``imagehash.phash``
(PIL-based, MIT) instead of ``cv2.img_hash.pHash`` to avoid pulling
``opencv-contrib-python`` (this project ships ``opencv-python-headless``).
"""

from __future__ import annotations

import base64
import io
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np


def is_advanced_dedup_available() -> bool:
    try:
        import imagehash  # noqa: F401
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def _decode_pil(b64: str) -> "Optional[object]":
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _to_gray_array(b64: str) -> "Optional[np.ndarray]":
    pil = _decode_pil(b64)
    if pil is None:
        return None
    return np.asarray(pil.convert("L"), dtype=np.uint8)


def is_blurry(frame_b64: str, threshold: float = 100.0) -> bool:
    """Return True when the frame's Laplacian variance is below ``threshold``."""
    if not is_advanced_dedup_available():
        raise RuntimeError(
            "advanced_dedup requires imagehash + cv2. "
            "Install with: pip install 'nemo-retriever[multimedia]'."
        )

    import cv2

    gray = _to_gray_array(frame_b64)
    if gray is None:
        return True  # treat undecodable as blurry / discardable
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < float(threshold)


def compute_entropy(frame_b64: str) -> "Optional[float]":
    """Shannon entropy of the grayscale histogram (256 bins)."""
    gray = _to_gray_array(frame_b64)
    if gray is None:
        return None
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    prob = hist / total
    # Suppress log(0) -> 0 contributions.
    return float(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum())


def _phash_int(b64: str) -> Optional[int]:
    import imagehash

    pil = _decode_pil(b64)
    if pil is None:
        return None
    try:
        h = imagehash.phash(pil)
        return int(str(h), 16)
    except Exception:
        return None


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def advanced_dedup_indices(
    frames_b64: Sequence[str],
    *,
    blur_threshold: float = 100.0,
    similarity_threshold: int = 5,
    entropy_gain_threshold: float = 0.1,
) -> List[int]:
    """Return indices to keep after blur/pHash/entropy filtering.

    Steps:
      1. Drop frames where ``is_blurry``.
      2. Cluster surviving frames by pHash hamming distance ≤
         ``similarity_threshold``.
      3. Within each cluster, sort by original frame index; always keep
         the first; subsequently include a frame only if its entropy
         exceeds the previously-kept frame's entropy by more than
         ``entropy_gain_threshold``.
    """
    if not frames_b64:
        return []

    if not is_advanced_dedup_available():
        raise RuntimeError(
            "advanced_dedup requires imagehash + cv2. "
            "Install with: pip install 'nemo-retriever[multimedia]'."
        )

    # Step 1: blur filter.
    survivors: List[int] = []
    phashes: Dict[int, Optional[int]] = {}
    entropies: Dict[int, float] = {}
    for idx, b64 in enumerate(frames_b64):
        if not isinstance(b64, str):
            continue
        if is_blurry(b64, threshold=blur_threshold):
            continue
        h = _phash_int(b64)
        if h is None:
            continue
        e = compute_entropy(b64)
        if e is None:
            continue
        survivors.append(idx)
        phashes[idx] = h
        entropies[idx] = e

    if not survivors:
        return []

    # Step 2: pHash cluster (greedy, deterministic by survivor order).
    clusters: Dict[int, List[int]] = defaultdict(list)
    cluster_anchor: Dict[int, int] = {}  # cluster_key -> phash anchor
    for idx in survivors:
        h = phashes[idx]
        joined_key: Optional[int] = None
        for key, anchor_h in cluster_anchor.items():
            if _hamming(h, anchor_h) <= int(similarity_threshold):
                joined_key = key
                break
        if joined_key is None:
            cluster_anchor[h] = h
            joined_key = h
        clusters[joined_key].append(idx)

    # Step 3: per-cluster entropy-gain selection.
    kept: List[int] = []
    for cluster_idxs in clusters.values():
        cluster_idxs.sort()
        last_entropy: Optional[float] = None
        for idx in cluster_idxs:
            e = entropies[idx]
            if last_entropy is None:
                kept.append(idx)
                last_entropy = e
            elif (e - last_entropy) > float(entropy_gain_threshold):
                kept.append(idx)
                last_entropy = e
    kept.sort()
    return kept
