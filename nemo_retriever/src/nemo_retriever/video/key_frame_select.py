# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSIM-based key-frame selection within a scene.

Ports ``KeyFrameDetector.py`` from the reference implementation.  The
first frame of each scene is always kept (establishing shot); subsequent
frames are kept when the normalized SSIM to their predecessor falls
below ``mean - std`` of the SSIM distribution (after rejecting
``z_threshold``-sigma outliers).
"""

from __future__ import annotations

import base64
import io
import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def is_skimage_available() -> bool:
    try:
        import skimage.metrics  # noqa: F401
    except ImportError:
        return False
    return True


def _decode_grayscale(b64: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("L")
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


def _decode_grayscale_batch(frames_b64: Sequence[str]) -> List[Optional[np.ndarray]]:
    """Decode each base64-PNG to a grayscale numpy array; ``None`` on failure."""
    return [_decode_grayscale(b) for b in frames_b64]


def _ssim_pairs_from_decoded(decoded: Sequence[Optional[np.ndarray]]) -> List[float]:
    """Internal helper: SSIM scores from already-decoded grays.

    Returns ``len(decoded) - 1`` scores.  A pair whose either frame is
    ``None`` (decode failed) contributes ``1.0`` so the run isn't treated
    as a key-frame boundary on a transient decode error; the caller
    *also* checks the per-frame decode validity to keep the frame
    immediately after an undecodable one.
    """
    if not is_skimage_available():
        raise RuntimeError(
            "key_frame_select requires scikit-image. "
            "Install with: pip install 'nemo-retriever[multimedia]'."
        )

    from skimage.metrics import structural_similarity

    if len(decoded) < 2:
        return []

    scores: List[float] = []
    for prev, curr in zip(decoded, decoded[1:]):
        if prev is None or curr is None or prev.shape != curr.shape:
            scores.append(1.0)
            continue
        try:
            scores.append(float(structural_similarity(prev, curr)))
        except Exception:
            logger.debug("SSIM computation failed; treating pair as identical", exc_info=True)
            scores.append(1.0)
    return scores


def ssim_pairs(frames_b64: Sequence[str]) -> List[float]:
    """Compute SSIM between each consecutive pair.  Returns ``len(frames) - 1`` floats.

    A pair whose either frame fails to decode contributes ``1.0`` (= "no
    visual change") — the *score-level* defence against decode hiccups.
    The frame-level defence (keeping the frame after an undecodable one)
    lives in :func:`select_key_frame_indices`, which has access to
    per-frame decode validity.
    """
    return _ssim_pairs_from_decoded(_decode_grayscale_batch(frames_b64))


def select_key_frame_indices(
    frames_b64: Sequence[str],
    z_threshold: float = 2.0,
) -> List[int]:
    """Return indices to keep from a scene's frame list.

    Decoding happens once per frame; the resulting grays are reused for
    both SSIM scoring and the per-frame defensive keep-rule.

    Algorithm:
    * Always keep frame 0 (establishing shot).
    * For ``len(frames) >= 2``, compute SSIM between consecutive pairs,
      normalize to ``[0, 1]``, reject ``z_threshold``-sigma outliers
      from the *normalized score* distribution before computing the
      keep statistic, and keep frame ``i`` when
      ``normalized_ssim[i-1] < mean − std``.
    * Always keep a frame whose own bytes fail to decode *and* a frame
      whose predecessor failed to decode — without two valid grays we
      have no SSIM signal, so we cannot assert "no visual change".
    """
    n = len(frames_b64)
    if n == 0:
        return []
    if n == 1:
        return [0]

    decoded = _decode_grayscale_batch(frames_b64)
    scores = np.asarray(_ssim_pairs_from_decoded(decoded), dtype=np.float64)

    if scores.size == 0 or not np.isfinite(scores).any():
        return list(range(n))

    out: List[int] = [0]

    s_min, s_max = float(np.min(scores)), float(np.max(scores))
    if s_max - s_min < 1e-9:
        # Uniform-score path: no signal from SSIM.  Defensively keep
        # anything whose decode validity changed (covers C1 — a missing
        # frame masking a real scene change in its neighbour).
        for i in range(1, n):
            if decoded[i] is None or decoded[i - 1] is None:
                out.append(i)
        return out

    normalized = (scores - s_min) / (s_max - s_min)
    mean = float(np.mean(normalized))
    std = float(np.std(normalized))
    if std > 0.0:
        z = (normalized - mean) / std
        kept_mask = np.abs(z) <= float(z_threshold)
        kept_dist = normalized[kept_mask] if kept_mask.any() else normalized
    else:
        kept_dist = normalized
    threshold = float(np.mean(kept_dist) - np.std(kept_dist))

    for i in range(1, n):
        if decoded[i] is None or decoded[i - 1] is None:
            out.append(i)
            continue
        if normalized[i - 1] < threshold:
            out.append(i)
    return out
