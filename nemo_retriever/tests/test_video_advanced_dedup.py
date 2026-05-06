# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_retriever.video.advanced_dedup."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("imagehash")
pytest.importorskip("cv2")

from PIL import Image

from nemo_retriever.video.advanced_dedup import (
    advanced_dedup_indices,
    compute_entropy,
    is_blurry,
)


def _b64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _solid(value: int, size: int = 64) -> str:
    return _b64_png(np.full((size, size, 3), value, dtype=np.uint8))


def _noise(seed: int, size: int = 64) -> str:
    rng = np.random.default_rng(seed)
    return _b64_png(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8))


def _blurry(value: int, size: int = 64) -> str:
    """A solid-color frame: zero Laplacian variance, definitely blurry."""
    return _solid(value, size)


def test_is_blurry_flags_solid_color_frames() -> None:
    assert is_blurry(_blurry(120), threshold=100.0) is True


def test_is_blurry_passes_high_variance_frames() -> None:
    assert is_blurry(_noise(42), threshold=100.0) is False


def test_compute_entropy_higher_for_noise_than_solid() -> None:
    e_solid = compute_entropy(_solid(120))
    e_noise = compute_entropy(_noise(7))
    assert e_solid is not None and e_noise is not None
    assert e_noise > e_solid


def test_advanced_dedup_drops_blurry_keeps_unique_clusters() -> None:
    """Three groups: blurry (drop), two near-identical noise (cluster), one unique noise."""
    frames = [
        _blurry(0),       # idx 0 — drop (blurry)
        _noise(1),        # idx 1 — keep (first of cluster A)
        _noise(1),        # idx 2 — drop (same noise — clusters with #1, no entropy gain)
        _noise(99),       # idx 3 — keep (different noise — different cluster)
    ]
    kept = advanced_dedup_indices(
        frames,
        blur_threshold=100.0,
        similarity_threshold=8,
        entropy_gain_threshold=0.1,
    )
    assert 0 not in kept
    assert 1 in kept
    assert 3 in kept
    assert 2 not in kept


def test_advanced_dedup_empty_input() -> None:
    assert advanced_dedup_indices([], blur_threshold=100.0) == []


def test_advanced_dedup_undecodable_b64_dropped_silently() -> None:
    kept = advanced_dedup_indices(
        [_noise(1), "@@@bad@@@", _noise(2)],
        blur_threshold=100.0,
        similarity_threshold=5,
        entropy_gain_threshold=0.1,
    )
    assert 0 in kept and 2 in kept and 1 not in kept


def test_advanced_dedup_keeps_high_entropy_gain_within_cluster() -> None:
    """Two near-identical noise frames with significantly different entropy
    should both survive when their pHashes cluster together."""
    # Two frames with the same RNG seed produce different pixel layouts (different draws)
    # but pHash ~ similar (both are noise).  Use sizes that yield distinct entropies.
    rng_low = np.random.default_rng(0)
    rng_high = np.random.default_rng(0)
    # Frame A: histogram concentrated in the lower half (lower entropy).
    frame_a = _b64_png(rng_low.integers(0, 64, size=(64, 64, 3), dtype=np.uint8))
    # Frame B: full-range noise (high entropy).
    frame_b = _b64_png(rng_high.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))

    # Verify the two frames are visually similar enough for pHash to cluster them
    # but have entropy gain > 0.5 (well above the default 0.1 threshold).
    e_a = compute_entropy(frame_a)
    e_b = compute_entropy(frame_b)
    assert e_a is not None and e_b is not None
    assert (e_b - e_a) > 0.5

    kept = advanced_dedup_indices(
        [frame_a, frame_b],
        blur_threshold=100.0,
        similarity_threshold=64,  # very loose — force them into one cluster
        entropy_gain_threshold=0.1,
    )
    # Both must be kept: A (first of cluster) and B (entropy gain > threshold).
    assert kept == [0, 1]
