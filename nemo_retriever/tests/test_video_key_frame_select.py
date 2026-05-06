# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_retriever.video.key_frame_select."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("skimage.metrics")

from PIL import Image

from nemo_retriever.video.key_frame_select import (
    select_key_frame_indices,
    ssim_pairs,
)


def _b64_png_from_array(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _solid(value: int, size: int = 32) -> str:
    return _b64_png_from_array(np.full((size, size, 3), value, dtype=np.uint8))


def test_first_frame_always_kept_for_two_frame_input() -> None:
    """A two-frame group should at least keep the first frame (establishing shot)."""
    frames_b64 = [_solid(50), _solid(50)]
    kept = select_key_frame_indices(frames_b64, z_threshold=2.0)
    assert kept[0] == 0


def test_select_keeps_visually_changing_frames_drops_static_runs() -> None:
    """A run of identical frames followed by a clear visual change keeps both endpoints."""
    frames_b64 = [_solid(50), _solid(50), _solid(50), _solid(200), _solid(200)]
    kept = select_key_frame_indices(frames_b64, z_threshold=2.0)
    assert 0 in kept
    assert 3 in kept
    # A pure-noise mid-run identical frame need not be kept.
    assert 1 not in kept


def test_single_frame_input_kept_unchanged() -> None:
    kept = select_key_frame_indices([_solid(120)], z_threshold=2.0)
    assert kept == [0]


def test_empty_input_returns_empty() -> None:
    assert select_key_frame_indices([], z_threshold=2.0) == []


def test_ssim_pairs_returns_len_minus_one_scores() -> None:
    frames_b64 = [_solid(0), _solid(0), _solid(255)]
    scores = ssim_pairs(frames_b64)
    assert len(scores) == 2
    # First pair (identical) ~ 1.0; second pair (very different) << first.
    assert scores[0] > scores[1]


def test_select_handles_undecodable_b64_by_keeping_frame() -> None:
    """A frame whose b64 is corrupt should not crash; helper keeps it."""
    frames_b64 = [_solid(50), "not-real-base64!!!", _solid(50)]
    kept = select_key_frame_indices(frames_b64, z_threshold=2.0)
    assert 0 in kept
    assert 1 in kept  # corrupt frame kept defensively


def test_ssim_pairs_treats_undecodable_frame_as_identical() -> None:
    """The 1.0-sentinel contract: pairs adjacent to an undecodable frame contribute 1.0."""
    scores = ssim_pairs([_solid(50), "###bad###", _solid(200)])
    assert len(scores) == 2
    assert scores[0] == 1.0
    assert scores[1] == 1.0


def test_undecodable_frame_does_not_mask_subsequent_scene_change() -> None:
    """Regression for C1: a corrupt frame between two visually different frames
    must not cause the scene-change frame to be silently dropped."""
    frames_b64 = [_solid(50), "###bad###", _solid(220)]
    kept = select_key_frame_indices(frames_b64, z_threshold=2.0)
    # Frame 0 always kept.  Frame 1 (corrupt) kept defensively.  Frame 2
    # (the scene change) MUST also be kept — its predecessor failed to
    # decode, so we have no SSIM evidence that nothing changed.
    assert 0 in kept
    assert 1 in kept
    assert 2 in kept
