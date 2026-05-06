# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene boundary detection for the video extraction pipeline.

Wraps PySceneDetect's ``ContentDetector`` to return ``(start_seconds,
end_seconds)`` tuples for each shot in a video file.  Static videos
that produce zero detector scenes fall back to a single-scene label
that spans the whole video so the downstream pipeline always has a
scene to attach frames to.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)


def is_scenedetect_available() -> bool:
    try:
        import scenedetect  # noqa: F401
    except ImportError:
        return False
    return True


def detect_scenes(path: str, threshold: float = 30.0) -> List[Tuple[float, float]]:
    """Return ``[(start_seconds, end_seconds), ...]`` for shots in ``path``.

    Uses PySceneDetect's ``ContentDetector``.  When the detector returns
    zero scenes (very short or static video) the helper falls back to a
    single scene spanning ``[0, video_duration]``.
    """
    if not is_scenedetect_available():
        raise RuntimeError(
            "scene_detection requires PySceneDetect. "
            "Install with: pip install 'nemo-retriever[multimedia]'."
        )

    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(path)
    _dur = getattr(video, "duration", None)
    if _dur is not None:
        duration_secs = float(
            _dur.seconds if hasattr(_dur, "seconds") else _dur.get_seconds()
        )
    else:
        duration_secs = 0.0
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=float(threshold)))
    manager.detect_scenes(video)
    scene_list = manager.get_scene_list()

    if not scene_list:
        # Static video / no shot changes detected.
        if duration_secs <= 0.0:
            # Best effort: read total frames + fps via OpenCV.
            duration_secs = _probe_duration_seconds(path)
        return [(0.0, max(duration_secs, 0.0))]

    def _to_secs(tc: object) -> float:
        return float(tc.seconds if hasattr(tc, "seconds") else tc.get_seconds())  # type: ignore[union-attr]

    return [(_to_secs(start), _to_secs(end)) for start, end in scene_list]


def _probe_duration_seconds(path: str) -> float:
    try:
        import cv2
    except ImportError:
        return 0.0
    cap = cv2.VideoCapture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0.0 and frames > 0.0:
            return frames / fps
    finally:
        cap.release()
    return 0.0


def assign_scene_ids(
    frame_timestamps: Sequence[float],
    scenes: Sequence[Tuple[float, float]],
) -> List[Tuple[int, float, float]]:
    """Map each frame timestamp to ``(scene_id, scene_start, scene_end)``.

    Clamping rules for out-of-range timestamps:
    - ``ts < scenes[0][0]`` (negative or pre-video) clamps to scene 0
    - ``ts >= scenes[-1][1]`` (overrun past last scene's end) clamps to
      the last scene; this absorbs rounding noise on trailing frames.

    Raises ``ValueError`` when ``scenes`` is empty.
    """
    if not scenes:
        raise ValueError("assign_scene_ids requires at least one scene")
    out: List[Tuple[int, float, float]] = []
    last_idx = len(scenes) - 1
    first_start = scenes[0][0]
    for ts in frame_timestamps:
        if ts < first_start:
            scene_id = 0
        else:
            scene_id = last_idx
            for idx, (s_start, s_end) in enumerate(scenes):
                if s_start <= ts < s_end:
                    scene_id = idx
                    break
        s_start, s_end = scenes[scene_id]
        out.append((scene_id, float(s_start), float(s_end)))
    return out
