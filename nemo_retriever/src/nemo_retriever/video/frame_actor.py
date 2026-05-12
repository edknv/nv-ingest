# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VideoFrameActor: Ray Data map_batches callable for video frame extraction.

Consumes rows from rd.read_binary_files (path, bytes) and produces one row
per frame with path, source_path, image_b64, bytes, page_number, metadata.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from nemo_retriever.audio.media_interface import MediaInterface
from nemo_retriever.audio.media_interface import is_media_available
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.params import VideoFrameParams
from nemo_retriever.video import _content_types as _CT

logger = logging.getLogger(__name__)

_FRAME_TMPDIR_PREFIX = "retriever_video_frames_pid"


def _frame_tmpdir_prefix() -> str:
    """Tmpdir prefix tagged with the current PID so orphans from dead Ray
    workers (SIGKILL on memory pressure, init-retry restarts) can be
    distinguished from live ones at startup."""
    return f"{_FRAME_TMPDIR_PREFIX}{os.getpid()}_"


def _reap_orphaned_frame_tmpdirs(tmp_root: str = tempfile.gettempdir()) -> int:
    """Remove ``retriever_video_frames_pid<PID>_*`` directories whose owning
    PID is no longer alive. Returns the number of dirs removed.

    Why: prior runs that crashed via SIGKILL (Ray actor eviction, oomd, hard
    OOM) leave 80+ GiB of extracted frames per dead worker behind, because
    ``tempfile.TemporaryDirectory``'s context-manager cleanup never runs.
    20 such orphans is how we filled a 1.5 TB disk to 97%. Runs at module
    import so it fires once per Ray worker startup. Silent on any failure —
    cleanup is best-effort.
    """
    removed = 0
    try:
        entries = os.listdir(tmp_root)
    except OSError:
        return 0
    for name in entries:
        if not name.startswith(_FRAME_TMPDIR_PREFIX):
            continue
        # Parse the PID suffix: "retriever_video_frames_pid<PID>_<random>".
        rest = name[len(_FRAME_TMPDIR_PREFIX):]
        pid_str, _, _ = rest.partition("_")
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        # /proc/<pid> exists iff the process is alive (Linux). Skip when
        # the owner is still around — that's a live extract actor.
        if Path(f"/proc/{pid}").exists():
            continue
        path = os.path.join(tmp_root, name)
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not remove orphan frame tmpdir %s: %s", path, exc)
            continue
        removed += 1
        logger.warning("Reaped orphaned video-frame tmpdir from dead worker: %s", path)
    return removed


# Fire on import so each fresh Ray worker reclaims disk from prior crashes
# before its own extract starts.
_reap_orphaned_frame_tmpdirs()

# Adaptive fps tiers: (max_duration_seconds, fps).  Picked when the
# duration is at or below max_duration_seconds.  Pattern doubles
# duration as it halves fps, so the per-video frame budget stays at
# roughly 7,200 frames regardless of length.  Tunable per deployment
# via the source if a corpus has very different length distributions.
_ADAPTIVE_FPS_TIERS: tuple[tuple[float, float], ...] = (
    (3600.0, 2.0),    # <= 1h
    (7200.0, 1.0),    # <= 2h
    (14400.0, 0.5),   # <= 4h
    (28800.0, 0.25),  # <= 8h
)
_ADAPTIVE_FPS_FALLBACK: float = 0.125  # > 8h


def _choose_fps_for_duration(duration_secs: float) -> float:
    """Pick a sampling fps from :data:`_ADAPTIVE_FPS_TIERS`."""
    if duration_secs <= 0:
        return _ADAPTIVE_FPS_TIERS[0][1]  # default to highest fps when duration unknown
    for max_secs, fps in _ADAPTIVE_FPS_TIERS:
        if duration_secs <= max_secs:
            return fps
    return _ADAPTIVE_FPS_FALLBACK


# Output columns for downstream (OCR, embed, VDB).
FRAME_COLUMNS = [
    "path",
    "source_path",
    "image_b64",
    "page_number",
    "metadata",
    "bytes",
    "_content_type",
]


@designer_component(
    name="Video Frame Extractor",
    category="Video",
    compute="cpu",
    description="Extracts video frames at a fixed fps via ffmpeg",
    category_color="#ff6b6b",
)
class VideoFrameActor(AbstractOperator, CPUOperator):
    """
    Ray Data map_batches callable: DataFrame with path -> DataFrame of frame rows.

    Each output row has:
      - ``path``: original video path (frames are not persisted on disk;
        ``image_b64`` / ``bytes`` carry the pixels)
      - ``source_path``: original video path
      - ``image_b64``: base64-encoded PNG (the ``VideoFrameOCRActor`` reads this)
      - ``bytes``: raw PNG bytes (kept for compatibility with Ray Data binary readers)
      - ``page_number``: frame index (0, 1, 2, ...)
      - ``metadata``: dict with ``frame_timestamp_seconds``, ``segment_start_seconds``,
        ``segment_end_seconds``, ``fps``, ``source_path``, ``modality="video_frame"``,
        ``_content_type="video_frame"``.

    Frames are streamed to disk to avoid OOM on long videos.
    """

    def __init__(self, params: VideoFrameParams | None = None) -> None:
        super().__init__(params=params)
        if not is_media_available():
            raise RuntimeError(
                "VideoFrameActor requires ffmpeg. Install with: pip install ffmpeg-python and system ffmpeg."
            )
        self._params = params or VideoFrameParams()
        self._interface = MediaInterface()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame(columns=FRAME_COLUMNS)

        out_rows: List[Dict[str, Any]] = []
        for _, row in batch_df.iterrows():
            path = row.get("path")
            if path is None:
                continue
            path_str = str(path)
            if not path_str.strip():
                continue
            try:
                frame_rows = _extract_one(path_str, self._params, self._interface)
                out_rows.extend(frame_rows)
            except Exception as e:
                logger.exception("Error extracting frames from %s: %s", path_str, e)
                continue

        if not out_rows:
            return pd.DataFrame(columns=FRAME_COLUMNS)
        return pd.DataFrame(out_rows)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data


def _extract_one_legacy(
    source_path: str,
    params: VideoFrameParams,
    interface: MediaInterface,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> List[Dict[str, Any]]:
    """Extract frames from one video file (or a time window of it) and return row dicts.

    When ``start_seconds`` / ``end_seconds`` are set the ffmpeg invocation is
    bounded to that window of the source video (no temp slicing — the original
    file is read with output-side ``-ss`` / ``-t``). Frame timestamps in the
    returned metadata are shifted back into the source video's absolute
    timeline so downstream stages do not need to know about the chunking.
    """
    fps = float(params.fps)
    # When chunking, ``adaptive_fps`` should reflect chunk duration, not the
    # whole video — short chunks otherwise force a low fps tier and miss
    # detail. If a window was supplied, derive duration from it; else fall
    # back to probing the source file.
    if getattr(params, "adaptive_fps", False):
        # Reuse scene_detection's CV2 fallback prober to avoid a hard PySceneDetect dep.
        from nemo_retriever.video.scene_detection import _probe_duration_seconds

        if end_seconds is not None and end_seconds > start_seconds:
            duration_secs = float(end_seconds) - float(start_seconds)
        else:
            duration_secs = _probe_duration_seconds(source_path)
            if start_seconds > 0.0 and duration_secs > start_seconds:
                duration_secs = duration_secs - float(start_seconds)
        adaptive = _choose_fps_for_duration(duration_secs)
        if adaptive != fps:
            logger.info(
                "Adaptive fps: %s (duration %.1fs) -> %.3f fps (override of configured %.3f).",
                source_path,
                duration_secs,
                adaptive,
                fps,
            )
        fps = adaptive
    half_window = 0.5 / fps
    abs_offset = float(start_seconds or 0.0)
    with tempfile.TemporaryDirectory(prefix=_frame_tmpdir_prefix()) as tmpdir:
        frames = interface.extract_frames(
            source_path,
            tmpdir,
            fps=fps,
            max_frames=params.max_frames,
            start_seconds=abs_offset,
            end_seconds=end_seconds,
        )
        if not frames:
            logger.warning("No frames extracted from %s (ffmpeg returned 0 files)", source_path)
            return []

        rows: List[Dict[str, Any]] = []
        for idx, (frame_path, timestamp) in enumerate(frames):
            try:
                with open(frame_path, "rb") as f:
                    frame_bytes = f.read()
            except Exception as e:
                logger.warning("Could not read frame %s: %s", frame_path, e)
                continue
            # Free the on-disk frame the moment we have it in memory. ffmpeg
            # wrote all frames before returning, so the tmpdir peaks at the
            # full extract size (~80 GiB for a long video) — deleting as we
            # read brings it back down before the next chunk extracts.
            try:
                os.unlink(frame_path)
            except OSError:
                pass
            image_b64 = base64.b64encode(frame_bytes).decode("ascii")
            # Shift bounded-window timestamps back into the source-video timeline.
            abs_ts = float(timestamp) + abs_offset
            metadata = {
                "source_path": source_path,
                "frame_index": idx,
                "fps": fps,
                "frame_timestamp_seconds": abs_ts,
                "segment_start_seconds": max(0.0, abs_ts - half_window),
                "segment_end_seconds": abs_ts + half_window,
                "modality": _CT.VIDEO_FRAME,
                "_content_type": _CT.VIDEO_FRAME,
            }
            rows.append(
                {
                    # frame_path lives inside ``tmpdir`` which is deleted on
                    # return; consumers read ``image_b64`` / ``bytes``, not
                    # the file. Publish the source video instead of a stale ref.
                    "path": source_path,
                    "source_path": source_path,
                    "image_b64": image_b64,
                    "page_number": idx,
                    "metadata": metadata,
                    "bytes": frame_bytes,
                    "_content_type": _CT.VIDEO_FRAME,
                }
            )
        return rows


def _extract_one(
    source_path: str,
    params: VideoFrameParams,
    interface: MediaInterface,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> List[Dict[str, Any]]:
    """Route to scene-aware extraction when ``params.scene_detection.enabled``.

    Forwards ``start_seconds`` / ``end_seconds`` so per-chunk extraction
    (used by :class:`~nemo_retriever.video.VideoFrameExtractActor`) flows
    through both the legacy and scene-aware code paths.
    """
    if params.scene_detection.enabled:
        return _scene_aware_extract(
            source_path,
            params,
            interface,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
    return _extract_one_legacy(
        source_path,
        params,
        interface,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _emit_video_time_chunks(
    source_path: str,
    params: VideoFrameParams,
) -> List[Dict[str, Any]]:
    """Emit ``video_time_chunk`` descriptor rows — one per chunk_seconds window.

    Each row carries ``path``, ``source_path``, content type, and metadata
    with ``chunk_start_seconds`` / ``chunk_end_seconds`` and the configured
    ``fps``.  No frames are extracted yet — that happens in
    :class:`~nemo_retriever.video.VideoFrameExtractActor` (one Ray task
    per chunk, so a long video parallelises across actors).

    Falls back to a single ``[0.0, 0.0]`` chunk (downstream interprets as
    "extract the whole file") when the source duration cannot be probed.
    """
    from nemo_retriever.video.scene_detection import _probe_duration_seconds

    duration = _probe_duration_seconds(source_path)
    if duration <= 0.0:
        # Unknown duration — emit a single chunk covering the whole file.
        duration = 0.0
    chunk_seconds = max(1, int(params.time_chunking.chunk_seconds))
    chunks: List[Dict[str, Any]] = []
    if duration <= 0.0:
        # Single chunk; downstream extracts the whole file (end=0 -> open-ended).
        chunks.append(
            {
                "path": source_path,
                "source_path": source_path,
                "_content_type": _CT.VIDEO_TIME_CHUNK,
                "metadata": {
                    "_content_type": _CT.VIDEO_TIME_CHUNK,
                    "modality": _CT.VIDEO_TIME_CHUNK,
                    "source_path": source_path,
                    "chunk_index": 0,
                    "chunk_start_seconds": 0.0,
                    "chunk_end_seconds": 0.0,  # 0 = "until end"
                    "fps": float(params.fps),
                },
            }
        )
        return chunks
    t = 0.0
    chunk_idx = 0
    while t < duration:
        end = min(t + chunk_seconds, duration)
        chunks.append(
            {
                "path": source_path,
                "source_path": source_path,
                "_content_type": _CT.VIDEO_TIME_CHUNK,
                "metadata": {
                    "_content_type": _CT.VIDEO_TIME_CHUNK,
                    "modality": _CT.VIDEO_TIME_CHUNK,
                    "source_path": source_path,
                    "chunk_index": chunk_idx,
                    "chunk_start_seconds": float(t),
                    "chunk_end_seconds": float(end),
                    "fps": float(params.fps),
                },
            }
        )
        chunk_idx += 1
        t = end
    return chunks


def _attach_scene_metadata(
    rows: List[Dict[str, Any]],
    scenes: Sequence[tuple[float, float]],
) -> List[Dict[str, Any]]:
    """Stamp ``scene_id`` / ``scene_start_seconds`` / ``scene_end_seconds`` onto each row."""
    from nemo_retriever.video.scene_detection import assign_scene_ids

    timestamps = [
        float((r.get("metadata") or {}).get("frame_timestamp_seconds") or 0.0) for r in rows
    ]
    labels = assign_scene_ids(timestamps, scenes)
    for row, (scene_id, s_start, s_end) in zip(rows, labels):
        md = row.get("metadata") or {}
        md["scene_id"] = int(scene_id)
        md["scene_start_seconds"] = float(s_start)
        md["scene_end_seconds"] = float(s_end)
        row["metadata"] = md
    return rows


def _filter_indices(rows: List[Dict[str, Any]], keep: Sequence[int]) -> List[Dict[str, Any]]:
    keep_set = {int(i) for i in keep}
    return [r for i, r in enumerate(rows) if i in keep_set]


def _scene_aware_extract(
    source_path: str,
    params: VideoFrameParams,
    interface: MediaInterface,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> List[Dict[str, Any]]:
    """Run scene-aware frame extraction with optional SSIM and advanced dedup.

    Strategy: extract all frames once at ``params.fps`` (matching the
    legacy path), label each with its scene_id from PySceneDetect, then
    apply SSIM key-frame select and advanced dedup *per-scene* before
    merging the surviving rows.

    When ``start_seconds`` / ``end_seconds`` are provided (per-chunk
    extraction from :class:`~nemo_retriever.video.VideoFrameExtractActor`),
    raw extraction is bounded to that window and PySceneDetect runs on
    the full source file; we filter the returned scene list down to the
    window, clamping any scene that crosses a chunk border. This means
    a scene boundary that crosses a chunk border becomes two scenes —
    an accepted trade-off to gain per-chunk parallelism.
    """
    from nemo_retriever.video.advanced_dedup import advanced_dedup_indices
    from nemo_retriever.video.key_frame_select import select_key_frame_indices
    from nemo_retriever.video.scene_detection import detect_scenes

    raw_rows = _extract_one_legacy(
        source_path,
        params,
        interface,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    if not raw_rows:
        return raw_rows

    try:
        scenes = detect_scenes(source_path, threshold=params.scene_detection.threshold)
    except Exception:
        logger.warning(
            "Scene detection failed for %s; falling back to single scene",
            source_path,
            exc_info=True,
        )
        max_ts = max(
            float((r.get("metadata") or {}).get("frame_timestamp_seconds") or 0.0)
            for r in raw_rows
        )
        scenes = [(0.0, max_ts + 1.0)]
    assert scenes, "detect_scenes must return at least one scene; see scene_detection.detect_scenes fallback"

    # Restrict / clamp scenes to the chunk window when one was supplied.
    chunk_start = float(start_seconds or 0.0)
    chunk_end = float(end_seconds) if (end_seconds is not None and end_seconds > 0.0) else None
    if chunk_start > 0.0 or chunk_end is not None:
        clamped: List[tuple[float, float]] = []
        for s_start, s_end in scenes:
            s_start = float(s_start)
            s_end = float(s_end)
            if chunk_end is not None and s_start >= chunk_end:
                continue
            if s_end <= chunk_start:
                continue
            clamped.append(
                (
                    max(s_start, chunk_start),
                    min(s_end, chunk_end) if chunk_end is not None else s_end,
                )
            )
        if clamped:
            scenes = clamped
        else:
            # No scene overlapped the window; fall back to a single scene
            # spanning the chunk so frames still get a scene_id stamped.
            max_ts = max(
                float((r.get("metadata") or {}).get("frame_timestamp_seconds") or 0.0)
                for r in raw_rows
            )
            scenes = [(chunk_start, max(chunk_end if chunk_end is not None else max_ts + 1.0, max_ts + 1.0))]

    rows = _attach_scene_metadata(list(raw_rows), scenes)

    by_scene: Dict[int, List[int]] = {}
    for i, row in enumerate(rows):
        scene_id = int((row.get("metadata") or {}).get("scene_id") or 0)
        by_scene.setdefault(scene_id, []).append(i)

    keep_indices: List[int] = []
    for scene_id in sorted(by_scene.keys()):
        scene_idxs = by_scene[scene_id]
        scene_b64 = [rows[i].get("image_b64") or "" for i in scene_idxs]

        if params.key_frame_selection.enabled and len(scene_idxs) > 1:
            try:
                local = select_key_frame_indices(
                    scene_b64,
                    z_threshold=params.key_frame_selection.z_threshold,
                )
            except Exception:
                logger.warning(
                    "SSIM key-frame select failed; keeping all in scene %d",
                    scene_id,
                    exc_info=True,
                )
                local = list(range(len(scene_idxs)))
            scene_idxs = [scene_idxs[i] for i in local]
            scene_b64 = [rows[i].get("image_b64") or "" for i in scene_idxs]

        if params.advanced_dedup.enabled and scene_idxs:
            try:
                local = advanced_dedup_indices(
                    scene_b64,
                    blur_threshold=params.advanced_dedup.blur_threshold,
                    similarity_threshold=params.advanced_dedup.similarity_threshold,
                    entropy_gain_threshold=params.advanced_dedup.entropy_gain_threshold,
                )
            except Exception:
                logger.warning(
                    "Advanced dedup failed; keeping all in scene %d",
                    scene_id,
                    exc_info=True,
                )
                local = list(range(len(scene_idxs)))
            scene_idxs = [scene_idxs[i] for i in local]

        # All-blurry-scene fallback: if filtering emptied the scene
        # (e.g. solid title cards or black leader pass blur threshold),
        # keep the first raw frame from that scene so retrieval still
        # has at least one row anchored to its time window.
        if not scene_idxs and by_scene[scene_id]:
            scene_idxs = [by_scene[scene_id][0]]
            logger.debug(
                "All frames filtered out in scene %d; keeping first raw frame as fallback",
                scene_id,
            )

        keep_indices.extend(scene_idxs)

    keep_indices.sort()
    return _filter_indices(rows, keep_indices)


def _dhash(image_b64: str, hash_size: int = 8) -> Optional[int]:
    """Difference-hash of a base64-encoded PNG, packed into a 64-bit integer.

    Resize to ``(hash_size+1) x hash_size`` grayscale, compare each pixel to
    its right neighbour, pack the results as bits. Two frames with similar
    overall brightness layout end up close in Hamming distance, even if
    individual pixel values differ from encoder noise. Returns ``None`` if
    decoding fails so the caller can fall back to keeping the row.
    """
    try:
        import numpy as np
        from PIL import Image
    except Exception:  # pragma: no cover
        return None
    try:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.int16)
    except Exception:
        return None
    diff = arr[:, :-1] > arr[:, 1:]  # (hash_size, hash_size) bool
    return int.from_bytes(np.packbits(diff).tobytes(), "big")


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _seg_field(md: Any, key: str, default: float = 0.0) -> float:
    if isinstance(md, dict):
        try:
            return float(md.get(key) or default)
        except (TypeError, ValueError):
            return default
    return default


def dedup_video_frames(
    batch_df: pd.DataFrame,
    max_hamming_distance: int = 5,
    max_dropped_frames: int = 2,
) -> pd.DataFrame:
    """Collapse runs of perceptually-similar adjacent frames into one row each.

    Per ``source_path``, frames are sorted by ``segment_start_seconds`` and
    walked in order. A frame joins the current run when:
      - its dhash is within ``max_hamming_distance`` bits of the run's hash, AND
      - the gap to the run's current end is at most
        ``max_dropped_frames / fps`` seconds (so a small number of dropped
        frames in the middle is tolerated, but a long disappearance closes the run).
    The run's first frame is kept; its ``segment_end_seconds`` (and
    ``frame_timestamp_seconds`` midpoint) is extended to span the whole run.
    Other frames in the run are dropped.

    Without the time-window extension the kept row would say a slide was
    visible only for its first frame (e.g. [2.0, 4.0]) even when it stayed
    on screen for minutes — utterances during that span would miss the
    midpoint-overlap recall match. This function fixes that.
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df
    if "image_b64" not in batch_df.columns:
        return batch_df

    threshold = max(0, int(max_hamming_distance))
    max_drop = max(0, int(max_dropped_frames))

    work = batch_df.reset_index(drop=True).copy()
    work["__seg_start"] = work["metadata"].apply(lambda m: _seg_field(m, "segment_start_seconds"))

    output_rows: List[Dict[str, Any]] = []
    for _source, group in work.groupby("source_path", sort=False):
        group = group.sort_values("__seg_start")
        active_hash: Optional[int] = None
        active_idx: Optional[int] = None  # index in output_rows
        active_end: Optional[float] = None

        for _, row in group.iterrows():
            md = row.get("metadata")
            row_start = _seg_field(md, "segment_start_seconds")
            row_end = _seg_field(md, "segment_end_seconds")
            fps = _seg_field(md, "fps", default=1.0) or 1.0
            b64 = row.get("image_b64")

            row_dict = row.to_dict()
            row_dict.pop("__seg_start", None)

            h: Optional[int] = None
            if isinstance(b64, str) and b64:
                h = _dhash(b64)

            max_gap = float(max_drop) / max(fps, 0.001)
            can_merge = (
                active_hash is not None
                and h is not None
                and active_end is not None
                and _hamming(h, active_hash) <= threshold
                and (row_start - active_end) <= max_gap
            )
            if can_merge:
                # Extend the kept run's window through this frame; mutate
                # the kept metadata in place rather than copying it each merge.
                kept_md = output_rows[active_idx]["metadata"]
                if row_end > float(kept_md.get("segment_end_seconds") or 0.0):
                    kept_md["segment_end_seconds"] = row_end
                    start = float(kept_md.get("segment_start_seconds") or 0.0)
                    kept_md["frame_timestamp_seconds"] = (start + row_end) / 2.0
                kept_md["dedup_merged_count"] = int(kept_md.get("dedup_merged_count", 1)) + 1
                active_end = row_end
            else:
                # Detach metadata into a fresh dict so subsequent in-place mutations
                # don't leak back into the source DataFrame's row.
                md = row_dict.get("metadata")
                if isinstance(md, dict):
                    row_dict["metadata"] = dict(md)
                output_rows.append(row_dict)
                active_idx = len(output_rows) - 1
                active_end = row_end
                active_hash = h  # may be None for unhashable frames; that breaks the run

    if not output_rows:
        return pd.DataFrame()
    return pd.DataFrame(output_rows)


def video_path_to_frames_df(path: str, params: VideoFrameParams | None = None) -> pd.DataFrame:
    """Synchronous loader: one video file path -> DataFrame of frame rows.

    Columns match :data:`FRAME_COLUMNS`. Used by inprocess ingest() when
    ``_pipeline_type == "video"``.
    """
    if not is_media_available():
        raise RuntimeError("video_path_to_frames_df requires ffmpeg.")
    params = params or VideoFrameParams()
    interface = MediaInterface()
    rows = _extract_one(path, params, interface)
    if not rows:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    return pd.DataFrame(rows)
