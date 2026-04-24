# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared media (audio/video) chunking logic for the retriever.

Minimal copy of ffmpeg/ffprobe and MediaInterface semantics from
nv-ingest-api dataloader so the retriever stays self-contained.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import ffmpeg

    _FFMPEG_AVAILABLE = True
except Exception:
    ffmpeg = None  # type: ignore[assignment]
    _FFMPEG_AVAILABLE = False


class SplitType:
    """Split strategy for media; values match nv-ingest-api."""

    SIZE = "size"
    TIME = "time"
    FRAME = "frame"


def _probe(
    filename: str,
    format: Optional[str] = None,
    file_handle: Any = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    if not _FFMPEG_AVAILABLE or ffmpeg is None:
        raise RuntimeError("ffmpeg is required for media probing; install ffmpeg-python and system ffmpeg.")
    args = ["ffprobe", "-show_format", "-show_streams", "-of", "json"]
    args += ffmpeg._utils.convert_kwargs_to_cmd_line_args(kwargs)
    if file_handle:
        args += ["pipe:"]
    else:
        args += [filename]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    communicate_kwargs: dict = {}
    if timeout is not None:
        communicate_kwargs["timeout"] = timeout
    if file_handle:
        communicate_kwargs["input"] = file_handle
    out, err = p.communicate(**communicate_kwargs)
    if p.returncode != 0:
        raise ffmpeg._run.Error("ffprobe", out, err)
    return json.loads(out.decode("utf-8"))


def _get_audio_from_video(input_path: str, output_file: str, cache_path: Optional[str] = None) -> Optional[Path]:
    """Extract audio from a video file. Returns output Path or None on failure."""
    if not _FFMPEG_AVAILABLE or ffmpeg is None:
        raise RuntimeError("ffmpeg is required; install ffmpeg-python and system ffmpeg.")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        (
            ffmpeg.input(str(input_path))
            .output(str(output_path), acodec="libmp3lame", map="0:a")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return output_path
    except ffmpeg.Error as e:
        logger.error("FFmpeg error for file %s: %s", input_path, e.stderr.decode())
        return None


def _effective_cores() -> int:
    return int(max((os.cpu_count() or 4) * 0.2, 4))


def _coerce_float(value: Any) -> Optional[float]:
    """Return float(value) if it parses to a positive number, else None."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_duration(probe: dict, stream: dict, file_size: int) -> Optional[float]:
    """Best-effort duration lookup tolerant of incomplete ffprobe output.

    Order: container-level ``format.duration`` → stream-level ``duration`` →
    audio-bitrate fallback (``file_size * 8 / bit_rate``). Some MP3s omit
    ``format.duration`` entirely (no Xing/VBRI header), so we can't rely on it.
    """
    fmt = probe.get("format") or {}
    duration = _coerce_float(fmt.get("duration"))
    if duration is not None:
        return duration
    duration = _coerce_float(stream.get("duration"))
    if duration is not None:
        return duration
    bitrate = _coerce_float(fmt.get("bit_rate")) or _coerce_float(stream.get("bit_rate"))
    if bitrate is not None and file_size > 0:
        return (file_size * 8) / bitrate
    return None


class _LoaderInterface(ABC):
    @abstractmethod
    def split(self, input_path: str, output_dir: str, split_interval: int = 0) -> Any:
        pass

    @abstractmethod
    def _get_path_metadata(self, path: Optional[str] = None) -> Any:
        pass


class MediaInterface(_LoaderInterface):
    """Split and probe media files (audio/video) using ffmpeg/ffprobe."""

    def __init__(self) -> None:
        self.path_metadata: dict = {}

    def probe_media(
        self,
        path_file: Path,
        split_interval: int,
        split_type: str,
        file_handle: Any = None,
    ) -> Tuple[Optional[Any], Optional[float], Optional[float]]:
        """Return (probe, num_splits, duration)."""
        num_splits = None
        duration = None
        probe = None
        try:
            file_size = path_file.stat().st_size
            if file_handle:
                probe = _probe("pipe:", format=path_file.suffix, file_handle=file_handle)
            else:
                probe = _probe(str(path_file), format=path_file.suffix)

            streams = probe.get("streams") or []
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
            video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

            # Choose which stream defines sample-rate/duration. MP3s often
            # include an embedded cover-art "video" stream as streams[0], so
            # for audio-suffixed files we always prefer the audio stream.
            audio_exts = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
            use_audio = path_file.suffix.lower() in audio_exts or (audio_stream is not None and video_stream is None)

            if use_audio:
                if audio_stream is None:
                    raise ValueError(f"No audio stream found in {path_file}")
                sample_rate = float(audio_stream["sample_rate"])
                duration = _resolve_duration(probe, audio_stream, file_size)
            elif video_stream is not None:
                sample_rate = float(video_stream["avg_frame_rate"].split("/")[0])
                duration = _resolve_duration(probe, video_stream, file_size)
            else:
                raise ValueError(f"No usable audio/video stream in {path_file}")

            if duration is None:
                raise ValueError(f"Could not determine duration for {path_file}")

            num_splits = self.find_num_splits(file_size, sample_rate, duration, split_interval, split_type)
        except ffmpeg.Error as e:
            logger.error("FFmpeg error for file %s: %s", path_file, e.stderr.decode())
        except (ValueError, KeyError) as e:
            logger.error("Error probing media %s: %s", path_file, e)
        return (probe, num_splits, duration)

    def get_audio_from_video(
        self,
        input_path: str,
        output_file: str,
        cache_path: Optional[str] = None,
    ) -> Optional[Path]:
        return _get_audio_from_video(input_path, output_file, cache_path)

    def split(
        self,
        input_path: str,
        output_dir: str,
        split_interval: int = 0,
        split_type: str = SplitType.SIZE,
        cache_path: Optional[str] = None,
        video_audio_separate: bool = False,
        audio_only: bool = False,
    ) -> List[str]:
        """Split media into chunk files. Returns list of chunk file paths."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        original_input_path = input_path
        path_input = Path(input_path)
        if audio_only and path_input.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            out_mp3 = output_dir / f"{path_input.stem}.mp3"
            result = self.get_audio_from_video(str(input_path), str(out_mp3), cache_path)
            if result is None:
                return []
            input_path = str(result)
        path_file = Path(input_path)
        file_name = path_file.stem
        suffix = path_file.suffix
        output_pattern = output_dir / f"{file_name}_chunk_%04d{suffix}"
        num_splits = 0
        cache_path = cache_path or output_dir
        try:
            probe, num_splits, duration = self.probe_media(path_file, split_interval, split_type)
            if num_splits is None or duration is None or num_splits <= 0:
                return []
            segment_time = math.ceil(duration / num_splits)
            output_kwargs = {
                "f": "segment",
                "segment_time": segment_time,
                "c": "copy",
                "map": "0",
                "threads": _effective_cores(),
            }
            if suffix.lower() == ".mp4":
                output_kwargs.update(
                    {
                        "force_key_frames": f"expr:gte(t,n_forced*{segment_time})",
                        "crf": 22,
                        "g": 50,
                        "sc_threshold": 0,
                    }
                )
            (
                ffmpeg.input(str(input_path))
                .output(str(output_pattern), **output_kwargs)
                .run(capture_stdout=True, capture_stderr=True)
            )
            self.path_metadata[str(input_path)] = probe
        except ffmpeg.Error as e:
            logger.error("FFmpeg error for file %s: %s", original_input_path, e.stderr.decode())
            return []
        # Use actual chunk files produced by ffmpeg (may differ from num_splits)
        files = sorted(str(p) for p in output_dir.glob(f"{file_name}_chunk_*{suffix}") if p.is_file())
        if video_audio_separate and suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            for f in files:
                fp = Path(f)
                audio_path = self.get_audio_from_video(f, str(fp.with_suffix(".mp3")), str(cache_path))
                if audio_path is not None:
                    files.append(str(audio_path))
        return files

    def find_num_splits(
        self,
        file_size: int,
        sample_rate: float,
        duration: float,
        split_interval: int,
        split_type: str,
    ) -> float:
        if split_type == SplitType.SIZE:
            return math.ceil(file_size / split_interval)
        if split_type == SplitType.TIME:
            return math.ceil(duration / split_interval)
        if split_type == SplitType.FRAME:
            seconds_cap = split_interval / sample_rate
            return math.ceil(duration / seconds_cap)
        raise ValueError(f"Invalid split type: {split_type}")

    def _get_path_metadata(self, path: Optional[str] = None) -> dict:
        return self.path_metadata


def is_media_available() -> bool:
    """True if ffmpeg-python is installed and the ffprobe binary is on PATH."""
    return _FFMPEG_AVAILABLE and ffmpeg is not None and shutil.which("ffprobe") is not None
