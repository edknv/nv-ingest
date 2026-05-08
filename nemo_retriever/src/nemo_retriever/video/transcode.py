# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VideoTranscodeActor: pre-transcode slow-codec videos to H.264.

Runs before VideoSplitActor.  Probes each video's codec; if it's not in
the fast-decode allowlist, transcodes to H.264 using NVENC (or libx264
fallback) and caches the output on disk.  Subsequent runs reuse cached
transcodes, which is where the real speedup lives — the first transcode
pass costs about as much as the original software decode would have,
because both bottleneck on the same software decoder.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.params import VideoTranscodeParams

logger = logging.getLogger(__name__)


def _ffprobe_codec(path: str) -> str:
    """Return the lowercase codec name for the first video stream, or empty string on error."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip().lower()
    except Exception:
        logger.warning("ffprobe failed on %s; treating as unknown codec", path, exc_info=True)
        return ""


def _encoder_available(encoder: str) -> bool:
    """Return True if the given ffmpeg encoder is available on this host."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return f" {encoder} " in result.stdout
    except Exception:
        return False


def _resolve_encoder(params: VideoTranscodeParams) -> str:
    """Pick the configured encoder if available, else the fallback."""
    if _encoder_available(params.encoder):
        return params.encoder
    if _encoder_available(params.encoder_fallback):
        logger.warning(
            "Encoder %s not available; falling back to %s.",
            params.encoder, params.encoder_fallback,
        )
        return params.encoder_fallback
    raise RuntimeError(
        f"Neither {params.encoder!r} nor {params.encoder_fallback!r} is available in ffmpeg."
    )


def _transcode_one(src: str, dest: str, *, encoder: str, preset: str, crf: int) -> None:
    """Run ffmpeg to transcode src -> dest using the given encoder."""
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", src,
        "-c:v", encoder,
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "copy",
        "-loglevel", "warning",
        dest,
    ]
    logger.info("Transcoding %s -> %s (encoder=%s preset=%s crf=%d)", src, dest, encoder, preset, crf)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
    if result.returncode != 0:
        # Clean up partial file.
        try:
            os.remove(dest)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg failed transcoding {src}: {result.stderr[:500]}")


@designer_component(
    name="Video Transcode",
    category="Video",
    compute="cpu",
    description="Pre-transcodes slow-codec videos (e.g. AV1) to H.264 with disk caching.",
)
class VideoTranscodeActor(AbstractOperator, CPUOperator):
    """Pre-transcode videos before frame extraction.

    Each video is probed; if its codec is in the configured fast-list,
    the row passes through unchanged.  Otherwise the video is transcoded
    to H.264 (or whichever ``target_codec`` is set) and cached on disk.
    Subsequent runs reuse the cached transcodes.
    """

    def __init__(self, params: VideoTranscodeParams | None = None) -> None:
        super().__init__(params=params)
        self._params = params or VideoTranscodeParams()
        self._encoder = _resolve_encoder(self._params)
        # Pre-create the cache dir so concurrent actors don't race.
        Path(self._params.cache_dir).mkdir(parents=True, exist_ok=True)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df
        if not self._params.enabled:
            return batch_df

        out = batch_df.copy()
        skip_codecs = {c.lower() for c in self._params.skip_codecs}

        for idx, row in out.iterrows():
            path = row.get("path")
            if not isinstance(path, str) or not path:
                continue
            codec = _ffprobe_codec(path)
            if not codec:
                # Couldn't probe; pass through and let VideoSplitActor try.
                continue
            if codec in skip_codecs:
                continue

            cache_name = Path(path).stem + f".transcoded.{self._params.target_codec}.mp4"
            cache_path = str(Path(self._params.cache_dir) / cache_name)

            if not os.path.exists(cache_path):
                try:
                    _transcode_one(
                        path,
                        cache_path,
                        encoder=self._encoder,
                        preset=self._params.preset,
                        crf=self._params.crf,
                    )
                except Exception:
                    logger.exception("Transcode failed for %s; passing through original.", path)
                    continue
            else:
                logger.info("Reusing cached transcode for %s -> %s", path, cache_path)

            out.at[idx, "path"] = cache_path
            # Drop bytes if present — VideoSplitActor reads from disk via path.
            if "bytes" in out.columns:
                out.at[idx, "bytes"] = None

        return out

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
