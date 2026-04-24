# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for nemo_retriever.audio: ASRActor (with mocked Parakeet client).

Avoids importing nemo_retriever.model (and thus torch) by not eagerly loading
the model package; local-ASR tests inject a fake nemo_retriever.model.local
into sys.modules so the real module is never loaded.
"""

import base64
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_retriever.audio.asr_actor import ASRActor, ASRCPUActor, ASRGPUActor
from nemo_retriever.audio.asr_actor import apply_asr_to_df
from nemo_retriever.params import ASRParams


def _install_fake_local_module(mock_model):
    """Install a fake ``nemo_retriever.model.local`` whose ParakeetCTC1B1ASR yields mock_model.

    Returns the previous module (if any) so callers can restore it.
    """
    mock_class = MagicMock(return_value=mock_model)
    mock_local = MagicMock()
    mock_local.ParakeetCTC1B1ASR = mock_class
    prev_local = sys.modules.get("nemo_retriever.model.local")
    sys.modules["nemo_retriever.model.local"] = mock_local
    return prev_local


def _restore_local_module(prev_local):
    if prev_local is None:
        sys.modules.pop("nemo_retriever.model.local", None)
    else:
        sys.modules["nemo_retriever.model.local"] = prev_local


def test_asr_actor_empty_batch():
    with patch("nemo_retriever.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_get.return_value = mock_client

        params = ASRParams(audio_endpoints=("localhost:50051", None))
        actor = ASRActor(params=params)
        empty = pd.DataFrame(columns=["path", "bytes"])
        out = actor(empty)

        assert isinstance(out, pd.DataFrame)
        assert "text" in out.columns
        assert len(out) == 0
        mock_client.infer.assert_not_called()


def test_asr_actor_mock_transcribe():
    with patch("nemo_retriever.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.infer.return_value = ([], "hello world transcript")
        mock_get.return_value = mock_client

        params = ASRParams(audio_endpoints=("localhost:50051", None))
        actor = ASRActor(params=params)
        raw = b"\x00\x00\x00\x00"
        batch = pd.DataFrame(
            [
                {
                    "path": "/tmp/chunk.wav",
                    "bytes": raw,
                    "source_path": "/tmp/source.wav",
                    "duration": 1.0,
                    "chunk_index": 0,
                    "metadata": {"source_path": "/tmp/source.wav", "chunk_index": 0, "duration": 1.0},
                    "page_number": 0,
                }
            ]
        )
        out = actor(batch)

        assert len(out) == 1
        assert out["text"].iloc[0] == "hello world transcript"
        assert out["path"].iloc[0] == "/tmp/chunk.wav"
        assert out["source_path"].iloc[0] == "/tmp/source.wav"
        mock_client.infer.assert_called_once()
        call_arg = mock_client.infer.call_args[0][0]
        assert call_arg == base64.b64encode(raw).decode("ascii")


def test_apply_asr_to_df():
    with patch("nemo_retriever.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.infer.return_value = ([], "applied transcript")
        mock_get.return_value = mock_client

        batch = pd.DataFrame(
            [
                {
                    "path": "/p",
                    "bytes": b"x",
                    "source_path": "/s",
                    "duration": 0.5,
                    "chunk_index": 0,
                    "metadata": {},
                    "page_number": 0,
                }
            ]
        )
        out = apply_asr_to_df(batch, asr_params={"audio_endpoints": ("localhost:50051", None)})
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 1
        assert out["text"].iloc[0] == "applied transcript"


def test_asr_actor_remote_segment_audio():
    with patch("nemo_retriever.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.infer.return_value = (
            [
                {"start": 0.0, "end": 1.0, "text": "Hello world."},
                {"start": 1.0, "end": 2.5, "text": "How are you?"},
            ],
            "Hello world. How are you?",
        )
        mock_get.return_value = mock_client

        params = ASRParams(audio_endpoints=("localhost:50051", None), segment_audio=True)
        actor = ASRActor(params=params)
        batch = pd.DataFrame(
            [
                {
                    "path": "/tmp/chunk.wav",
                    "bytes": b"fake_audio",
                    "source_path": "/tmp/source.wav",
                    "duration": 2.5,
                    "chunk_index": 3,
                    "metadata": {"source_path": "/tmp/source.wav", "chunk_index": 3, "duration": 2.5},
                    "page_number": 3,
                }
            ]
        )
        out = actor(batch)

        assert len(out) == 2
        assert out["text"].tolist() == ["Hello world.", "How are you?"]
        assert out["page_number"].tolist() == [3, 3]
        assert out["chunk_index"].tolist() == [3, 3]
        assert out["metadata"].iloc[0]["segment_index"] == 0
        assert out["metadata"].iloc[0]["segment_count"] == 2
        assert out["metadata"].iloc[0]["segment_start"] == 0.0
        assert out["metadata"].iloc[0]["segment_end"] == 1.0
        assert out["metadata"].iloc[1]["segment_index"] == 1
        assert out["metadata"].iloc[1]["segment_start"] == 1.0
        assert out["metadata"].iloc[1]["segment_end"] == 2.5


def test_apply_asr_to_df_segment_audio():
    with patch("nemo_retriever.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.infer.return_value = (
            [
                {"start": 0.0, "end": 0.4, "text": "First sentence."},
                {"start": 0.4, "end": 0.8, "text": "Second sentence!"},
            ],
            "First sentence. Second sentence!",
        )
        mock_get.return_value = mock_client

        batch = pd.DataFrame(
            [
                {
                    "path": "/p",
                    "bytes": b"x",
                    "source_path": "/s",
                    "duration": 0.8,
                    "chunk_index": 0,
                    "metadata": {},
                    "page_number": 0,
                }
            ]
        )
        out = apply_asr_to_df(
            batch,
            asr_params={"audio_endpoints": ("localhost:50051", None), "segment_audio": True},
        )
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 2
        assert out["text"].tolist() == ["First sentence.", "Second sentence!"]
        assert out["metadata"].iloc[0]["segment_count"] == 2


def test_local_asr_does_not_call_get_client():
    """When audio_endpoints are both null, ASRCPUActor uses local model and does not call _get_client."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ["mocked local transcript"]
    prev_local = _install_fake_local_module(mock_model)
    try:
        with patch("nemo_retriever.audio.asr_actor._get_client") as mock_get:
            params = ASRParams(audio_endpoints=(None, None))
            actor = ASRCPUActor(params=params)

            mock_get.assert_not_called()
            assert actor._client is None
            assert actor._model is mock_model

            batch = pd.DataFrame(
                [
                    {
                        "path": "/tmp/chunk.wav",
                        "bytes": b"fake_audio_bytes",
                        "source_path": "/tmp/source.wav",
                        "duration": 1.0,
                        "chunk_index": 0,
                        "metadata": {},
                        "page_number": 0,
                    }
                ]
            )
            out = actor(batch)

            assert len(out) == 1
            assert out["text"].iloc[0] == "mocked local transcript"
            mock_model.transcribe.assert_called_once()
            call_args = mock_model.transcribe.call_args[0][0]
            assert isinstance(call_args, list)
            assert len(call_args) == 1
    finally:
        _restore_local_module(prev_local)


def test_local_asr_apply_asr_to_df():
    """apply_asr_to_df with audio_endpoints=(None, None) uses local model via the GPU variant when mocked."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ["apply local text"]
    prev_local = _install_fake_local_module(mock_model)
    try:
        with patch("nemo_retriever.audio.asr_actor._get_client") as mock_get:
            batch = pd.DataFrame(
                [
                    {
                        "path": "/p",
                        "bytes": b"x",
                        "source_path": "/s",
                        "duration": 0.5,
                        "chunk_index": 0,
                        "metadata": {},
                        "page_number": 0,
                    }
                ]
            )
            out = apply_asr_to_df(batch, asr_params={"audio_endpoints": (None, None)})

            mock_get.assert_not_called()
            assert len(out) == 1
            assert out["text"].iloc[0] == "apply local text"
    finally:
        _restore_local_module(prev_local)


def test_asr_gpu_actor_uses_local_model():
    """ASRGPUActor loads the local model and never constructs a remote client."""
    mock_model = MagicMock()
    prev_local = _install_fake_local_module(mock_model)
    try:
        with patch("nemo_retriever.audio.asr_actor._get_client") as mock_get:
            actor = ASRGPUActor(params=ASRParams(audio_endpoints=(None, None)))
            mock_get.assert_not_called()
            assert actor._client is None
            assert actor._model is mock_model
    finally:
        _restore_local_module(prev_local)


def test_asr_gpu_actor_rejects_remote_endpoints():
    """Constructing ASRGPUActor with a configured remote endpoint raises ValueError."""
    with pytest.raises(ValueError, match="does not support remote endpoints"):
        ASRGPUActor(params=ASRParams(audio_endpoints=("localhost:50051", None)))


def test_asr_archetype_prefers_cpu_when_remote():
    """ASRActor.prefers_cpu_variant is True iff remote endpoints are configured."""
    remote_params = ASRParams(audio_endpoints=("localhost:50051", None))
    local_params = ASRParams(audio_endpoints=(None, None))

    assert ASRActor.prefers_cpu_variant({"params": remote_params}) is True
    assert ASRActor.prefers_cpu_variant({"params": local_params}) is False
    assert ASRActor.prefers_cpu_variant(None) is False
    assert ASRActor.prefers_cpu_variant({}) is False
