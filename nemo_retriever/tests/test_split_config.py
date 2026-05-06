# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for resolve_split_params."""

from __future__ import annotations

import pytest

from nemo_retriever.params import HtmlChunkParams, TextChunkParams
from nemo_retriever.params.utils import resolve_split_params


def test_resolve_split_params_behavior():
    """Single omnibus test: defaults, dict overrides, False off-switch, unknown key validation."""
    # Defaults: text/html default-ON; pdf/audio/image/video default-OFF.
    out = resolve_split_params(None)
    assert isinstance(out["text"], TextChunkParams)
    assert isinstance(out["html"], HtmlChunkParams)
    assert out["audio"] is None
    assert out["pdf"] is None
    assert out["image"] is None
    assert out["video"] is None

    # Dict override flips a default-off key on with custom params.
    out = resolve_split_params({"pdf": {"max_tokens": 256}, "text": False})
    assert isinstance(out["pdf"], TextChunkParams)
    assert out["pdf"].max_tokens == 256
    # Explicit False disables a default-on key.
    assert out["text"] is None

    # Unknown top-level key raises.
    with pytest.raises(ValueError, match="Unknown split_config key"):
        resolve_split_params({"pptx": {"max_tokens": 256}})
