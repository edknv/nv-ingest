# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for multimodal embedding helpers and explode_content_to_rows.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Pure helpers from main_text_embed (no transitive-import issues)
# ---------------------------------------------------------------------------
from nemo_retriever.text_embed.main_text_embed import (
    _format_image_input_string,
    _format_text_image_pair_input_string,
    _image_from_row,
    _multimodal_callable_runner,
)

# ---------------------------------------------------------------------------
# Stub heavy internal modules so ``from nemo_retriever.ingest_modes.inprocess``
# can be imported in lightweight CI (only pytest, pandas, pydantic, pyyaml).
#
# The ``nemo_retriever.ingest_modes`` __init__.py eagerly imports batch/fused/online
# which pull in ray, torch, nemotron_*, nv_ingest_api, etc.  And inprocess.py
# itself imports model/local (torch, nemotron_*), page_elements, ocr, and
# pdf.extract — each with their own heavy transitive deps.
#
# Rather than chasing every third-party leaf dependency, we pre-populate
# sys.modules for the heavy *internal* nemo_retriever sub-packages with MagicMock.
# This cuts off the entire transitive tree at the root.
# ---------------------------------------------------------------------------
_HEAVY_INTERNAL = [
    # -- sibling ingest modes (prevents batch.py/fused.py from loading) ------
    "nemo_retriever.ingest_modes.batch",
    "nemo_retriever.ingest_modes.fused",
    "nemo_retriever.ingest_modes.online",
    # -- model / ML packages (torch, nemotron_*, transformers) ---------------
    "nemo_retriever.model.local",
    "nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder",
    "nemo_retriever.model.local.nemotron_page_elements_v3",
    "nemo_retriever.model.local.nemotron_ocr_v1",
    "nemo_retriever.model.local.nemotron_table_structure_v1",
    "nemo_retriever.model.local.nemotron_graphic_elements_v1",
    # -- detection / OCR (nemotron_page_elements_v3, PIL, requests) ----------
    "nemo_retriever.page_elements",
    "nemo_retriever.page_elements.page_elements",
    "nemo_retriever.ocr",
    "nemo_retriever.ocr.ocr",
    # -- PDF (pypdfium2, nv_ingest_api via pdf/__init__ → __main__ → stage) --
    "nemo_retriever.pdf",
    "nemo_retriever.pdf.__main__",
    "nemo_retriever.pdf.config",
    "nemo_retriever.pdf.io",
    "nemo_retriever.pdf.stage",
    "nemo_retriever.pdf.extract",
    "nemo_retriever.pdf.split",
]
for _mod_name in _HEAVY_INTERNAL:
    sys.modules.setdefault(_mod_name, MagicMock())

from nemo_retriever.ingest_modes.inprocess import explode_content_to_rows  # noqa: E402


# ===================================================================
# Pure helpers
# ===================================================================


class TestImageFromRow:
    def test_returns_b64_when_present(self):
        row = pd.Series({"_image_b64": "abc123"})
        assert _image_from_row(row) == "abc123"

    @pytest.mark.parametrize("value", [None, "", "   ", 42])
    def test_returns_none_for_missing_empty_whitespace(self, value):
        data = {"_image_b64": value} if value is not None else {}
        row = pd.Series(data)
        assert _image_from_row(row) is None


class TestFormatInputStrings:
    def test_format_image_input_string(self):
        result = _format_image_input_string("AAAA")
        assert result == "data:image/png;base64,AAAA"

    def test_format_image_input_string_custom_mime(self):
        result = _format_image_input_string("BBBB", mime="image/jpeg")
        assert result == "data:image/jpeg;base64,BBBB"

    def test_format_text_image_pair_input_string(self):
        result = _format_text_image_pair_input_string("hello world", "CCCC")
        assert result == "hello world\ndata:image/png;base64,CCCC"


# ===================================================================
# _multimodal_callable_runner
# ===================================================================


class TestMultimodalCallableRunner:
    def test_image_mode(self):
        """Image-only mode calls embedder.embed_images() and returns embeddings."""
        embedder = MagicMock()
        embedder.embed_images.return_value = [[0.1, 0.2], [0.3, 0.4]]

        df = pd.DataFrame(
            {
                "text": ["page one", "page two"],
                "_image_b64": ["img1_b64", "img2_b64"],
            }
        )

        result = _multimodal_callable_runner(
            df,
            embedder=embedder,
            batch_size=64,
            embed_modality="image",
        )

        embedder.embed_images.assert_called_once()
        assert result["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
        assert len(result["info_msgs"]) == 2

    def test_text_image_fallback(self):
        """text_image mode: rows with images use embed_text_image(), rows without fall back to embed()."""
        embedder = MagicMock()
        # Row 0 has image -> embed_text_image
        # Row 1 has no image -> embed (text-only fallback)
        embedder.embed_text_image.return_value = [[1.0, 2.0]]
        embedder.embed.return_value = [[3.0, 4.0]]

        df = pd.DataFrame(
            {
                "text": ["with image", "text only"],
                "_image_b64": ["imgB64", ""],
            }
        )

        result = _multimodal_callable_runner(
            df,
            embedder=embedder,
            batch_size=64,
            embed_modality="text_image",
        )

        embedder.embed_text_image.assert_called_once()
        embedder.embed.assert_called_once()
        # Order must be preserved: row 0 (multimodal), row 1 (text fallback)
        assert result["embeddings"] == [[1.0, 2.0], [3.0, 4.0]]
        assert len(result["info_msgs"]) == 2


# ===================================================================
# explode_content_to_rows
# ===================================================================


class TestExplodeContentToRows:
    def test_text_mode_tags_modality(self):
        """Default text mode tags every row with _embed_modality='text' and no _image_b64."""
        df = pd.DataFrame(
            {
                "text": ["Hello world"],
                "table": [[{"text": "cell data"}]],
            }
        )

        result = explode_content_to_rows(df, embed_scope="element")

        assert "_embed_modality" in result.columns
        assert list(result["_embed_modality"]) == ["text", "text"]
        assert "_image_b64" not in result.columns

    @patch("nemo_retriever.ingest_modes.inprocess._crop_b64_image_by_norm_bbox")
    def test_text_image_carries_image(self, mock_crop):
        """text_image mode copies page image to _image_b64, crops for structured content."""
        mock_crop.return_value = ("cropped_b64", None)

        df = pd.DataFrame(
            {
                "text": ["some page text"],
                "page_image": [{"image_b64": "full_page_b64"}],
                "table": [[{"text": "table cell", "bbox_xyxy_norm": [0.1, 0.2, 0.9, 0.8]}]],
            }
        )

        result = explode_content_to_rows(df, modality="text_image", embed_scope="element")

        assert "_image_b64" in result.columns
        images = list(result["_image_b64"])
        modalities = list(result["_embed_modality"])

        # Row 0: page text row gets full page image
        assert images[0] == "full_page_b64"
        assert modalities[0] == "text_image"

        # Row 1: structured content row gets cropped image
        assert images[1] == "cropped_b64"
        assert modalities[1] == "text_image"

        mock_crop.assert_called_once_with(
            "full_page_b64",
            bbox_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
        )


# ===================================================================
# explode_content_to_rows — page scope
# ===================================================================


class TestExplodeContentToRowsPageScope:
    def test_page_scope_produces_one_row_per_page(self):
        """Page scope keeps one row per page — no explosion into element rows."""
        df = pd.DataFrame(
            {
                "text": ["Hello world"],
                "table": [[{"text": "cell data"}]],
                "chart": [[{"text": "chart caption"}]],
            }
        )

        result = explode_content_to_rows(df, embed_scope="page")

        assert len(result) == 1

    def test_page_scope_concatenates_all_text(self):
        """Page scope concatenates page text + table + chart + infographic text."""
        df = pd.DataFrame(
            {
                "text": ["Page paragraph."],
                "table": [[{"text": "Table content."}]],
                "chart": [[{"text": "Chart caption."}]],
                "infographic": [[{"text": "Infographic text."}]],
            }
        )

        result = explode_content_to_rows(df, embed_scope="page")

        combined = result["text"].iloc[0]
        assert "Page paragraph." in combined
        assert "Table content." in combined
        assert "Chart caption." in combined
        assert "Infographic text." in combined

    def test_page_scope_uses_full_page_image(self):
        """Page scope uses the full page image — no cropping."""
        df = pd.DataFrame(
            {
                "text": ["some text"],
                "page_image": [{"image_b64": "full_page_b64"}],
                "table": [[{"text": "cell", "bbox_xyxy_norm": [0.1, 0.2, 0.9, 0.8]}]],
            }
        )

        result = explode_content_to_rows(df, modality="text_image", embed_scope="page")

        assert len(result) == 1
        assert result["_image_b64"].iloc[0] == "full_page_b64"
        assert result["_embed_modality"].iloc[0] == "text_image"

    def test_page_scope_empty_page(self):
        """An empty page (no text, no structured content) still produces one row."""
        df = pd.DataFrame(
            {
                "text": [""],
                "table": [[]],
            }
        )

        result = explode_content_to_rows(df, embed_scope="page")

        assert len(result) == 1
        assert result["text"].iloc[0] == ""

    def test_page_scope_multiple_pages(self):
        """Page scope produces exactly one row per page across multiple pages."""
        df = pd.DataFrame(
            {
                "text": ["Page 1 text", "Page 2 text"],
                "table": [
                    [{"text": "table1"}],
                    [{"text": "table2a"}, {"text": "table2b"}],
                ],
            }
        )

        result = explode_content_to_rows(df, embed_scope="page")

        assert len(result) == 2
        assert "table1" in result["text"].iloc[0]
        assert "table2a" in result["text"].iloc[1]
        assert "table2b" in result["text"].iloc[1]

    def test_element_scope_unchanged(self):
        """Default element scope still explodes into per-element rows (regression test)."""
        df = pd.DataFrame(
            {
                "text": ["Hello world"],
                "table": [[{"text": "cell data"}]],
            }
        )

        result = explode_content_to_rows(df, embed_scope="element")

        # Element scope: 1 text row + 1 table row = 2 rows
        assert len(result) == 2
