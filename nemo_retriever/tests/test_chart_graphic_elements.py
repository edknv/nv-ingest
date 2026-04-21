# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the graphic-elements chart stage and its OCR-side joining."""

from __future__ import annotations

import base64
import importlib
import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_retriever.utils.table_and_chart import join_graphic_elements_and_ocr_output


def _can_import(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


_needs_pil = pytest.mark.skipif(not _can_import("PIL"), reason="PIL (Pillow) not installed")
_needs_requests = pytest.mark.skipif(not _can_import("requests"), reason="requests not installed")
_needs_torch = pytest.mark.skipif(not _can_import("torch"), reason="torch not installed")
_needs_cv2 = pytest.mark.skipif(not _can_import("cv2"), reason="cv2 (opencv) not installed")


def _make_b64_png(width: int = 200, height: int = 100) -> str:
    """Create a small synthetic PNG image encoded as base64."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# join_graphic_elements_and_ocr_output tests
# ---------------------------------------------------------------------------


class TestJoinGraphicElementsAndOCR:
    """Test the core joining function with synthetic data."""

    def test_empty_ocr_returns_empty(self) -> None:
        ge_dets = [
            {"bbox_xyxy_norm": [0.0, 0.0, 0.5, 0.5], "label_name": "chart_title"},
        ]
        result = join_graphic_elements_and_ocr_output(ge_dets, [], (100, 200))
        assert result == ""

    def test_no_ge_dets_returns_empty(self) -> None:
        ocr_preds = [
            {"left": 0.0, "right": 0.5, "upper": 0.0, "lower": 0.5, "text": "hello"},
        ]
        result = join_graphic_elements_and_ocr_output([], ocr_preds, (100, 200))
        assert result == ""

    def test_matching_ge_and_ocr(self) -> None:
        ge_dets = [
            {"bbox_xyxy_norm": [0.0, 0.0, 1.0, 0.3], "label_name": "chart_title"},
            {"bbox_xyxy_norm": [0.0, 0.7, 1.0, 1.0], "label_name": "xlabel"},
        ]
        ocr_preds = [
            {"left": 0.1, "right": 0.9, "upper": 0.05, "lower": 0.25, "text": "Sales Chart"},
            {"left": 0.1, "right": 0.9, "upper": 0.75, "lower": 0.95, "text": "Quarter"},
        ]
        result = join_graphic_elements_and_ocr_output(ge_dets, ocr_preds, (100, 200))
        assert result  # Non-empty
        assert "Sales Chart" in result or "Quarter" in result


# ---------------------------------------------------------------------------
# graphic_elements_ocr_page_elements tests
# ---------------------------------------------------------------------------


def _make_chart_page_df(
    width: int = 200,
    height: int = 100,
    has_chart: bool = True,
) -> pd.DataFrame:
    """Build a single-row page DataFrame with a chart detection."""
    image_b64 = _make_b64_png(width, height)
    detections = []
    if has_chart:
        detections.append(
            {
                "label_name": "chart",
                "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
                "score": 0.95,
            }
        )
    return pd.DataFrame(
        [
            {
                "page_image": {"image_b64": image_b64},
                "page_elements_v3": {"detections": detections},
                "page_elements_v3_counts_by_label": {"chart": len(detections)},
            }
        ]
    )


@_needs_pil
@_needs_requests
@_needs_torch
class TestGraphicElementsOCRPageElements:
    """Test the graphic-elements stage with mocked models."""

    def test_no_charts_produces_empty_chart_column(self) -> None:
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        df = _make_chart_page_df(has_chart=False)
        mock_ge_model = MagicMock()

        result = graphic_elements_ocr_page_elements(df, graphic_elements_model=mock_ge_model)
        assert "chart" in result.columns
        assert "graphic_elements_v1" in result.columns
        assert "graphic_elements_ocr_v1" in result.columns
        assert result.iloc[0]["chart"] == []
        assert result.iloc[0]["graphic_elements_v1"]["regions"] == []
        mock_ge_model.invoke.assert_not_called()

    def test_no_page_image_produces_empty_chart_column(self) -> None:
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        df = pd.DataFrame(
            [
                {
                    "page_image": {},
                    "page_elements_v3": {
                        "detections": [
                            {"label_name": "chart", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]},
                        ]
                    },
                }
            ]
        )
        mock_ge_model = MagicMock()

        result = graphic_elements_ocr_page_elements(df, graphic_elements_model=mock_ge_model)
        assert result.iloc[0]["chart"] == []
        assert result.iloc[0]["graphic_elements_v1"]["regions"] == []

    def test_with_mocked_model_produces_structure(self) -> None:
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        import torch

        df = _make_chart_page_df(width=200, height=100)

        mock_ge_model = MagicMock()
        mock_ge_model._model = MagicMock()
        mock_ge_model._model.labels = ["chart_title", "xlabel", "ylabel"]

        mock_pred = {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 0.3]]),
            "labels": torch.tensor([0]),  # chart_title
            "scores": torch.tensor([0.9]),
        }
        mock_ge_model.preprocess.return_value = torch.zeros(1, 3, 100, 200)
        mock_ge_model.invoke.return_value = mock_pred

        result = graphic_elements_ocr_page_elements(df, graphic_elements_model=mock_ge_model)

        # chart column carries a structure-only placeholder (no OCR text).
        assert "chart" in result.columns
        chart_entries = result.iloc[0]["chart"]
        assert len(chart_entries) == 1
        assert chart_entries[0]["bbox_xyxy_norm"] == [0.0, 0.0, 1.0, 1.0]
        assert "chart_title" in chart_entries[0]["text"]
        assert chart_entries[0]["counts_by_label"] == {"chart_title": 1}
        assert len(chart_entries[0]["detections"]) == 1

        # graphic_elements_v1 page-level column populated with region payload.
        assert "graphic_elements_v1" in result.columns
        ge_payload = result.iloc[0]["graphic_elements_v1"]
        assert set(ge_payload.keys()) >= {"regions", "timing", "error"}
        regions = ge_payload["regions"]
        assert len(regions) == 1
        region = regions[0]
        assert region["label_name"] == "chart"
        assert region["bbox_xyxy_norm"] == [0.0, 0.0, 1.0, 1.0]
        assert len(region["detections"]) == 1
        assert len(region["orig_shape_hw"]) == 2
        assert region["counts_by_label"] == {"chart_title": 1}

    def test_with_no_detections_produces_empty_summary(self) -> None:
        """When the GE model returns no detections, the chart entry has empty text."""
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        import torch

        df = _make_chart_page_df(width=200, height=100)

        mock_ge_model = MagicMock()
        mock_ge_model._model = MagicMock()
        mock_ge_model._model.labels = ["chart_title"]

        mock_pred = {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "scores": torch.zeros(0),
        }
        mock_ge_model.preprocess.return_value = torch.zeros(1, 3, 100, 200)
        mock_ge_model.invoke.return_value = mock_pred

        result = graphic_elements_ocr_page_elements(df, graphic_elements_model=mock_ge_model)

        chart_entries = result.iloc[0]["chart"]
        assert len(chart_entries) == 1
        assert chart_entries[0]["text"] == ""
        assert chart_entries[0]["counts_by_label"] == {}

    def test_model_error_recorded_in_metadata(self) -> None:
        """When the GE model raises, it should be recorded in metadata, not crash."""
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        import torch

        df = _make_chart_page_df(width=200, height=100)

        mock_ge_model = MagicMock()
        mock_ge_model._model = MagicMock()
        mock_ge_model._model.labels = ["chart_title"]
        mock_ge_model.preprocess.return_value = torch.zeros(1, 3, 100, 200)
        mock_ge_model.invoke.side_effect = RuntimeError("model exploded")

        result = graphic_elements_ocr_page_elements(df, graphic_elements_model=mock_ge_model)

        meta = result.iloc[0]["graphic_elements_ocr_v1"]
        assert meta["error"] is not None
        assert meta["error"]["type"] == "RuntimeError"
        assert "model exploded" in meta["error"]["message"]

    def test_requires_graphic_elements_model_when_no_url(self) -> None:
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        df = _make_chart_page_df()
        with pytest.raises(ValueError, match="graphic_elements_model"):
            graphic_elements_ocr_page_elements(df)


# ---------------------------------------------------------------------------
# GraphicElementsActor tests
# ---------------------------------------------------------------------------


@_needs_pil
class TestGraphicElementsActor:
    """Test the Ray actor wrapper."""

    def test_actor_error_returns_dataframe_with_error(self) -> None:
        """Actor should never raise; errors go into metadata columns."""
        from nemo_retriever.chart.chart_detection import GraphicElementsGPUActor

        actor = GraphicElementsGPUActor.__new__(GraphicElementsGPUActor)
        actor._graphic_elements_model = None
        actor._graphic_elements_invoke_url = ""
        actor._api_key = None
        actor._request_timeout_s = 120.0
        actor._remote_retry = None

        df = _make_chart_page_df()
        # This will fail because model is None and no URL set.
        result = actor(df)
        assert "chart" in result.columns
        assert "graphic_elements_ocr_v1" in result.columns
        meta = result.iloc[0]["graphic_elements_ocr_v1"]
        assert meta["error"] is not None


# ---------------------------------------------------------------------------
# OCR stage joining against graphic_elements_v1
# ---------------------------------------------------------------------------


def _make_page_df_with_ge_regions(
    *,
    ocr_bbox: list[float],
    ge_regions: list[dict],
    width: int = 200,
    height: int = 100,
) -> pd.DataFrame:
    image_b64 = _make_b64_png(width, height)
    return pd.DataFrame(
        [
            {
                "page_image": {"image_b64": image_b64},
                "page_elements_v3": {
                    "detections": [
                        {
                            "label_name": "chart",
                            "bbox_xyxy_norm": ocr_bbox,
                            "score": 0.95,
                        }
                    ]
                },
                "page_elements_v3_counts_by_label": {"chart": 1},
                "graphic_elements_v1": {
                    "regions": ge_regions,
                    "timing": {"seconds": 0.0},
                    "error": None,
                },
            }
        ]
    )


@_needs_pil
@_needs_requests
@_needs_torch
class TestOCRJoinsGraphicElements:
    """When use_graphic_elements=True, OCR stage should join GE + OCR."""

    def _single_title_detection(self) -> list[dict]:
        return [
            {"bbox_xyxy_norm": [0.0, 0.0, 1.0, 0.3], "label_name": "chart_title", "score": 0.9},
        ]

    def _ocr_preds_inside_and_outside(self) -> list[dict]:
        # "Title" falls inside the chart_title bbox; "Outside" is in the
        # bottom half, outside any GE region.
        return [
            {"left": 0.1, "right": 0.9, "upper": 0.05, "lower": 0.25, "text": "Title"},
            {"left": 0.1, "right": 0.9, "upper": 0.75, "lower": 0.95, "text": "Outside"},
        ]

    def test_local_path_joins_ge_and_ocr(self) -> None:
        from nemo_retriever.ocr.shared import ocr_page_elements

        bbox = [0.0, 0.0, 1.0, 1.0]
        ge_regions = [
            {
                "bbox_xyxy_norm": bbox,
                "label_name": "chart",
                "detections": self._single_title_detection(),
                "orig_shape_hw": [100, 200],
                "counts_by_label": {"chart_title": 1},
            }
        ]
        df = _make_page_df_with_ge_regions(ocr_bbox=bbox, ge_regions=ge_regions)

        ocr_model = MagicMock()
        ocr_model.invoke.return_value = self._ocr_preds_inside_and_outside()

        result = ocr_page_elements(
            df,
            model=ocr_model,
            extract_charts=True,
            use_graphic_elements=True,
        )

        entries = result.iloc[0]["chart"]
        assert len(entries) == 1
        text = entries[0]["text"]
        # Join path keeps only OCR text that overlaps a GE detection bbox.
        assert "Title" in text
        assert "Outside" not in text, f"expected structure-aware join to drop non-GE text, got: {text!r}"

    def test_local_path_falls_back_when_no_ge_match(self) -> None:
        """No matching GE region -> OCR-only chart text fallback."""
        from nemo_retriever.ocr.shared import ocr_page_elements

        # Empty regions → no match for the chart crop bbox.
        df = _make_page_df_with_ge_regions(ocr_bbox=[0.0, 0.0, 1.0, 1.0], ge_regions=[])

        ocr_model = MagicMock()
        ocr_model.invoke.return_value = self._ocr_preds_inside_and_outside()

        result = ocr_page_elements(
            df,
            model=ocr_model,
            extract_charts=True,
            use_graphic_elements=True,
        )

        entries = result.iloc[0]["chart"]
        assert len(entries) == 1
        text = entries[0]["text"]
        # Fallback path keeps all OCR words.
        assert "Title" in text
        assert "Outside" in text, f"expected OCR-only fallback to include all text, got: {text!r}"

    def test_remote_path_joins_ge_and_ocr(self) -> None:
        from nemo_retriever.ocr.shared import ocr_page_elements

        bbox = [0.0, 0.0, 1.0, 1.0]
        ge_regions = [
            {
                "bbox_xyxy_norm": bbox,
                "label_name": "chart",
                "detections": self._single_title_detection(),
                "orig_shape_hw": [100, 200],
                "counts_by_label": {"chart_title": 1},
            }
        ]
        df = _make_page_df_with_ge_regions(ocr_bbox=bbox, ge_regions=ge_regions)

        with patch(
            "nemo_retriever.ocr.shared.invoke_image_inference_batches",
            return_value=[self._ocr_preds_inside_and_outside()],
        ):
            result = ocr_page_elements(
                df,
                invoke_url="http://fake-ocr",
                extract_charts=True,
                use_graphic_elements=True,
            )

        entries = result.iloc[0]["chart"]
        assert len(entries) == 1
        text = entries[0]["text"]
        assert "Title" in text
        assert "Outside" not in text, f"expected structure-aware join to drop non-GE text, got: {text!r}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


@_needs_cv2
class TestGraphicElementsOCRConfig:
    def test_load_config_defaults(self) -> None:
        from nemo_retriever.chart.config import load_graphic_elements_ocr_config_from_dict

        cfg = load_graphic_elements_ocr_config_from_dict({})
        assert cfg.graphic_elements_invoke_url == ""
        assert cfg.api_key == ""
        assert cfg.request_timeout_s == 60.0

    def test_load_config_with_values(self) -> None:
        from nemo_retriever.chart.config import load_graphic_elements_ocr_config_from_dict

        cfg = load_graphic_elements_ocr_config_from_dict(
            {
                "graphic_elements_invoke_url": "http://ge:8000",
                "api_key": "secret",
                "request_timeout_s": 60.0,
            }
        )
        assert cfg.graphic_elements_invoke_url == "http://ge:8000"
        assert cfg.api_key == "secret"
        assert cfg.request_timeout_s == 60.0


# ---------------------------------------------------------------------------
# build_plan tests
# ---------------------------------------------------------------------------


@_needs_cv2
class TestBuildPlanChartStructure:
    def test_use_graphic_elements_selects_chart_structure_stage(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(
            stage_names_from_flags(
                extract_charts=True,
                use_graphic_elements=True,
            )
        )
        assert "enrich_graphic_elements" in names
        assert "enrich_chart" not in names

    def test_no_graphic_elements_selects_default_chart_stage(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(stage_names_from_flags(extract_charts=True))
        assert "enrich_chart" in names
        assert "enrich_graphic_elements" not in names

    def test_no_extract_charts_yields_no_chart_stage(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(stage_names_from_flags(extract_charts=False, use_graphic_elements=True))
        assert "enrich_graphic_elements" not in names
        assert "enrich_chart" not in names

    def test_graphic_elements_flag_does_not_affect_table_stages(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(
            stage_names_from_flags(
                extract_tables=True,
                extract_charts=True,
                use_graphic_elements=True,
                use_table_structure=True,
                table_output_format="markdown",
            )
        )
        assert "enrich_table_structure" in names
        assert "enrich_graphic_elements" in names


# ---------------------------------------------------------------------------
# _prediction_to_detections string labels test
# ---------------------------------------------------------------------------


@_needs_torch
class TestPredictionToDetectionsStringLabels:
    def test_string_labels_handled(self) -> None:
        import torch
        from nemo_retriever.chart.chart_detection import _prediction_to_detections

        pred = {
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            "labels": ["chart_title", "xlabel"],
            "scores": torch.tensor([0.9, 0.8]),
        }
        dets = _prediction_to_detections(pred, label_names=[])
        assert len(dets) == 2
        assert dets[0]["label_name"] == "chart_title"
        assert dets[1]["label_name"] == "xlabel"

    def test_integer_labels_still_work(self) -> None:
        import torch
        from nemo_retriever.chart.chart_detection import _prediction_to_detections

        pred = {
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
        }
        dets = _prediction_to_detections(pred, label_names=["chart_title", "xlabel"])
        assert len(dets) == 1
        assert dets[0]["label_name"] == "xlabel"
        assert dets[0]["label"] == 1
