from typing import Any, Dict, List, Optional, Tuple, Union

import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
from ..model import BaseModel, RunMode

from PIL import Image


class NemotronOCRV1(BaseModel):
    """
    Nemotron OCR v1 model for optical character recognition.

    End-to-end OCR model that integrates:
    - Text detector for region localization
    - Text recognizer for transcription
    - Relational model for layout and reading order analysis
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        from nemotron_ocr.inference.pipeline import NemotronOCR  # local-only import

        if model_dir:
            self._model = NemotronOCR(model_dir=model_dir)
        else:
            self._model = NemotronOCR()
        # NemotronOCR is a high-level pipeline (not an nn.Module). We can optionally
        # TensorRT-compile individual submodules (e.g. the detector backbone) but
        # must keep post-processing (NMS, box decoding, etc.) in eager PyTorch/C++.
        self._enable_trt = os.getenv("SLIMGEST_ENABLE_TORCH_TRT", "").strip().lower() in {"1", "true", "yes", "on"}
        if self._enable_trt and self._model is not None:
            self._maybe_compile_submodules()

    def _maybe_compile_submodules(self) -> None:
        """
        Best-effort TensorRT compilation of internal nn.Modules.
        Any failure falls back to eager PyTorch without breaking initialization.
        """
        try:
            import torch_tensorrt  # type: ignore
        except Exception:
            return

        # Detector is the safest candidate: input is a BCHW image tensor.
        if self._model is None:
            return

        detector = getattr(self._model, "detector", None)
        if not isinstance(detector, torch.nn.Module):
            return

        # NemotronOCR is a high-level pipeline (not an nn.Module). We can optionally
        # TensorRT-compile individual submodules (e.g. the detector backbone) but
        # must keep post-processing (NMS, box decoding, etc.) in eager PyTorch/C++.
        # TRT is compiled for B=1; invoke_batch() bypasses TRT via eager PyTorch.
        try:
            trt_input = torch_tensorrt.Input((1, 3, 1024, 1024), dtype=torch.float16)
        except TypeError:
            # Older/newer API variants: fall back to named arg.
            trt_input = torch_tensorrt.Input(shape=(1, 3, 1024, 1024), dtype=torch.float16)

        # If any torchvision NMS makes it into a compiled graph elsewhere, forcing
        # that op to run in Torch avoids hard failures.
        compile_kwargs: Dict[str, Any] = {
            "inputs": [trt_input],
            "enabled_precisions": {torch.float16},
        }
        if hasattr(torch_tensorrt, "compile"):
            for k in ("torch_executed_ops", "torch_executed_modules"):
                if k == "torch_executed_ops":
                    compile_kwargs[k] = {"torchvision::nms"}
                elif k == "torch_executed_modules":
                    compile_kwargs[k] = set()
            try:
                self._model.detector = torch_tensorrt.compile(detector, **compile_kwargs)
            except Exception:
                # Leave detector as-is on any failure.
                return

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor."""
        # no-op for now
        return tensor

    @staticmethod
    def _tensor_to_png_b64(img: torch.Tensor) -> str:
        """
        Convert a CHW/BCHW tensor into a base64-encoded PNG.

        Accepts:
          - CHW (3,H,W) or (1,H,W)
        Returns:
          - base64 string (no data: prefix)
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(img)}")
        if img.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")

        x = img.detach()
        if x.device.type != "cpu":
            x = x.cpu()

        # Convert to uint8 in [0,255]
        if x.dtype.is_floating_point:
            maxv = float(x.max().item()) if x.numel() else 1.0
            # Heuristic: treat [0,1] images as normalized.
            if maxv <= 1.5:
                x = x * 255.0
            x = x.clamp(0, 255).to(dtype=torch.uint8)
        else:
            x = x.clamp(0, 255).to(dtype=torch.uint8)

        c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        if c == 1:
            arr = x.squeeze(0).numpy()
            pil = Image.fromarray(arr, mode="L").convert("RGB")
        elif c == 3:
            arr = x.permute(1, 2, 0).contiguous().numpy()
            pil = Image.fromarray(arr, mode="RGB")
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {c}")

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _extract_text(obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, dict):
            for k in ("text", "output_text", "generated_text", "ocr_text"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # Some APIs return nested structures; best-effort flatten common shapes.
            if "words" in obj and isinstance(obj["words"], list):
                parts: List[str] = []
                for w in obj["words"]:
                    if isinstance(w, dict) and isinstance(w.get("text"), str):
                        parts.append(w["text"])
                if parts:
                    return " ".join(parts).strip()
        return str(obj).strip()

    def invoke(
        self,
        input_data: Union[torch.Tensor, str, bytes, np.ndarray, io.BytesIO],
        merge_level: str = "paragraph",
    ) -> Any:
        """
        Invoke OCR locally.

        Supports:
          - file path (str) **only if it exists**
          - base64 (str/bytes) (str is treated as base64 unless it is an existing file path)
          - NumPy array (HWC)
          - io.BytesIO
          - torch.Tensor (CHW/BCHW): converted to base64 PNG internally for compatibility
        """
        if self._model is None:
            raise RuntimeError("Local OCR model was not initialized.")

        # Convert torch tensors to base64 bytes (NemotronOCR expects file path/base64/ndarray/BytesIO).
        if isinstance(input_data, torch.Tensor):
            if input_data.ndim == 4:
                out: List[Any] = []
                for i in range(int(input_data.shape[0])):
                    b64 = self._tensor_to_png_b64(input_data[i])
                    out.extend(self._model(b64.encode("utf-8"), merge_level=merge_level))
                return out
            if input_data.ndim == 3:
                b64 = self._tensor_to_png_b64(input_data)
                return self._model(b64.encode("utf-8"), merge_level=merge_level)
            raise ValueError(f"Unsupported torch tensor shape for OCR: {tuple(input_data.shape)}")

        # Disambiguate str: existing file path vs base64 string.
        if isinstance(input_data, str):
            # s = input_data.strip()
            # breakpoint()
            # if s and Path(s).is_file():
            #     return self._model(s, merge_level=merge_level)
            # Treat as base64 string (nemotron_ocr expects bytes for base64).
            return self._model(input_data.encode("utf-8"), merge_level=merge_level)

        # bytes / ndarray / BytesIO are supported directly by nemotron_ocr.
        return self._model(input_data, merge_level=merge_level)

    def invoke_batch(
        self,
        inputs: List[np.ndarray],
        merge_levels: List[str],
    ) -> List[Any]:
        """
        Batched OCR inference on multiple HWC uint8 numpy arrays.

        Runs all images through the detector, recognizer, and relational
        models in a single batched forward pass for near-linear speedup.

        Parameters
        ----------
        inputs : list[np.ndarray]
            List of HWC uint8 crop arrays.
        merge_levels : list[str]
            Per-image merge level (``"word"`` / ``"sentence"`` / ``"paragraph"``).

        Returns
        -------
        list[list[dict]]
            One prediction list per input image.
        """
        if self._model is None:
            raise RuntimeError("Local OCR model was not initialized.")
        if not inputs:
            return []

        # If the upstream pipeline already exposes batch_call, prefer it.
        if hasattr(self._model, "batch_call"):
            return self._model.batch_call(inputs, merge_levels=merge_levels)

        return self._batched_inference(inputs, merge_levels)

    # ------------------------------------------------------------------
    # Batched inference using upstream model sub-modules directly
    # ------------------------------------------------------------------

    def _batched_inference(
        self,
        inputs: List[np.ndarray],
        merge_levels: List[str],
    ) -> List[Any]:
        import torch.nn.functional as F
        from torch import amp

        from nemotron_ocr.inference.pipeline import (
            PAD_COLOR,
            INFER_LENGTH,
            DETECTOR_DOWNSAMPLE,
            MERGE_LEVELS,
            NMS_PROB_THRESHOLD,
            NMS_IOU_THRESHOLD,
            NMS_MAX_REGIONS,
        )
        from nemotron_ocr.inference.pre_processing import interpolate_and_pad, pad_to_square
        from nemotron_ocr.inference.post_processing.data.text_region import TextBlock
        from nemotron_ocr.inference.post_processing.research_ops import (
            parse_relational_results,
            reorder_boxes,
        )
        from nemotron_ocr_cpp import (
            quad_non_maximal_suppression,
            region_counts_to_indices,
            rrect_to_quads,
        )

        m = self._model  # upstream NemotronOCR instance

        for ml in merge_levels:
            if ml not in MERGE_LEVELS:
                raise ValueError(f"Invalid merge level: {ml}. Must be one of {MERGE_LEVELS}.")

        # --- Convert each HWC ndarray → CHW float16, pad to square, resize ---
        # Each crop has a different size, so we must resize each to 1024×1024
        # *before* concatenating (torch.cat requires matching spatial dims).
        image_tensors = [m._load_image_to_tensor(img) for img in inputs]

        original_shapes: List[Tuple[int, ...]] = []
        padded_lengths: List[int] = []
        resized_singles: List[torch.Tensor] = []
        for img_t in image_tensors:
            orig_shape = img_t.shape[1:]  # (H, W)
            original_shapes.append(orig_shape)
            pl = max(orig_shape)
            padded_lengths.append(pl)
            resized = interpolate_and_pad(
                pad_to_square(img_t, pl, how="bottom_right").unsqueeze(0),
                PAD_COLOR,
                INFER_LENGTH,
            )
            resized_singles.append(resized)

        batch_tensor = torch.cat(resized_singles, dim=0)
        B = batch_tensor.shape[0]

        # --- Detector (single batched forward pass) ---
        with amp.autocast("cuda", enabled=True), torch.no_grad():
            det_conf, _, det_rboxes, det_feature_3 = m.detector(batch_tensor.cuda())

        # --- NMS (batch-aware since upstream CUDA fix) ---
        with amp.autocast("cuda", enabled=True), torch.no_grad():
            e2e_det_conf = torch.sigmoid(det_conf)
            e2e_det_coords = rrect_to_quads(det_rboxes.float(), DETECTOR_DOWNSAMPLE)

            quads, confidence, region_counts = quad_non_maximal_suppression(
                e2e_det_coords,
                e2e_det_conf,
                prob_threshold=NMS_PROB_THRESHOLD,
                iou_threshold=NMS_IOU_THRESHOLD,
                kernel_height=2,
                kernel_width=3,
                max_regions=NMS_MAX_REGIONS,
                verbose=False,
            )[:3]

        # --- Quad rectification & grid sampling ---
        if quads.shape[0] == 0:
            rec_rectified_quads = torch.empty(
                0, 128, 8, 32, dtype=torch.float32, device=batch_tensor.device
            )
            rel_rectified_quads = torch.empty(
                0, 128, 2, 3, dtype=torch.float32, device=batch_tensor.device
            )
        else:
            rec_rectified_quads = m.recognizer_quad_rectifier(
                quads.detach(), batch_tensor.shape[2], batch_tensor.shape[3]
            )
            rel_rectified_quads = m.relational_quad_rectifier(
                quads.cuda().detach(), batch_tensor.shape[2], batch_tensor.shape[3]
            )

            input_indices = region_counts_to_indices(region_counts, quads.shape[0])

            rec_rectified_quads = m.grid_sampler(
                det_feature_3.float(), rec_rectified_quads.float(), input_indices
            )
            rel_rectified_quads = m.grid_sampler(
                det_feature_3.float().cuda(),
                rel_rectified_quads,
                input_indices.cuda(),
            )

        # --- Recognizer ---
        if rec_rectified_quads.shape[0] == 0:
            rec_output = torch.empty(
                0, 32, 858, dtype=torch.float16, device=rec_rectified_quads.device
            )
            rec_features = torch.empty(
                0, 32, 256, dtype=torch.float16, device=rec_rectified_quads.device
            )
        else:
            with amp.autocast("cuda", enabled=True), torch.no_grad():
                rec_output, rec_features = m.recognizer(rec_rectified_quads.cuda())

        # --- Relational + per-image post-processing ---
        all_predictions: List[list] = []

        if region_counts.sum() > 0:
            rel_output = m.relational(
                rel_rectified_quads.cuda(),
                quads.cuda(),
                region_counts.cpu(),
                rec_features.cuda(),
            )
            words, lines, line_var = (
                rel_output["words"],
                rel_output["lines"],
                rel_output["line_log_var_unc"],
            )

            with amp.autocast("cuda", enabled=True), torch.no_grad():
                words = [F.softmax(r, dim=1, dtype=torch.float32)[:, 1:] for r in words]

                output = {
                    "sequences": F.softmax(rec_output, dim=2, dtype=torch.float32),
                    "region_counts": region_counts,
                    "quads": quads,
                    "raw_detector_confidence": e2e_det_conf,
                    "confidence": confidence,
                    "relations": words,
                    "line_relations": lines,
                    "line_rel_var": line_var,
                    "fg_colors": None,
                    "fonts": None,
                    "tt_log_var_uncertainty": None,
                    "e2e_recog_features": rec_features,
                }

            quads = output["quads"]

            # Per-image scale factors (crops may have different padded_lengths).
            lengths: List[float] = []
            for count, pl in zip(region_counts, padded_lengths):
                lengths.extend([pl / INFER_LENGTH] * count.item())

            lengths_tensor = torch.tensor(
                lengths, dtype=torch.float32, device=quads.device
            ).view(quads.shape[0], 1, 1)
            quads *= lengths_tensor

            batch_results = m.recog_encoder.convert_targets_to_labels(
                output, image_size=None, is_gt=False
            )
            relation_batch = m.relation_encoder.convert_targets_to_labels(
                output, image_size=None, is_gt=False
            )

            for example, rel_example in zip(batch_results, relation_batch):
                example.relation_graph = rel_example.relation_graph
                example.prune_invalid_relations()

            for example in batch_results:
                if example.relation_graph is None:
                    continue
                for paragraph in example.relation_graph:
                    block = []
                    for line in paragraph:
                        for relational_idx in line:
                            block.append(example[relational_idx])
                    if block:
                        example.blocks.append(TextBlock(block))

            for example in batch_results:
                for text_region in example:
                    text_region.region = text_region.region.vertices

            # Per-image prediction assembly.
            for example, merge_level, original_shape in zip(
                batch_results, merge_levels, original_shapes
            ):
                predictions: List[dict] = []
                boxes, texts, scores = parse_relational_results(
                    example, level=merge_level
                )
                boxes, texts, scores = reorder_boxes(
                    boxes, texts, scores, mode="top_left", dbscan_eps=10
                )

                orig_h, orig_w = original_shape

                if len(boxes) == 0:
                    pass  # no detections → empty predictions
                else:
                    boxes_array = np.array(boxes).reshape(-1, 4, 2)
                    boxes_array[:, :, 0] = boxes_array[:, :, 0] / orig_w
                    boxes_array[:, :, 1] = boxes_array[:, :, 1] / orig_h
                    boxes = boxes_array.astype(np.float16).tolist()

                    for box, text, conf in zip(boxes, texts, scores):
                        if box == "nan":
                            break
                        predictions.append(
                            {
                                "text": text,
                                "confidence": conf,
                                "left": min(p[0] for p in box),
                                "upper": max(p[1] for p in box),
                                "right": max(p[0] for p in box),
                                "lower": min(p[1] for p in box),
                            }
                        )

                all_predictions.append(predictions)
        else:
            all_predictions = [[] for _ in range(B)]

        return all_predictions

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "Nemotron OCR v1"

    @property
    def model_type(self) -> str:
        """Model category/type."""
        return "ocr"

    @property
    def model_runmode(self) -> RunMode:
        """Execution mode: local, NIM, or build-endpoint."""
        return "local"

    @property
    def input(self) -> Any:
        """
        Input schema for the model.

        Returns:
            dict: Schema describing RGB image input with variable dimensions
        """
        return {
            "type": "image",
            "format": "RGB",
            "supported_formats": ["PNG", "JPEG"],
            "data_types": ["float32", "uint8"],
            "dimensions": "variable (H x W)",
            "batch_support": True,
            "value_range": {"float32": "[0, 1]", "uint8": "[0, 255] (auto-converted)"},
            "aggregation_levels": ["word", "sentence", "paragraph"],
            "description": "Document or scene image in RGB format with automatic multi-scale resizing",
        }

    @property
    def output(self) -> Any:
        """
        Output schema for the model.

        Returns:
            dict: Schema describing OCR output format
        """
        return {
            "type": "ocr_results",
            "format": "structured",
            "structure": {
                "boxes": "List[List[List[float]]] - quadrilateral bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]",
                "texts": "List[str] - recognized text strings",
                "confidences": "List[float] - confidence scores per detection",
            },
            "properties": {
                "reading_order": True,
                "layout_analysis": True,
                "multi_line_support": True,
                "multi_block_support": True,
            },
            "description": "Structured OCR results with bounding boxes, recognized text, and confidence scores",
        }

    @property
    def input_batch_size(self) -> int:
        """Maximum or default input batch size."""
        return 8
