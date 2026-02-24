"""Shared image helpers for the retriever pipeline.

All functions operate on HWC uint8 RGB numpy arrays, the canonical
in-memory representation for page images after PDF extraction.
"""

from __future__ import annotations

import base64
import io
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


def np_hwc_to_chw_float_tensor(
    arr: np.ndarray,
) -> Tuple["torch.Tensor", Tuple[int, int]]:
    """Convert an HWC uint8 numpy array to a CHW float32 torch tensor.

    Returns ``(tensor, (H, W))``.  The tensor values are in the 0-255 range
    (consistent with YOLOX preprocessing which pads with 114.0).
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required.")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    t = (
        torch.from_numpy(np.ascontiguousarray(arr))
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
    )
    return t, (h, w)


def crop_np_by_norm_bbox(
    arr: np.ndarray,
    bbox_xyxy_norm: Sequence[float],
) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    """Crop an HWC uint8 array by a normalised xyxy bounding box.

    Returns ``(crop_array, (crop_H, crop_W))`` or *None* if the crop is
    degenerate (zero-area, tiny, or bad coordinates).
    """
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if w <= 1 or h <= 1:
        return None

    try:
        x1n, y1n, x2n, y2n = [float(v) for v in bbox_xyxy_norm]
    except Exception:
        return None

    def _clamp_int(v: float, lo: int, hi: int) -> int:
        if v != v:  # NaN
            return lo
        return int(min(max(v, float(lo)), float(hi)))

    x1 = _clamp_int(x1n * w, 0, w)
    x2 = _clamp_int(x2n * w, 0, w)
    y1 = _clamp_int(y1n * h, 0, h)
    y2 = _clamp_int(y2n * h, 0, h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = arr[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[0], crop.shape[1]
    if cw <= 1 or ch <= 1:
        return None

    return crop, (ch, cw)


def np_rgb_to_b64(
    arr: np.ndarray,
    *,
    image_format: str = "jpeg",
    jpeg_quality: int = 95,
) -> Tuple[str, str]:
    """Encode an HWC uint8 RGB numpy array as a base64 string.

    Parameters
    ----------
    arr : np.ndarray
        HWC uint8 RGB image array.
    image_format : str
        ``"png"``, ``"jpeg"``, or ``"jpg"`` (alias for ``"jpeg"``).
    jpeg_quality : int
        JPEG quality (1-100). Only used when *image_format* is JPEG.

    Returns
    -------
    (b64_str, mime_type) : tuple[str, str]
        Base64-encoded image bytes and the corresponding MIME type
        (e.g. ``"image/png"`` or ``"image/jpeg"``).
    """
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required for image encoding.")

    fmt = image_format.lower().strip()
    if fmt in ("jpg", "jpeg"):
        pil_format = "JPEG"
        mime_type = "image/jpeg"
    elif fmt == "png":
        pil_format = "PNG"
        mime_type = "image/png"
    else:
        raise ValueError(
            f"Unsupported image_format={image_format!r}. Use 'png', 'jpeg', or 'jpg'."
        )

    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    if pil_format == "JPEG":
        img.save(buf, format=pil_format, quality=int(jpeg_quality))
    else:
        img.save(buf, format=pil_format)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64_str, mime_type


def np_rgb_to_b64_png(arr: np.ndarray) -> str:
    """Encode an HWC uint8 RGB numpy array as a base64-encoded PNG string.

    Backward-compatible wrapper around :func:`np_rgb_to_b64`.
    """
    b64_str, _ = np_rgb_to_b64(arr, image_format="png")
    return b64_str
