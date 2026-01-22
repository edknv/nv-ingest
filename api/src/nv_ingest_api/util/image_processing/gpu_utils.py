# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU utilities for image processing using CuPy.

This module provides GPU-accelerated image processing functions with automatic
CPU fallback when CUDA is not available.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try to import CuPy for GPU acceleration and verify CUDA device is available
try:
    import cupy as cp

    # Verify that a CUDA device is actually available
    cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE = True
    logger.info("CuPy available with CUDA device - GPU acceleration enabled for image processing")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    logger.info("CuPy not available - using CPU fallback for image processing")
except Exception as e:
    # CuPy imported but no CUDA device available
    cp = None
    GPU_AVAILABLE = False
    logger.info(f"CuPy available but no CUDA device detected - using CPU fallback: {e}")


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return GPU_AVAILABLE


def to_gpu(arr: np.ndarray):
    """
    Move a numpy array to GPU memory if available.

    Parameters
    ----------
    arr : np.ndarray
        Input numpy array.

    Returns
    -------
    cupy.ndarray or np.ndarray
        GPU array if CuPy is available, otherwise returns the input unchanged.
    """
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


def to_cpu(arr) -> np.ndarray:
    """
    Move an array from GPU to CPU memory.

    Parameters
    ----------
    arr : cupy.ndarray or np.ndarray
        Input array (GPU or CPU).

    Returns
    -------
    np.ndarray
        CPU numpy array.
    """
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def pad_image_gpu(
    array: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    background_color: int = 255,
) -> np.ndarray:
    """
    Pad an image using GPU acceleration.

    Parameters
    ----------
    array : np.ndarray
        Input image array of shape (H, W, C).
    pad_top : int
        Padding to add at the top.
    pad_bottom : int
        Padding to add at the bottom.
    pad_left : int
        Padding to add on the left.
    pad_right : int
        Padding to add on the right.
    background_color : int, optional
        Value to use for padding. Defaults to 255 (white).

    Returns
    -------
    np.ndarray
        Padded image array.
    """
    if not GPU_AVAILABLE:
        return np.pad(
            array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=background_color,
        )

    try:
        arr_gpu = cp.asarray(array)
        padded = cp.pad(
            arr_gpu,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=background_color,
        )
        return cp.asnumpy(padded)
    except Exception as e:
        logger.warning(f"GPU padding failed, falling back to CPU: {type(e).__name__}: {e}")
        return np.pad(
            array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=background_color,
        )


def resize_image_gpu(
    array: np.ndarray,
    target_size: tuple,
    order: int = 1,
) -> np.ndarray:
    """
    Resize an image using GPU acceleration.

    Parameters
    ----------
    array : np.ndarray
        Input image array of shape (H, W, C).
    target_size : tuple
        Target size as (width, height).
    order : int, optional
        Interpolation order (0=nearest, 1=bilinear, 3=cubic). Defaults to 1.

    Returns
    -------
    np.ndarray
        Resized image array.
    """
    import cv2

    if not GPU_AVAILABLE:
        return cv2.resize(array, target_size, interpolation=cv2.INTER_LINEAR)

    try:
        # Lazy import to defer JIT compilation until actually needed
        from cupyx.scipy.ndimage import zoom as cupy_zoom

        target_width, target_height = target_size
        height, width = array.shape[:2]

        scale_y = target_height / height
        scale_x = target_width / width

        arr_gpu = cp.asarray(array)

        # Handle different number of channels
        if arr_gpu.ndim == 3:
            zoom_factors = (scale_y, scale_x, 1)
        else:
            zoom_factors = (scale_y, scale_x)

        resized = cupy_zoom(arr_gpu, zoom_factors, order=order)
        return cp.asnumpy(resized).astype(array.dtype)
    except Exception as e:
        logger.warning(f"GPU resize failed, falling back to CPU: {type(e).__name__}: {e}")
        return cv2.resize(array, target_size, interpolation=cv2.INTER_LINEAR)


def convert_rgb_to_bgr_gpu(array: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR using GPU acceleration.

    Parameters
    ----------
    array : np.ndarray
        Input RGB image array of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        BGR image array.
    """
    import cv2

    if not GPU_AVAILABLE:
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    try:
        arr_gpu = cp.asarray(array)
        # Flip channels: RGB -> BGR
        converted = arr_gpu[..., ::-1].copy()
        return cp.asnumpy(converted)
    except Exception as e:
        logger.warning(f"GPU RGB to BGR failed, falling back to CPU: {type(e).__name__}: {e}")
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def convert_bgr_to_rgb_gpu(array: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB using GPU acceleration.

    Parameters
    ----------
    array : np.ndarray
        Input BGR image array of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        RGB image array.
    """
    import cv2

    if not GPU_AVAILABLE:
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    try:
        arr_gpu = cp.asarray(array)
        # Flip channels: BGR -> RGB
        converted = arr_gpu[..., ::-1].copy()
        return cp.asnumpy(converted)
    except Exception as e:
        logger.warning(f"GPU BGR to RGB failed, falling back to CPU: {type(e).__name__}: {e}")
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def rgba_to_rgb_gpu(rgba_array: np.ndarray, background_color: int = 255) -> np.ndarray:
    """
    Convert RGBA image to RGB by alpha blending with a background color using GPU.

    Parameters
    ----------
    rgba_array : np.ndarray
        Input RGBA image array of shape (H, W, 4).
    background_color : int, optional
        Background color for alpha blending. Defaults to 255 (white).

    Returns
    -------
    np.ndarray
        RGB image array of shape (H, W, 3).
    """
    def _cpu_rgba_to_rgb(rgba_array, background_color):
        rgb = rgba_array[:, :, :3].astype(np.float32)
        alpha = rgba_array[:, :, 3:4].astype(np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        rgb_image = rgb * alpha + background_color * (1 - alpha)
        return rgb_image.astype(np.uint8)

    if not GPU_AVAILABLE:
        return _cpu_rgba_to_rgb(rgba_array, background_color)

    try:
        arr_gpu = cp.asarray(rgba_array)
        rgb = arr_gpu[:, :, :3].astype(cp.float32)
        alpha = arr_gpu[:, :, 3:4].astype(cp.float32)

        if alpha.max() > 1.0:
            alpha = alpha / 255.0

        rgb_image = rgb * alpha + background_color * (1 - alpha)
        return cp.asnumpy(rgb_image).astype(np.uint8)
    except Exception as e:
        logger.warning(f"GPU RGBA to RGB failed, falling back to CPU: {type(e).__name__}: {e}")
        return _cpu_rgba_to_rgb(rgba_array, background_color)
