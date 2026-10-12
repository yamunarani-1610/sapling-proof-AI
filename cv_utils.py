"""Image input normalization helpers for the computer-vision pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
from numpy.typing import NDArray

ImageInput = (
    str
    | os.PathLike[str]
    | bytes
    | bytearray
    | memoryview
    | NDArray[np.generic]
    | BinaryIO
)


def _read_upload(upload: Any) -> bytes:
    """Read an UploadedFile or file-like object without changing its cursor."""
    getvalue = getattr(upload, "getvalue", None)
    if callable(getvalue):
        contents = getvalue()
    else:
        read = getattr(upload, "read", None)
        if not callable(read):
            raise ValueError("Unsupported image input; expected an image path, bytes, array, or uploaded file.")

        try:
            position = upload.tell()
        except (AttributeError, OSError):
            position = None

        try:
            if position is not None:
                upload.seek(0)
            contents = read()
        except (OSError, ValueError) as exc:
            raise ValueError("Could not read the uploaded image.") from exc
        finally:
            if position is not None:
                try:
                    upload.seek(position)
                except (AttributeError, OSError, ValueError):
                    pass

    if not isinstance(contents, (bytes, bytearray, memoryview)):
        raise ValueError("Uploaded image contents must be bytes.")
    if not contents:
        raise ValueError("Image upload is empty.")
    return bytes(contents)


def _decode_bytes(contents: bytes) -> NDArray[np.uint8]:
    if not contents:
        raise ValueError("Image upload is empty.")

    encoded = np.frombuffer(contents, dtype=np.uint8)
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError("Image bytes could not be decoded.") from exc
    if image is None:
        raise ValueError("Invalid or unsupported image file.")
    return image


def load_image_bgr(image_input: ImageInput) -> NDArray[np.uint8]:
    """Load a supported image input and normalize it to an 8-bit BGR array.

    NumPy arrays with one, three, or four channels are accepted; three-channel
    arrays are interpreted as BGR and four-channel arrays as BGRA.
    """
    if isinstance(image_input, np.ndarray):
        image = image_input
    elif isinstance(image_input, (bytes, bytearray, memoryview)):
        image = _decode_bytes(bytes(image_input))
    elif isinstance(image_input, (str, os.PathLike, Path)):
        path = os.fspath(image_input)
        if not path:
            raise ValueError("Image path cannot be empty.")
        try:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except cv2.error as exc:
            raise ValueError(f"Could not read image file: {path}") from exc
        if image is None:
            raise ValueError(f"Could not read a supported image file: {path}")
    else:
        image = _decode_bytes(_read_upload(image_input))

    if image.size == 0:
        raise ValueError("Image array is empty.")
    if image.dtype != np.uint8:
        raise ValueError("Images must use 8-bit unsigned integer pixel values (uint8).")
    if image.ndim == 2:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        except cv2.error as exc:
            raise ValueError("Grayscale image could not be converted to BGR.") from exc
    elif image.ndim == 3 and image.shape[2] == 1:
        try:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        except cv2.error as exc:
            raise ValueError("Grayscale image could not be converted to BGR.") from exc
    elif image.ndim == 3 and image.shape[2] == 4:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        except cv2.error as exc:
            raise ValueError("BGRA image could not be converted to BGR.") from exc
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Unsupported image dimensions; use a grayscale, BGR, or BGRA image."
        )

    if image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("Image must have non-zero height and width.")
    return np.ascontiguousarray(image)
