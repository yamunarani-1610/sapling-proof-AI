from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest

from cv_engine import compare_images


def _textured_image(seed: int = 21, width: int = 360, height: int = 280) -> np.ndarray:
    """Build a reproducible image with corners and local visual structure."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(55):
        center = (
            int(rng.integers(8, width - 8)),
            int(rng.integers(8, height - 8)),
        )
        color = tuple(int(channel) for channel in rng.integers(30, 256, size=3))
        radius = int(rng.integers(3, 14))
        cv2.circle(image, center, radius, color, thickness=-1)
    for _ in range(18):
        start = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )
        end = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )
        color = tuple(int(channel) for channel in rng.integers(60, 256, size=3))
        cv2.line(image, start, end, color, thickness=2)
    return image


def _translated_image(image: np.ndarray, dx: int = 7, dy: int = 5) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )


def test_same_image_finds_geometric_visual_evidence() -> None:
    image = _textured_image()
    result = compare_images(image, image.copy())

    assert result["result_status"] == "VISUAL_MATCH_EVIDENCE_FOUND"
    assert result["manual_review"] is False
    assert result["good_matches"] >= 8
    assert result["inlier_matches"] >= 6
    assert result["inlier_ratio"] >= 0.5


def test_slight_translation_finds_geometric_visual_evidence() -> None:
    image = _textured_image(seed=32)
    result = compare_images(image, _translated_image(image))

    assert result["result_status"] == "VISUAL_MATCH_EVIDENCE_FOUND"
    assert result["inlier_matches"] >= 6
    assert result["inlier_ratio"] >= 0.5


def test_unrelated_images_do_not_report_match_evidence() -> None:
    result = compare_images(_textured_image(seed=4), _textured_image(seed=900))

    assert result["result_status"] != "VISUAL_MATCH_EVIDENCE_FOUND"
    assert result["manual_review"] is True or result["good_matches"] < 8
    assert result["reason"]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"not an image", "Invalid or unsupported"),
        (b"", "empty"),
    ],
)
def test_invalid_or_empty_image_bytes_raise_value_error(
    contents: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_images(contents, _textured_image())


def test_low_texture_images_report_insufficient_evidence() -> None:
    low_texture = np.full((128, 128), 120, dtype=np.uint8)
    result = compare_images(low_texture, low_texture.copy())

    assert result["result_status"] == "INSUFFICIENT_VISUAL_EVIDENCE"
    assert result["keypoints_image1"] == 0
    assert result["keypoints_image2"] == 0
    assert result["good_matches"] == 0
    assert result["match_visualization"] is None


def test_invalid_ratio_configuration_raises_value_error() -> None:
    with pytest.raises(ValueError, match="ratio"):
        compare_images(_textured_image(), _textured_image(), ratio=1.0)


def test_grayscale_bgr_bgra_and_uploaded_file_inputs_are_supported() -> None:
    image = _textured_image()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    uploaded = BytesIO()
    success, encoded = cv2.imencode(".png", image)
    assert success
    uploaded.write(encoded.tobytes())

    for candidate in (gray, image, bgra, uploaded):
        result = compare_images(candidate, image)
        assert result["keypoints_image1"] > 0
        assert result["result_status"] == "VISUAL_MATCH_EVIDENCE_FOUND"


def test_encoded_bytes_and_image_path_are_supported(tmp_path) -> None:
    image = _textured_image()
    success, encoded = cv2.imencode(".png", image)
    assert success
    image_path = tmp_path / "sapling.png"
    assert cv2.imwrite(str(image_path), image)

    for candidate in (encoded.tobytes(), image_path):
        result = compare_images(candidate, image)
        assert result["result_status"] == "VISUAL_MATCH_EVIDENCE_FOUND"


def test_invalid_array_dimensions_raise_value_error() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        compare_images(np.zeros((30, 30, 2), dtype=np.uint8), _textured_image())


def test_result_shape_types_and_visualization_are_stable() -> None:
    image = _textured_image()
    result = compare_images(image, image)

    assert set(result) == {
        "result_status",
        "manual_review",
        "reason",
        "keypoints_image1",
        "keypoints_image2",
        "good_matches",
        "inlier_matches",
        "inlier_ratio",
        "match_visualization",
    }
    assert isinstance(result["manual_review"], bool)
    assert isinstance(result["reason"], str)
    for key in (
        "keypoints_image1",
        "keypoints_image2",
        "good_matches",
        "inlier_matches",
    ):
        assert isinstance(result[key], int)
    assert isinstance(result["inlier_ratio"], float)
    visualization = result["match_visualization"]
    assert isinstance(visualization, np.ndarray)
    assert visualization.dtype == np.uint8
    assert visualization.ndim == 3 and visualization.shape[2] == 3
    rgb_visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
    assert rgb_visualization.shape == visualization.shape


def test_insufficient_match_threshold_returns_insufficient_evidence() -> None:
    image = _textured_image()
    result = compare_images(image, image.copy(), min_good_matches=5000)

    assert result["result_status"] == "INSUFFICIENT_VISUAL_EVIDENCE"
    assert 0 < result["good_matches"] < 5000
    assert result["manual_review"] is False
