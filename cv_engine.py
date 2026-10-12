"""ORB-based visual feature comparison for sapling photographs.

The returned metrics describe image-feature correspondences only. They do not
prove biological identity, sapling identity, or survival.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

from cv_utils import ImageInput, load_image_bgr

ResultStatus = Literal[
    "VISUAL_MATCH_EVIDENCE_FOUND",
    "INSUFFICIENT_VISUAL_EVIDENCE",
    "MANUAL_REVIEW_REQUIRED",
]

_MIN_GEOMETRIC_INLIER_RATIO = 0.5


class ComparisonResult(TypedDict):
    result_status: ResultStatus
    manual_review: bool
    reason: str
    keypoints_image1: int
    keypoints_image2: int
    good_matches: int
    inlier_matches: int
    inlier_ratio: float
    match_visualization: NDArray[np.uint8] | None


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _make_result(
    status: ResultStatus,
    reason: str,
    keypoints_image1: int = 0,
    keypoints_image2: int = 0,
    good_matches: int = 0,
    inlier_matches: int = 0,
    inlier_ratio: float = 0.0,
    visualization: NDArray[np.uint8] | None = None,
) -> ComparisonResult:
    return {
        "result_status": status,
        "manual_review": status == "MANUAL_REVIEW_REQUIRED",
        "reason": reason,
        "keypoints_image1": int(keypoints_image1),
        "keypoints_image2": int(keypoints_image2),
        "good_matches": int(good_matches),
        "inlier_matches": int(inlier_matches),
        "inlier_ratio": float(inlier_ratio),
        "match_visualization": visualization,
    }


def _visualize_inliers(
    image1: NDArray[np.uint8],
    keypoints1: list[cv2.KeyPoint],
    image2: NDArray[np.uint8],
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    inlier_mask: NDArray[np.uint8],
) -> NDArray[np.uint8] | None:
    try:
        mask = inlier_mask.ravel().astype(np.uint8).tolist()
        return cv2.drawMatches(
            image1,
            keypoints1,
            image2,
            keypoints2,
            matches,
            None,
            matchColor=(0, 220, 0),
            singlePointColor=(90, 90, 90),
            matchesMask=mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    except cv2.error:
        return None


def compare_images(
    image1: ImageInput,
    image2: ImageInput,
    min_good_matches: int = 8,
    min_inliers: int = 6,
    ratio: float = 0.75,
    max_features: int = 1500,
) -> ComparisonResult:
    """Compare two images using ORB, Hamming matching, and RANSAC.

    Thresholds are conservative demo defaults, not scientifically validated
    criteria. A ``VISUAL_MATCH_EVIDENCE_FOUND`` result means only that enough
    geometrically consistent visual features were found to merit consideration.
    """
    _validate_positive_integer(min_good_matches, "min_good_matches")
    _validate_positive_integer(min_inliers, "min_inliers")
    _validate_positive_integer(max_features, "max_features")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
        raise ValueError("ratio must be a number strictly between 0 and 1.")

    bgr1 = load_image_bgr(image1)
    bgr2 = load_image_bgr(image2)
    try:
        gray1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=max_features)
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    except cv2.error as exc:
        return _make_result(
            "MANUAL_REVIEW_REQUIRED",
            f"OpenCV could not process the images: {exc}",
        )

    count1 = len(keypoints1)
    count2 = len(keypoints2)
    if descriptors1 is None or descriptors2 is None or count1 == 0 or count2 == 0:
        return _make_result(
            "INSUFFICIENT_VISUAL_EVIDENCE",
            "One or both images have no detectable ORB descriptors; there is insufficient automated evidence.",
            count1,
            count2,
        )

    try:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        candidate_pairs = matcher.knnMatch(descriptors1, descriptors2, k=2)
    except cv2.error as exc:
        return _make_result(
            "MANUAL_REVIEW_REQUIRED",
            f"OpenCV could not match image descriptors: {exc}",
            count1,
            count2,
        )

    good_matches = [
        pair[0]
        for pair in candidate_pairs
        if len(pair) == 2 and pair[0].distance < float(ratio) * pair[1].distance
    ]
    good_count = len(good_matches)
    if good_count < min_good_matches:
        return _make_result(
            "INSUFFICIENT_VISUAL_EVIDENCE",
            f"Only {good_count} good descriptor matches were found; at least "
            f"{min_good_matches} are required for geometric verification.",
            count1,
            count2,
            good_count,
        )
    if good_count < 4:
        return _make_result(
            "INSUFFICIENT_VISUAL_EVIDENCE",
            "At least four correspondences are required to estimate a homography.",
            count1,
            count2,
            good_count,
        )

    points1 = np.float32(
        [keypoints1[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    points2 = np.float32(
        [keypoints2[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    try:
        _homography, mask = cv2.findHomography(
            points1, points2, cv2.RANSAC, 5.0
        )
    except cv2.error as exc:
        return _make_result(
            "MANUAL_REVIEW_REQUIRED",
            f"RANSAC homography estimation failed: {exc}",
            count1,
            count2,
            good_count,
        )

    if mask is None:
        return _make_result(
            "MANUAL_REVIEW_REQUIRED",
            "RANSAC could not establish consistent image geometry; manual review is required.",
            count1,
            count2,
            good_count,
        )

    inlier_count = int(np.count_nonzero(mask))
    inlier_ratio = inlier_count / good_count
    visualization = _visualize_inliers(
        bgr1, keypoints1, bgr2, keypoints2, good_matches, mask
    )
    if (
        inlier_count < min_inliers
        or inlier_ratio < _MIN_GEOMETRIC_INLIER_RATIO
    ):
        return _make_result(
            "MANUAL_REVIEW_REQUIRED",
            "Some candidate correspondences were found, but geometrically consistent "
            "evidence is below the demo threshold; manual review is required.",
            count1,
            count2,
            good_count,
            inlier_count,
            inlier_ratio,
            visualization,
        )

    return _make_result(
        "VISUAL_MATCH_EVIDENCE_FOUND",
        "Geometrically consistent visual features were found. This is not proof of "
        "sapling identity, biological identity, or survival.",
        count1,
        count2,
        good_count,
        inlier_count,
        inlier_ratio,
        visualization,
    )


__all__ = ["ComparisonResult", "compare_images"]
