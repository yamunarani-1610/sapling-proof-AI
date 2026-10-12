"""cv_engine.py

OpenCV feature-matching engine for SaplingProof AI.
Analyzes visual correspondences between reference and follow-up images using ORB,
Hamming-distance ratio test, and RANSAC geometric verification.

Important Disclaimer:
Matches report measurable visual feature correspondences only.
They DO NOT prove physical identity, tree survival, or sapling health.
"""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

# Status Constants
MATCH_EVIDENCE_FOUND = "MATCH_EVIDENCE_FOUND"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
NO_RELIABLE_MATCH = "NO_RELIABLE_MATCH"
INVALID_IMAGE = "INVALID_IMAGE"

# Tunable thresholds
MIN_KEYPOINTS = 15
MIN_GOOD_MATCHES = 10
MIN_RANSAC_INLIERS = 10
MIN_INLIER_RATIO = 0.25
BLUR_LAPLACIAN_VAR_THRESH = 40.0
RATIO_TEST_THRESH = 0.75


def _is_image_blurred(gray: np.ndarray, threshold: float = BLUR_LAPLACIAN_VAR_THRESH) -> Tuple[bool, float]:
    """Computes the Laplacian variance to estimate image sharpness."""
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return (var < threshold, var)


def _safe_convert_grayscale(img: np.ndarray) -> Optional[np.ndarray]:
    """Safely converts an image array to single-channel uint8 grayscale."""
    if img is None or not isinstance(img, np.ndarray):
        return None
    if img.size == 0 or img.ndim < 2:
        return None

    try:
        if img.ndim == 2:
            gray = img
        elif img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 1:
                gray = img[:, :, 0]
            else:
                return None
        else:
            return None

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray
    except Exception:
        return None


def compare_images(
    reference_image: np.ndarray,
    followup_image: np.ndarray,
    max_features: int = 1500,
) -> Dict[str, Any]:
    """Compares a reference image with a follow-up image using ORB and RANSAC.

    Parameters:
        reference_image: np.ndarray (OpenCV BGR/Grayscale image)
        followup_image: np.ndarray (OpenCV BGR/Grayscale image)
        max_features: Maximum number of keypoints to extract via ORB.

    Returns:
        Dictionary containing:
            - status (str): MATCH_EVIDENCE_FOUND, INSUFFICIENT_EVIDENCE,
                            NO_RELIABLE_MATCH, or INVALID_IMAGE.
            - explanation (str): Human-readable interpretation.
            - requires_human_review (bool): Flag indicating manual review is warranted.
            - ref_keypoints_count (int)
            - fol_keypoints_count (int)
            - good_matches_count (int)
            - inliers_count (int)
            - inlier_ratio (float)
            - ref_blur_metric (float)
            - fol_blur_metric (float)
            - visualization_image (np.ndarray or None): RGB/BGR image with drawn matches.
            - disclaimer (str): Standard legal / technical disclaimer.
    """
    disclaimer_text = (
        "Notice: Feature correspondences represent shared visual patterns (textures/edges) "
        "and do NOT certify botanical identity, continuous sapling survival, or plant health."
    )

    result_template: Dict[str, Any] = {
        "status": INVALID_IMAGE,
        "explanation": "Image validation failed.",
        "requires_human_review": True,
        "ref_keypoints_count": 0,
        "fol_keypoints_count": 0,
        "good_matches_count": 0,
        "inliers_count": 0,
        "inlier_ratio": 0.0,
        "ref_blur_metric": 0.0,
        "fol_blur_metric": 0.0,
        "visualization_image": None,
        "disclaimer": disclaimer_text,
    }

    ref_gray = _safe_convert_grayscale(reference_image)
    fol_gray = _safe_convert_grayscale(followup_image)

    if ref_gray is None or fol_gray is None:
        result_template["explanation"] = "One or both input images are invalid, empty, or corrupted."
        return result_template

    # Measure blur / sharpness
    is_ref_blurred, ref_blur_val = _is_image_blurred(ref_gray)
    is_fol_blurred, fol_blur_val = _is_image_blurred(fol_gray)
    result_template["ref_blur_metric"] = round(ref_blur_val, 2)
    result_template["fol_blur_metric"] = round(fol_blur_val, 2)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=15)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_fol, des_fol = orb.detectAndCompute(fol_gray, None)

    count_kp_ref = len(kp_ref) if kp_ref is not None else 0
    count_kp_fol = len(kp_fol) if kp_fol is not None else 0
    result_template["ref_keypoints_count"] = count_kp_ref
    result_template["fol_keypoints_count"] = count_kp_fol

    if count_kp_ref < MIN_KEYPOINTS or count_kp_fol < MIN_KEYPOINTS:
        result_template["status"] = INSUFFICIENT_EVIDENCE
        blur_note = ""
        if is_ref_blurred or is_fol_blurred:
            blur_note = " Low sharpness or excessive blur was detected."
        result_template["explanation"] = (
            f"Insufficient detectable texture features (Ref: {count_kp_ref}, Follow-up: {count_kp_fol})."
            f"{blur_note} Cannot conduct reliable feature matching."
        )
        return result_template

    if des_ref is None or des_fol is None or len(des_ref) < 2 or len(des_fol) < 2:
        result_template["status"] = INSUFFICIENT_EVIDENCE
        result_template["explanation"] = "Descriptors could not be extracted reliably from one or both images."
        return result_template

    # Match descriptors using BFMatcher with NORM_HAMMING and k=2 for ratio test
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw_matches = matcher.knnMatch(des_ref, des_fol, k=2)
    except Exception as e:
        result_template["status"] = NO_RELIABLE_MATCH
        result_template["explanation"] = f"Error during feature matching: {str(e)}"
        return result_template

    # Lowe's ratio test
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_TEST_THRESH * n.distance:
                good_matches.append(m)

    result_template["good_matches_count"] = len(good_matches)

    # Ensure color images are prepared for drawing
    vis_ref = reference_image if reference_image.ndim == 3 else cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)
    vis_fol = followup_image if followup_image.ndim == 3 else cv2.cvtColor(fol_gray, cv2.COLOR_GRAY2BGR)

    if len(good_matches) < MIN_GOOD_MATCHES:
        result_template["status"] = NO_RELIABLE_MATCH
        result_template["explanation"] = (
            f"Found only {len(good_matches)} good feature matches, which is below the minimum "
            f"threshold ({MIN_GOOD_MATCHES}) required for geometric consistency verification."
        )
        # Still render matches if any exist
        if good_matches:
            try:
                vis_img = cv2.drawMatches(
                    vis_ref, kp_ref, vis_fol, kp_fol, good_matches, None,
                    matchColor=(0, 165, 255),
                    singlePointColor=(100, 100, 100),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                result_template["visualization_image"] = vis_img
            except Exception:
                pass
        return result_template

    # Geometric verification using RANSAC Homography
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_fol[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is not None:
        inliers_mask = mask.ravel().tolist()
        inliers_count = int(sum(inliers_mask))
    else:
        inliers_mask = [0] * len(good_matches)
        inliers_count = 0

    inlier_ratio = round(inliers_count / max(len(good_matches), 1), 3)
    result_template["inliers_count"] = inliers_count
    result_template["inlier_ratio"] = inlier_ratio

    # Render match visualization
    draw_params = dict(
        matchColor=(0, 255, 0),  # Inliers in green
        singlePointColor=(100, 100, 100),
        matchesMask=inliers_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    try:
        vis_img = cv2.drawMatches(vis_ref, kp_ref, vis_fol, kp_fol, good_matches, None, **draw_params)
        result_template["visualization_image"] = vis_img
    except Exception:
        result_template["visualization_image"] = None

    # Determine final verdict
    if inliers_count >= MIN_RANSAC_INLIERS and inlier_ratio >= MIN_INLIER_RATIO:
        result_template["status"] = MATCH_EVIDENCE_FOUND
        result_template["requires_human_review"] = False
        result_template["explanation"] = (
            f"Found {inliers_count} geometrically consistent inlier correspondences "
            f"({round(inlier_ratio * 100, 1)}% inlier ratio) between the two scenes."
        )
    else:
        result_template["status"] = NO_RELIABLE_MATCH
        result_template["requires_human_review"] = True
        result_template["explanation"] = (
            f"Matches failed geometric consensus ({inliers_count} RANSAC inliers from "
            f"{len(good_matches)} preliminary candidate matches, {round(inlier_ratio * 100, 1)}% ratio). "
            f"Insufficient consistent geometry."
        )

    # Flag for human review if images had blur issues even if geometric matches were found
    if is_ref_blurred or is_fol_blurred:
        result_template["requires_human_review"] = True
        result_template["explanation"] += " Note: Image blur detected, manual verification strongly recommended."

    return result_template
