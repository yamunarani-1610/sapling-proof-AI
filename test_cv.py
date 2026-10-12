"""test_cv.py

Automated unit tests for SaplingProof AI's cv_engine.
Covers:
- Comparing an image with itself
- Comparing different images
- Blank and invalid inputs
- Blurred images
- Insufficient feature matches
- Stable output structure and error handling
"""

import unittest
import numpy as np
import cv2

from cv_engine import (
    compare_images,
    MATCH_EVIDENCE_FOUND,
    INSUFFICIENT_EVIDENCE,
    NO_RELIABLE_MATCH,
    INVALID_IMAGE,
)


def _generate_synthetic_leaf_canopy(seed: int = 42) -> np.ndarray:
    """Generates synthetic foliage pattern with distinctive organic shapes."""
    np.random.seed(seed)
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    for _ in range(45):
        center = (int(np.random.randint(50, 350)), int(np.random.randint(50, 350)))
        axes = (int(np.random.randint(15, 35)), int(np.random.randint(8, 18)))
        angle = int(np.random.randint(0, 180))
        color = (
            int(np.random.randint(20, 60)),
            int(np.random.randint(120, 240)),
            int(np.random.randint(20, 60)),
        )
        cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
        cv2.ellipse(img, center, axes, angle, 0, 360, (10, 80, 10), 1)

    # Stem structure
    cv2.line(img, (200, 400), (200, 150), (30, 70, 140), 5)
    return img


def _generate_synthetic_brick_wall() -> np.ndarray:
    """Generates a synthetic brick pattern (completely distinct visual class)."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    for y in range(0, 400, 30):
        shift = 20 if (y // 30) % 2 == 1 else 0
        for x in range(-20, 400, 40):
            cv2.rectangle(img, (x + shift, y), (x + shift + 36, y + 26), (60, 70, 180), -1)
            cv2.rectangle(img, (x + shift, y), (x + shift + 36, y + 26), (200, 200, 200), 1)
    return img


class TestSaplingProofCVEngine(unittest.TestCase):

    def setUp(self):
        self.canopy_1 = _generate_synthetic_leaf_canopy(seed=42)
        self.canopy_2 = _generate_synthetic_leaf_canopy(seed=999)
        self.brick_wall = _generate_synthetic_brick_wall()

    def test_output_structure_consistency(self):
        """Test that the output dictionary always contains expected keys and types."""
        result = compare_images(self.canopy_1, self.canopy_1)
        expected_keys = {
            "status",
            "explanation",
            "requires_human_review",
            "ref_keypoints_count",
            "fol_keypoints_count",
            "good_matches_count",
            "inliers_count",
            "inlier_ratio",
            "ref_blur_metric",
            "fol_blur_metric",
            "visualization_image",
            "disclaimer",
        }
        self.assertTrue(expected_keys.issubset(result.keys()))
        self.assertIsInstance(result["status"], str)
        self.assertIsInstance(result["explanation"], str)
        self.assertIsInstance(result["requires_human_review"], bool)
        self.assertIsInstance(result["inliers_count"], int)
        self.assertIsInstance(result["inlier_ratio"], float)

    def test_self_comparison(self):
        """Test comparing an image with itself yields match evidence."""
        result = compare_images(self.canopy_1, self.canopy_1)
        self.assertEqual(result["status"], MATCH_EVIDENCE_FOUND)
        self.assertFalse(result["requires_human_review"])
        self.assertGreater(result["ref_keypoints_count"], 15)
        self.assertGreater(result["good_matches_count"], 10)
        self.assertGreater(result["inliers_count"], 10)
        self.assertIsNotNone(result["visualization_image"])

    def test_different_distinct_images(self):
        """Test comparing two completely distinct scenes yields NO_RELIABLE_MATCH or INSUFFICIENT_EVIDENCE."""
        result = compare_images(self.canopy_1, self.brick_wall)
        self.assertIn(result["status"], [NO_RELIABLE_MATCH, INSUFFICIENT_EVIDENCE])
        self.assertTrue(result["requires_human_review"])

    def test_different_canopies_low_inlier_ratio(self):
        """Test comparing two distinct randomly-seeded canopies does not pass geometric consensus."""
        result = compare_images(self.canopy_1, self.canopy_2)
        self.assertIn(result["status"], [NO_RELIABLE_MATCH, INSUFFICIENT_EVIDENCE])
        self.assertTrue(result["requires_human_review"])

    def test_invalid_and_blank_inputs(self):
        """Test handling of None, empty array, or zero dimension inputs."""
        # None inputs
        res_none = compare_images(None, self.canopy_1)
        self.assertEqual(res_none["status"], INVALID_IMAGE)
        self.assertTrue(res_none["requires_human_review"])

        # Empty array
        empty_arr = np.array([], dtype=np.uint8)
        res_empty = compare_images(empty_arr, self.canopy_1)
        self.assertEqual(res_empty["status"], INVALID_IMAGE)

        # Blank/solid color image (no features)
        solid_white = np.ones((300, 300, 3), dtype=np.uint8) * 255
        res_solid = compare_images(solid_white, self.canopy_1)
        self.assertEqual(res_solid["status"], INSUFFICIENT_EVIDENCE)
        self.assertTrue(res_solid["requires_human_review"])

    def test_blurred_image_handling(self):
        """Test that severely blurred images trigger blur warning or insufficient evidence."""
        blurred_img = cv2.GaussianBlur(self.canopy_1, (45, 45), 0)
        result = compare_images(blurred_img, self.canopy_1)

        self.assertTrue(result["requires_human_review"])
        self.assertIn(result["status"], [INSUFFICIENT_EVIDENCE, NO_RELIABLE_MATCH, MATCH_EVIDENCE_FOUND])
        self.assertLess(result["ref_blur_metric"], result["fol_blur_metric"])

    def test_disclaimer_presence(self):
        """Test that output contains standard scientific/technical disclaimer."""
        result = compare_images(self.canopy_1, self.canopy_1)
        self.assertIn("disclaimer", result)
        self.assertIn("correspondences", result["disclaimer"].lower())


if __name__ == "__main__":
    unittest.main()
