"""test_e2e_workflow.py

Comprehensive end-to-end integration test for SaplingProof AI:
1. Tests cv_engine against sample_data pairs:
   - Positive correspondence pair (ref_sapling_A vs followup_sapling_A_matched)
   - Different foliage pair (ref_sapling_A vs followup_sapling_B_distinct)
   - Completely non-foliage pair (ref_sapling_A vs non_sapling_brick)
2. Tests persistence workflow:
   - Registering a sapling
   - Storing CV comparison result as audit record
   - Reviewer decision flow
   - Plot summary aggregations
"""

import unittest
import os
import cv2
import tempfile
import storage
from cv_engine import (
    compare_images,
    MATCH_EVIDENCE_FOUND,
    NO_RELIABLE_MATCH,
    INSUFFICIENT_EVIDENCE
)

class TestE2EWorkflow(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        storage.DB_DIR = self.temp_dir.name
        storage.DB_PATH = os.path.join(self.temp_dir.name, "e2e_sapling.db")
        storage.IMAGES_DIR = os.path.join(self.temp_dir.name, "e2e_images")
        storage.init_db()

        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
        self.ref_img = cv2.imread(os.path.join(sample_dir, "ref_sapling_A.png"))
        self.fol_match = cv2.imread(os.path.join(sample_dir, "followup_sapling_A_matched.png"))
        self.fol_distinct = cv2.imread(os.path.join(sample_dir, "followup_sapling_B_distinct.png"))
        self.fol_brick = cv2.imread(os.path.join(sample_dir, "non_sapling_brick.png"))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sample_image_cv_evaluations(self):
        self.assertIsNotNone(self.ref_img)
        self.assertIsNotNone(self.fol_match)
        self.assertIsNotNone(self.fol_distinct)
        self.assertIsNotNone(self.fol_brick)

        # 1. Matched pair should establish correspondence
        res_matched = compare_images(self.ref_img, self.fol_match)
        self.assertEqual(res_matched["status"], MATCH_EVIDENCE_FOUND)
        self.assertGreaterEqual(res_matched["inliers_count"], 10)
        self.assertGreaterEqual(res_matched["inlier_ratio"], 0.25)
        self.assertFalse(res_matched["requires_human_review"])

        # 2. Distinct sapling should fail correspondence
        res_distinct = compare_images(self.ref_img, self.fol_distinct)
        self.assertIn(res_distinct["status"], [NO_RELIABLE_MATCH, INSUFFICIENT_EVIDENCE])
        self.assertTrue(res_distinct["requires_human_review"])

        # 3. Brick wall vs sapling should fail correspondence
        res_brick = compare_images(self.ref_img, self.fol_brick)
        self.assertIn(res_brick["status"], [NO_RELIABLE_MATCH, INSUFFICIENT_EVIDENCE])
        self.assertTrue(res_brick["requires_human_review"])

    def test_full_pipeline_persistence(self):
        # Register
        sapling_data = {
            "id": "SPL-E2E-TEST-99",
            "species": "Dipterocarpus indicus",
            "planting_date": "2024-02-01",
            "plot_name": "Agumbe Rainforest Sector",
            "latitude": 13.518,
            "longitude": 75.093,
            "field_officer": "R. Sharma",
            "campaign": "Canopy Restoration 2024",
        }
        ok, msg = storage.register_sapling(sapling_data)
        self.assertTrue(ok)

        # Run comparison and save audit
        res = compare_images(self.ref_img, self.fol_match)
        audit_data = {
            "sapling_id": "SPL-E2E-TEST-99",
            "audit_date": "2024-04-15",
            "status": res["status"],
            "requires_human_review": res["requires_human_review"],
            "inliers_count": res["inliers_count"],
            "good_matches_count": res["good_matches_count"],
            "inlier_ratio": res["inlier_ratio"],
            "ref_keypoints_count": res["ref_keypoints_count"],
            "fol_keypoints_count": res["fol_keypoints_count"],
            "ref_blur_metric": res["ref_blur_metric"],
            "fol_blur_metric": res["fol_blur_metric"],
            "explanation": res["explanation"],
        }
        ok_audit, audit_id = storage.save_audit(audit_data)
        self.assertTrue(ok_audit)

        # Check retrieval
        audits = storage.get_all_audits("SPL-E2E-TEST-99")
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["status"], MATCH_EVIDENCE_FOUND)

        # Review decision
        ok_dec, _ = storage.update_audit_decision(audit_id, "VERIFIED", "Approved by field supervisor", "R. Sharma")
        self.assertTrue(ok_dec)

        # Check KPI
        kpi = storage.get_kpi_summary()
        self.assertEqual(kpi["total_saplings"], 1)
        self.assertEqual(kpi["matched_audits"], 1)


if __name__ == "__main__":
    unittest.main()
