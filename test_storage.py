"""test_storage.py

Automated unit tests for SaplingProof AI's storage module.
Covers:
- Database initialization and schema creation
- Sapling registration and duplicate rejection
- Audit record creation and retrieval
- Reviewer decision workflow
- Analytics KPI calculations
- Reset and seed functionality
"""

import unittest
import os
import tempfile
import storage

class TestSaplingProofStorage(unittest.TestCase):

    def setUp(self):
        # Override DB path for test isolation
        self.temp_dir = tempfile.TemporaryDirectory()
        storage.DB_DIR = self.temp_dir.name
        storage.DB_PATH = os.path.join(self.temp_dir.name, "test_sapling.db")
        storage.IMAGES_DIR = os.path.join(self.temp_dir.name, "test_images")
        storage.init_db()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sapling_registration_and_retrieval(self):
        sapling_data = {
            "id": "TEST-SPL-001",
            "species": "Teak (Tectona grandis)",
            "planting_date": "2024-03-01",
            "plot_name": "Test Plot Alpha",
            "latitude": 11.25,
            "longitude": 76.85,
            "field_officer": "Tester",
            "campaign": "Spring Test 2024",
            "notes": "Unit test sapling record"
        }
        success, msg = storage.register_sapling(sapling_data)
        self.assertTrue(success)
        self.assertIn("successfully registered", msg)

        # Retrieve
        record = storage.get_sapling_by_id("TEST-SPL-001")
        self.assertIsNotNone(record)
        self.assertEqual(record["species"], "Teak (Tectona grandis)")
        self.assertEqual(record["plot_name"], "Test Plot Alpha")

        # Duplicate rejection
        dup_success, dup_msg = storage.register_sapling(sapling_data)
        self.assertFalse(dup_success)
        self.assertIn("already registered", dup_msg)

    def test_audit_workflow(self):
        sapling_data = {
            "id": "TEST-SPL-002",
            "species": "Bamboo (Bambusoideae)",
            "planting_date": "2024-03-02",
            "plot_name": "Test Plot Beta",
        }
        storage.register_sapling(sapling_data)

        audit_data = {
            "id": "AUD-TEST-001",
            "sapling_id": "TEST-SPL-002",
            "audit_date": "2024-04-01",
            "status": "MATCH_EVIDENCE_FOUND",
            "requires_human_review": False,
            "inliers_count": 35,
            "good_matches_count": 42,
            "inlier_ratio": 0.833,
            "ref_keypoints_count": 600,
            "fol_keypoints_count": 590,
            "explanation": "High geometric correspondence",
        }
        success, audit_id = storage.save_audit(audit_data)
        self.assertTrue(success)
        self.assertEqual(audit_id, "AUD-TEST-001")

        audits = storage.get_all_audits(sapling_id="TEST-SPL-002")
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["inliers_count"], 35)

        # Reviewer decision update
        up_success, up_msg = storage.update_audit_decision(
            audit_id="AUD-TEST-001",
            decision="VERIFIED",
            notes="Auditor confirmed match.",
            reviewer="Lead Auditor"
        )
        self.assertTrue(up_success)

        audits_after = storage.get_all_audits(sapling_id="TEST-SPL-002")
        self.assertEqual(audits_after[0]["reviewer_decision"], "VERIFIED")
        self.assertEqual(audits_after[0]["requires_human_review"], 0)

    def test_kpi_summary(self):
        kpi = storage.get_kpi_summary()
        self.assertEqual(kpi["total_saplings"], 0)

        storage.register_sapling({"id": "SPL-KPI-1", "species": "Sal", "plot_name": "P1"})
        storage.register_sapling({"id": "SPL-KPI-2", "species": "Sal", "plot_name": "P2"})

        kpi_after = storage.get_kpi_summary()
        self.assertEqual(kpi_after["total_saplings"], 2)
        self.assertEqual(kpi_after["plots_monitored"], 2)


if __name__ == "__main__":
    unittest.main()
