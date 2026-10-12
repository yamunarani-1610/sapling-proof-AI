# Test Report: SaplingProof AI Verification Suite

**Date of Execution:** 2026-10-12  
**Platform:** Windows 11 / Python 3.11  
**Test Suite:** `unittest discover -v`  
**Status:** **12 Passed / 0 Failed (100% Success)**

---

## 1. Computer Vision Engine (`test_cv.py`)

| Test Case | Objective | Result |
|---|---|---|
| `test_self_comparison` | Evaluates identical image correspondence. Verifies `inliers_count > 10` and `MATCH_EVIDENCE_FOUND`. | **PASSED** |
| `test_different_canopies_low_inlier_ratio` | Compares two distinct foliage images with different random seeds. | **PASSED** |
| `test_different_distinct_images` | Compares canopy against brick wall. Verifies fallback to `NO_RELIABLE_MATCH`. | **PASSED** |
| `test_blurred_image_handling` | Injects Gaussian blur (kernel 45x45) and checks Laplacian variance detection and human review flag. | **PASSED** |
| `test_invalid_and_blank_inputs` | Tests empty arrays, `None` inputs, and blank solid color frames. Verifies graceful error codes. | **PASSED** |
| `test_output_structure_consistency` | Asserts output schema immutability, data types, and required metric keys. | **PASSED** |
| `test_disclaimer_presence` | Verifies presence of mandatory scientific correspondence notice. | **PASSED** |

---

## 2. Persistence & Storage Engine (`test_storage.py`)

| Test Case | Objective | Result |
|---|---|---|
| `test_sapling_registration_and_retrieval` | Registers sapling, retrieves by ID, asserts field integrity, rejects duplicate IDs. | **PASSED** |
| `test_audit_workflow` | Records comparison result, updates reviewer decision (`VERIFIED`), checks review flag toggle. | **PASSED** |
| `test_kpi_summary` | Verifies aggregation math for registered saplings, plots, audits, and flagged counts. | **PASSED** |

---

## 3. End-to-End Pipeline (`test_e2e_workflow.py`)

| Test Case | Objective | Result |
|---|---|---|
| `test_sample_image_cv_evaluations` | Evaluates real benchmark files (`ref_sapling_A.png`, `followup_sapling_A_matched.png`, `followup_sapling_B_distinct.png`, `non_sapling_brick.png`). Asserts expected verdicts. | **PASSED** |
| `test_full_pipeline_persistence` | Runs registration -> CV comparison -> audit recording -> supervisory approval -> KPI calculation. | **PASSED** |

---

## 4. Summary & Verification Notes
All 12 automated unit and integration tests executed cleanly without regressions. The application safely isolates SQLite transactions and image assets without depending on volatile external services.
