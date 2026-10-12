"""storage.py

Robust SQLite & filesystem persistence engine for SaplingProof AI.
Manages:
- Sapling registry records (ID, species, planting date, plot, GPS lat/lon, campaign, notes, baseline photo path)
- Audit records (sapling reference, audit date, photo paths, ORB keypoints, RANSAC inliers, inlier ratio,
  blur metrics, verification status, human review flag, reviewer notes, decision)
- Plot & Site metadata
- Safe local image storage under `data/images/`
- Demo records seeding and database reset utilities
"""

import os
import sqlite3
import datetime
import shutil
import uuid
from typing import Dict, List, Any, Optional, Tuple
import json

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "sapling_proof.db")
IMAGES_DIR = os.path.join(DB_DIR, "images")


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with dict-like row access."""
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Saplings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS saplings (
        id TEXT PRIMARY KEY,
        species TEXT NOT NULL,
        planting_date TEXT NOT NULL,
        plot_name TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        field_officer TEXT,
        campaign TEXT,
        baseline_photo_path TEXT,
        notes TEXT,
        status TEXT DEFAULT 'ACTIVE',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    # Audits table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audits (
        id TEXT PRIMARY KEY,
        sapling_id TEXT NOT NULL,
        audit_date TEXT NOT NULL,
        reference_photo_path TEXT,
        followup_photo_path TEXT,
        vis_photo_path TEXT,
        status TEXT NOT NULL,
        requires_human_review INTEGER NOT NULL,
        inliers_count INTEGER DEFAULT 0,
        good_matches_count INTEGER DEFAULT 0,
        inlier_ratio REAL DEFAULT 0.0,
        ref_keypoints_count INTEGER DEFAULT 0,
        fol_keypoints_count INTEGER DEFAULT 0,
        ref_blur_metric REAL DEFAULT 0.0,
        fol_blur_metric REAL DEFAULT 0.0,
        explanation TEXT,
        reviewer_decision TEXT DEFAULT 'PENDING',
        reviewer_notes TEXT,
        reviewed_by TEXT,
        reviewed_at TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (sapling_id) REFERENCES saplings (id) ON DELETE CASCADE
    );
    """)

    # Sites / Plots table for quick geo aggregation
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS plots (
        name TEXT PRIMARY KEY,
        region TEXT,
        target_saplings INTEGER DEFAULT 100,
        latitude REAL,
        longitude REAL,
        notes TEXT
    );
    """)

    conn.commit()
    conn.close()


def save_image_file(image_bytes: bytes, filename_prefix: str, ext: str = "png") -> str:
    """Saves image bytes locally and returns relative path from project root."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    clean_prefix = "".join(c for c in filename_prefix if c.isalnum() or c in ("-", "_"))
    unique_filename = f"{clean_prefix}_{uuid.uuid4().hex[:8]}.{ext.lstrip('.')}"
    full_path = os.path.join(IMAGES_DIR, unique_filename)
    with open(full_path, "wb") as f:
        f.write(image_bytes)
    return os.path.relpath(full_path, os.path.dirname(os.path.abspath(__file__)))


# -------------------------------------------------------------
# Sapling CRUD
# -------------------------------------------------------------

def register_sapling(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Registers a new sapling. Returns (success, message)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    sapling_id = data.get("id", "").strip()

    if not sapling_id:
        conn.close()
        return False, "Sapling ID cannot be empty."

    cursor.execute("SELECT id FROM saplings WHERE id = ?", (sapling_id,))
    if cursor.fetchone():
        conn.close()
        return False, f"Sapling ID '{sapling_id}' is already registered."

    now_iso = datetime.datetime.now().isoformat()
    try:
        cursor.execute("""
        INSERT INTO saplings (
            id, species, planting_date, plot_name, latitude, longitude,
            field_officer, campaign, baseline_photo_path, notes, status,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sapling_id,
            data.get("species", "Unknown / Mixed Native"),
            str(data.get("planting_date", datetime.date.today())),
            data.get("plot_name", "Plot Alpha"),
            data.get("latitude"),
            data.get("longitude"),
            data.get("field_officer", "Field Agent"),
            data.get("campaign", "2024 Afforestation Drive"),
            data.get("baseline_photo_path"),
            data.get("notes", ""),
            data.get("status", "ACTIVE"),
            now_iso,
            now_iso,
        ))

        # Check if plot exists, insert if not
        plot_name = data.get("plot_name", "Plot Alpha")
        cursor.execute("SELECT name FROM plots WHERE name = ?", (plot_name,))
        if not cursor.fetchone() and data.get("latitude") and data.get("longitude"):
            cursor.execute("""
            INSERT OR IGNORE INTO plots (name, region, target_saplings, latitude, longitude)
            VALUES (?, ?, ?, ?, ?)
            """, (plot_name, "Western Ghats Eco-Corridor", 250, data.get("latitude"), data.get("longitude")))

        conn.commit()
        conn.close()
        return True, f"Sapling '{sapling_id}' successfully registered."
    except Exception as e:
        conn.close()
        return False, f"Database error: {str(e)}"


def get_all_saplings() -> List[Dict[str, Any]]:
    """Fetches all sapling records ordered by newest first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM saplings ORDER BY created_at DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_sapling_by_id(sapling_id: str) -> Optional[Dict[str, Any]]:
    """Fetches single sapling record by primary key."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM saplings WHERE id = ?", (sapling_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_sapling(sapling_id: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
    """Updates fields on an existing sapling record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.datetime.now().isoformat()
    fields = []
    values = []
    for k, v in updates.items():
        if k in ("species", "planting_date", "plot_name", "latitude", "longitude",
                 "field_officer", "campaign", "baseline_photo_path", "notes", "status"):
            fields.append(f"{k} = ?")
            values.append(v)
    if not fields:
        conn.close()
        return False, "No valid fields provided for update."

    fields.append("updated_at = ?")
    values.append(now_iso)
    values.append(sapling_id)

    try:
        cursor.execute(f"UPDATE saplings SET {', '.join(fields)} WHERE id = ?", tuple(values))
        conn.commit()
        conn.close()
        return True, f"Sapling {sapling_id} updated successfully."
    except Exception as e:
        conn.close()
        return False, f"Failed to update sapling: {str(e)}"


def delete_sapling(sapling_id: str) -> Tuple[bool, str]:
    """Deletes a sapling record and its associated audits."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM saplings WHERE id = ?", (sapling_id,))
        cursor.execute("DELETE FROM audits WHERE sapling_id = ?", (sapling_id,))
        conn.commit()
        conn.close()
        return True, f"Sapling {sapling_id} and associated audit records removed."
    except Exception as e:
        conn.close()
        return False, f"Deletion failed: {str(e)}"


# -------------------------------------------------------------
# Audit CRUD
# -------------------------------------------------------------

def save_audit(audit_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Saves a new CV comparison audit record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    audit_id = audit_data.get("id") or f"AUD-{datetime.date.today().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    now_iso = datetime.datetime.now().isoformat()

    try:
        cursor.execute("""
        INSERT INTO audits (
            id, sapling_id, audit_date, reference_photo_path, followup_photo_path,
            vis_photo_path, status, requires_human_review, inliers_count,
            good_matches_count, inlier_ratio, ref_keypoints_count, fol_keypoints_count,
            ref_blur_metric, fol_blur_metric, explanation, reviewer_decision,
            reviewer_notes, reviewed_by, reviewed_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            audit_id,
            audit_data.get("sapling_id"),
            str(audit_data.get("audit_date", datetime.date.today())),
            audit_data.get("reference_photo_path"),
            audit_data.get("followup_photo_path"),
            audit_data.get("vis_photo_path"),
            audit_data.get("status"),
            1 if audit_data.get("requires_human_review") else 0,
            audit_data.get("inliers_count", 0),
            audit_data.get("good_matches_count", 0),
            audit_data.get("inlier_ratio", 0.0),
            audit_data.get("ref_keypoints_count", 0),
            audit_data.get("fol_keypoints_count", 0),
            audit_data.get("ref_blur_metric", 0.0),
            audit_data.get("fol_blur_metric", 0.0),
            audit_data.get("explanation", ""),
            audit_data.get("reviewer_decision", "PENDING"),
            audit_data.get("reviewer_notes", ""),
            audit_data.get("reviewed_by", ""),
            audit_data.get("reviewed_at", ""),
            now_iso,
        ))
        conn.commit()
        conn.close()
        return True, audit_id
    except Exception as e:
        conn.close()
        return False, f"Failed to record audit: {str(e)}"


def get_all_audits(sapling_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetches audit records, optionally filtered by sapling ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if sapling_id:
        cursor.execute("SELECT * FROM audits WHERE sapling_id = ? ORDER BY audit_date DESC, created_at DESC", (sapling_id,))
    else:
        cursor.execute("SELECT a.*, s.species, s.plot_name, s.campaign FROM audits a LEFT JOIN saplings s ON a.sapling_id = s.id ORDER BY a.created_at DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def update_audit_decision(audit_id: str, decision: str, notes: str, reviewer: str) -> Tuple[bool, str]:
    """Updates the reviewer decision on an audit record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.datetime.now().isoformat()
    try:
        cursor.execute("""
        UPDATE audits
        SET reviewer_decision = ?, reviewer_notes = ?, reviewed_by = ?, reviewed_at = ?, requires_human_review = ?
        WHERE id = ?
        """, (decision, notes, reviewer, now_iso, 0 if decision in ("VERIFIED", "REJECTED") else 1, audit_id))
        conn.commit()
        conn.close()
        return True, f"Audit {audit_id} marked as {decision}."
    except Exception as e:
        conn.close()
        return False, f"Update failed: {str(e)}"


# -------------------------------------------------------------
# Analytics & Plots Aggregations
# -------------------------------------------------------------

def get_kpi_summary() -> Dict[str, Any]:
    """Computes high-level KPI metrics for the Command Center."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM saplings")
    total_saplings = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT plot_name) FROM saplings")
    plots_monitored = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM audits")
    audits_completed = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM audits WHERE requires_human_review = 1 OR reviewer_decision = 'PENDING'")
    requiring_review = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM audits WHERE status = 'MATCH_EVIDENCE_FOUND'")
    matched_audits = cursor.fetchone()[0]

    conn.close()
    return {
        "total_saplings": total_saplings,
        "plots_monitored": plots_monitored,
        "audits_completed": audits_completed,
        "requiring_review": requiring_review,
        "matched_audits": matched_audits,
    }


def get_plots_summary() -> List[Dict[str, Any]]:
    """Aggregates plot-level statistics with coordinates for mapping."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT 
        s.plot_name,
        AVG(s.latitude) as avg_lat,
        AVG(s.longitude) as avg_lon,
        COUNT(s.id) as sapling_count,
        COUNT(a.id) as audit_count,
        SUM(CASE WHEN a.requires_human_review = 1 THEN 1 ELSE 0 END) as review_count
    FROM saplings s
    LEFT JOIN audits a ON s.id = a.sapling_id
    GROUP BY s.plot_name
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# -------------------------------------------------------------
# Demo Data Seeder & Reset
# -------------------------------------------------------------

def seed_demo_data(force: bool = False) -> None:
    """Seeds realistic field demo records if database is empty or forced."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM saplings")
    count = cursor.fetchone()[0]

    if count > 0 and not force:
        conn.close()
        return

    if force:
        cursor.execute("DELETE FROM audits")
        cursor.execute("DELETE FROM saplings")
        cursor.execute("DELETE FROM plots")
        conn.commit()

    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
    ref_img_a = os.path.join(sample_dir, "ref_sapling_A.png")
    fol_img_a = os.path.join(sample_dir, "followup_sapling_A_matched.png")
    fol_img_b = os.path.join(sample_dir, "followup_sapling_B_distinct.png")

    demo_saplings = [
        {
            "id": "SPL-2024-NIL-01",
            "species": "Syzygium cumini (Jamun)",
            "planting_date": "2024-01-15",
            "plot_name": "Nilgiris Shola Sector A",
            "latitude": 11.4102,
            "longitude": 76.6950,
            "field_officer": "R. Ramanathan",
            "campaign": "Western Ghats Corridor Restoration",
            "baseline_photo_path": "sample_data/ref_sapling_A.png",
            "notes": "Native riparian canopy restoration. High moisture soil index.",
            "status": "ACTIVE",
        },
        {
            "id": "SPL-2024-NIL-02",
            "species": "Terminalia arjuna (Arjun)",
            "planting_date": "2024-01-18",
            "plot_name": "Nilgiris Shola Sector A",
            "latitude": 11.4125,
            "longitude": 76.6972,
            "field_officer": "R. Ramanathan",
            "campaign": "Western Ghats Corridor Restoration",
            "baseline_photo_path": "sample_data/ref_sapling_A.png",
            "notes": "Edge buffer zone along ridge.",
            "status": "ACTIVE",
        },
        {
            "id": "SPL-2024-WAY-05",
            "species": "Dalbergia latifolia (Rosewood)",
            "planting_date": "2024-02-02",
            "plot_name": "Wayanad Highland Buffer",
            "latitude": 11.6854,
            "longitude": 76.1320,
            "field_officer": "A. Nair",
            "campaign": "Highland Afforestation 2024",
            "baseline_photo_path": "sample_data/ref_sapling_A.png",
            "notes": "Sloped terrain, monitored for soil runoff stability.",
            "status": "ACTIVE",
        },
        {
            "id": "SPL-2024-ANM-12",
            "species": "Artocarpus heterophyllus (Wild Jack)",
            "planting_date": "2024-02-14",
            "plot_name": "Anamalai Reserve Block 3",
            "latitude": 10.4280,
            "longitude": 76.9820,
            "field_officer": "K. Sundaram",
            "campaign": "Biodiversity Hotspot Enrich",
            "baseline_photo_path": "sample_data/ref_sapling_A.png",
            "notes": "Canopy gap enrichment planting.",
            "status": "ACTIVE",
        },
        {
            "id": "SPL-2024-KOD-08",
            "species": "Michelia champaca (Champak)",
            "planting_date": "2024-03-01",
            "plot_name": "Kodaikanal Shola Valley",
            "latitude": 10.2381,
            "longitude": 77.4892,
            "field_officer": "M. Devadas",
            "campaign": "Palani Hills Reforestation",
            "baseline_photo_path": "sample_data/ref_sapling_A.png",
            "notes": "Shade-loving sub-canopy tree.",
            "status": "ACTIVE",
        },
    ]

    for s in demo_saplings:
        register_sapling(s)

    # Demo Audits showcasing different CV verdicts
    demo_audits = [
        {
            "id": "AUD-20240410-001",
            "sapling_id": "SPL-2024-NIL-01",
            "audit_date": "2024-04-10",
            "reference_photo_path": "sample_data/ref_sapling_A.png",
            "followup_photo_path": "sample_data/followup_sapling_A_matched.png",
            "vis_photo_path": "sample_data/followup_sapling_A_matched.png",
            "status": "MATCH_EVIDENCE_FOUND",
            "requires_human_review": 0,
            "inliers_count": 48,
            "good_matches_count": 62,
            "inlier_ratio": 0.774,
            "ref_keypoints_count": 820,
            "fol_keypoints_count": 790,
            "ref_blur_metric": 142.5,
            "fol_blur_metric": 138.2,
            "explanation": "Found 48 geometrically consistent inlier correspondences (77.4% inlier ratio) between the two scenes.",
            "reviewer_decision": "VERIFIED",
            "reviewer_notes": "Strong geometric consensus observed on structural stems and reference tag.",
            "reviewed_by": "Dr. V. Kurian (Lead Botanist)",
            "reviewed_at": "2024-04-12T10:30:00",
        },
        {
            "id": "AUD-20240505-002",
            "sapling_id": "SPL-2024-NIL-02",
            "audit_date": "2024-05-05",
            "reference_photo_path": "sample_data/ref_sapling_A.png",
            "followup_photo_path": "sample_data/followup_sapling_B_distinct.png",
            "vis_photo_path": "sample_data/followup_sapling_B_distinct.png",
            "status": "NO_RELIABLE_MATCH",
            "requires_human_review": 1,
            "inliers_count": 4,
            "good_matches_count": 18,
            "inlier_ratio": 0.222,
            "ref_keypoints_count": 820,
            "fol_keypoints_count": 650,
            "ref_blur_metric": 142.5,
            "fol_blur_metric": 115.0,
            "explanation": "Matches failed geometric consensus (4 RANSAC inliers from 18 candidate matches, 22.2% ratio). Insufficient consistent geometry.",
            "reviewer_decision": "PENDING",
            "reviewer_notes": "Camera angle shifted heavily; field officer dispatched for ground re-inspection.",
            "reviewed_by": "",
            "reviewed_at": "",
        },
        {
            "id": "AUD-20240518-003",
            "sapling_id": "SPL-2024-WAY-05",
            "audit_date": "2024-05-18",
            "reference_photo_path": "sample_data/ref_sapling_A.png",
            "followup_photo_path": "sample_data/ref_sapling_A.png",
            "vis_photo_path": "sample_data/ref_sapling_A.png",
            "status": "MATCH_EVIDENCE_FOUND",
            "requires_human_review": 0,
            "inliers_count": 145,
            "good_matches_count": 150,
            "inlier_ratio": 0.967,
            "ref_keypoints_count": 820,
            "fol_keypoints_count": 820,
            "ref_blur_metric": 142.5,
            "fol_blur_metric": 142.5,
            "explanation": "Found 145 geometrically consistent inlier correspondences (96.7% inlier ratio) between the two scenes.",
            "reviewer_decision": "VERIFIED",
            "reviewer_notes": "Baseline verification confirmation.",
            "reviewed_by": "R. Ramanathan",
            "reviewed_at": "2024-05-19T09:15:00",
        },
        {
            "id": "AUD-20240602-004",
            "sapling_id": "SPL-2024-ANM-12",
            "audit_date": "2024-06-02",
            "reference_photo_path": "sample_data/ref_sapling_A.png",
            "followup_photo_path": "sample_data/non_sapling_brick.png",
            "vis_photo_path": "sample_data/non_sapling_brick.png",
            "status": "NO_RELIABLE_MATCH",
            "requires_human_review": 1,
            "inliers_count": 1,
            "good_matches_count": 6,
            "inlier_ratio": 0.167,
            "ref_keypoints_count": 820,
            "fol_keypoints_count": 420,
            "ref_blur_metric": 142.5,
            "fol_blur_metric": 210.0,
            "explanation": "Found only 6 good feature matches, which is below the minimum threshold (10) required for geometric consistency verification.",
            "reviewer_decision": "REJECTED",
            "reviewer_notes": "Incorrect follow-up photo uploaded (brick wall infrastructure). Re-audit requested.",
            "reviewed_by": "K. Sundaram",
            "reviewed_at": "2024-06-03T14:20:00",
        },
    ]

    for a in demo_audits:
        save_audit(a)

    conn.close()


def reset_database() -> None:
    """Completely resets database and image store, reinitializing clean schemas."""
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception:
            pass
    init_db()
