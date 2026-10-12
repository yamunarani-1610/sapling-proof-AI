"""app.py

SaplingProof AI — Research-Driven Environmental Monitoring Platform
Synthesizes design principles from:
- Linear: Dense, purposeful typography, subtle borders, high contrast, clean sidebar navigation.
- Stripe: Robust form controls, filterable data tables, clear feedback, and structured workflows.
- ArcGIS Field Maps: Real GPS coordinate pins, location telemetry, field-audit inspection workflows.

Pages:
1. Overview (Real-time telemetry, active queue, monitored plots, quick actions)
2. New Registration (Validation, auto-ID suggestion, image preview, duplicate prevention)
3. Visual Evidence Lab (Side-by-side comparison, ORB/RANSAC consensus, sharpness diagnostics, permanent audit saving)
4. Sapling Registry (Table & detail view, search, filters, inline edit, delete, CSV export)
5. Audit History (Saved comparisons, filterable timeline, reviewer decisions, notes, CSV export)
6. Sites & Map (Folium geospatial plots, real coordinate markers, plot summaries)
7. Analytics & Reports (Outcome distributions, review share, campaign statistics, CSV exports)
8. Settings & Data (Demo seeding, clean database reset, threshold reference)
"""

import io
import os
import csv
import datetime
from PIL import Image
import numpy as np
import cv2
import folium
import streamlit as st
import streamlit.components.v1 as components_v1

import storage
from cv_engine import (
    compare_images,
    MATCH_EVIDENCE_FOUND,
    INSUFFICIENT_EVIDENCE,
    NO_RELIABLE_MATCH,
    INVALID_IMAGE,
)
from components import (
    inject_forest_intelligence_theme,
    render_header,
    render_scientific_guardrail,
    render_kpi,
    render_status_pill,
)

# Initialize storage and schema
storage.init_db()

# Page Configuration
st.set_page_config(
    page_title="SaplingProof AI — Forest Intelligence Platform",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Forest Intelligence Theme
inject_forest_intelligence_theme()


def load_image_to_cv2(uploaded_file_or_path) -> np.ndarray:
    """Reads a Streamlit uploaded file or local filepath into an OpenCV BGR numpy array."""
    if uploaded_file_or_path is None:
        return None
    try:
        if isinstance(uploaded_file_or_path, str):
            if os.path.exists(uploaded_file_or_path):
                return cv2.imread(uploaded_file_or_path)
            return None
        file_bytes = np.asarray(bytearray(uploaded_file_or_path.read()), dtype=np.uint8)
        uploaded_file_or_path.seek(0)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# Navigation Management with session_state support
NAV_PAGES = [
    "Overview",
    "New Registration",
    "Visual Evidence Lab",
    "Sapling Registry",
    "Audit History",
    "Sites & Map",
    "Analytics & Reports",
    "Settings & Data",
]

if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Overview"

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem; padding-bottom: 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.12);">
            <span style="font-size: 1.8rem;">🌲</span>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">SaplingProof AI</h3>
                <span style="font-size: 0.7rem; color: #A7F3D0; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase;">Forest Intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Navigation Radio with index tracking
    current_index = NAV_PAGES.index(st.session_state["active_nav"]) if st.session_state["active_nav"] in NAV_PAGES else 0
    selected_page = st.radio(
        "Navigation",
        NAV_PAGES,
        index=current_index,
        label_visibility="collapsed",
        key="sidebar_nav_radio",
    )
    if selected_page != st.session_state["active_nav"]:
        st.session_state["active_nav"] = selected_page
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='font-size: 0.72rem; color: #82D1A3; text-transform: uppercase; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.8px;'>Active Monitoring Campaign</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.86rem; color: #FFFFFF; font-weight: 600; margin: 0;'>Western Ghats Corridor Restoration 2024</p>", unsafe_allow_html=True)
    st.caption("Forestry Department & NGO Consortium")

    st.markdown("---")
    kpis = storage.get_kpi_summary()
    st.caption(f"📊 Database: {kpis['total_saplings']} Saplings | {kpis['audits_completed']} Audits")
    if kpis["total_saplings"] == 0:
        if st.button("🌱 Load Sample Demo Data", key="sidebar_quick_seed", use_container_width=True):
            storage.seed_demo_data(force=True)
            st.rerun()

    st.caption("Engine: OpenCV ORB + RANSAC | Python 3.11")


nav_selection = st.session_state["active_nav"]


# ==============================================================================
# PAGE A: OVERVIEW / COMMAND CENTER
# ==============================================================================
if nav_selection == "Overview":
    render_header(
        title="Forest Intelligence Command Center",
        description="Operational afforestation monitoring, photographic correspondence auditing, and field inspection governance.",
        badge_text="LIVE FIELD TELEMETRY",
    )

    render_scientific_guardrail()

    # KPI Summary Row
    kpi_data = storage.get_kpi_summary()
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Registered Saplings", kpi_data["total_saplings"], "Baseline profiles on record", "🌱")
    with k2:
        render_kpi("Monitored Plots", kpi_data["plots_monitored"], "Active geographical sectors", "🗺️")
    with k3:
        render_kpi("Audits Completed", kpi_data["audits_completed"], f"{kpi_data['matched_audits']} verified geometric matches", "🔍")
    with k4:
        render_kpi("Review Required", kpi_data["requiring_review"], "Awaiting on-site inspection", "⚠️")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Operational Quick Actions
    st.markdown("#### ⚡ Operational Quick Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("🌱 Register Sapling", use_container_width=True, type="primary"):
            st.session_state["active_nav"] = "New Registration"
            st.rerun()
    with qa2:
        if st.button("🔬 New Photo Comparison", use_container_width=True):
            st.session_state["active_nav"] = "Visual Evidence Lab"
            st.rerun()
    with qa3:
        if st.button("📋 View Audit Records", use_container_width=True):
            st.session_state["active_nav"] = "Audit History"
            st.rerun()
    with qa4:
        if st.button("🗺️ Open Geospatial Map", use_container_width=True):
            st.session_state["active_nav"] = "Sites & Map"
            st.rerun()

    st.markdown("---")

    # Recent Verification Audits & Sector Breakdown
    col_audits, col_sectors = st.columns([3, 2])

    with col_audits:
        st.markdown("#### 📋 Recent Verification Audits")
        recent_audits = storage.get_all_audits()[:5]

        if recent_audits:
            for audit in recent_audits:
                status_pill = render_status_pill(audit["status"])
                dec = audit.get("reviewer_decision", "PENDING")
                badge_class = f"fp-badge-{dec.lower()}"

                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                        <div>
                            <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">
                                {audit['sapling_id']} <span style="font-size: 0.78rem; font-weight: 500; color: #64748B; margin-left: 8px;">Date: {audit['audit_date']}</span>
                            </div>
                            <div style="font-size: 0.82rem; color: #475569; margin-top: 3px;">
                                Inliers: <strong>{audit['inliers_count']}</strong> &nbsp;|&nbsp; Inlier Share: <strong>{round(audit['inlier_ratio']*100, 1)}%</strong> &nbsp;|&nbsp; Matches: <strong>{audit['good_matches_count']}</strong>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            {status_pill}
                            <div style="margin-top: 4px;"><span class="{badge_class}">{dec}</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No audit comparisons recorded yet. Use the Visual Evidence Lab or seed sample data from Settings.")

    with col_sectors:
        st.markdown("#### 📍 Active Sectors Overview")
        plots = storage.get_plots_summary()
        if plots:
            for p in plots:
                review_color = "#DC2626" if p["review_count"] > 0 else "#059669"
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; color: #0F172A; font-size: 0.92rem;">📍 {p['plot_name']}</span>
                            <span style="font-size: 0.78rem; background: #ECFDF5; color: #065F46; font-weight: 700; padding: 2px 7px; border-radius: 4px;">{p['sapling_count']} Saplings</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B; margin-top: 5px;">
                            Audits: <strong>{p['audit_count']}</strong> &nbsp;|&nbsp; Review Queue: <strong style="color: {review_color};">{p['review_count']}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No plot sectors configured yet.")


# ==============================================================================
# PAGE B: NEW SAPLING REGISTRATION
# ==============================================================================
elif nav_selection == "New Registration":
    render_header(
        title="Sapling Baseline Registration",
        description="Intake form for newly planted saplings to record baseline botanical identity, plot geolocation, and photographic references.",
        badge_text="FIELD REGISTRATION WORKFLOW",
    )

    all_current = storage.get_all_saplings()
    suggested_id = f"SPL-2024-{datetime.date.today().strftime('%m%d')}-{len(all_current) + 1:03d}"

    with st.container():
        st.markdown("#### 📝 Field Intake Form")
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            reg_id = st.text_input("Sapling Unique ID *", value=suggested_id, help="Must be unique across all afforestation campaigns.")
            reg_species = st.text_input("Botanical & Vernacular Species *", value="Syzygium cumini (Jamun)", help="Scientific name with vernacular reference.")
            reg_date = st.date_input("Planting Date *", value=datetime.date.today())
            reg_plot = st.text_input("Plot / Sector Name *", value="Nilgiris Shola Sector A", help="Afforestation micro-plot.")
            reg_campaign = st.text_input("Planting Campaign", value="Western Ghats Corridor Restoration 2024")

        with col_f2:
            col_lat, col_lon = st.columns(2)
            with col_lat:
                reg_lat = st.number_input("GPS Latitude (WGS84)", value=11.4102, format="%.5f")
            with col_lon:
                reg_lon = st.number_input("GPS Longitude (WGS84)", value=76.6950, format="%.5f")

            reg_officer = st.text_input("Field Officer / Observer", value="Officer Ramanathan")
            reg_notes = st.text_area("Field Notes & Habitat Conditions", value="Planted along riparian buffer. Organic coir support installed.")

        st.markdown("#### 📸 Baseline Reference Photo")
        reg_photo = st.file_uploader(
            "Upload high-resolution baseline reference photo (JPEG/PNG/WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            key="reg_uploader",
        )

        baseline_path = None
        if reg_photo:
            st.image(reg_photo, caption="Baseline Photo Preview", width=340)
            photo_bytes = reg_photo.read()
            baseline_path = storage.save_image_file(photo_bytes, filename_prefix=f"baseline_{reg_id}")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        col_submit, col_cancel = st.columns([1, 4])
        with col_submit:
            submit_btn = st.button("🌲 Register Sapling", type="primary", use_container_width=True)

        if submit_btn:
            if not reg_id.strip():
                st.error("Validation Error: Sapling ID is required.")
            elif not reg_species.strip():
                st.error("Validation Error: Botanical species is required.")
            elif not reg_plot.strip():
                st.error("Validation Error: Plot / Sector Name is required.")
            else:
                success, msg = storage.register_sapling({
                    "id": reg_id.strip(),
                    "species": reg_species.strip(),
                    "planting_date": str(reg_date),
                    "plot_name": reg_plot.strip(),
                    "latitude": reg_lat,
                    "longitude": reg_lon,
                    "field_officer": reg_officer.strip(),
                    "campaign": reg_campaign.strip(),
                    "baseline_photo_path": baseline_path,
                    "notes": reg_notes.strip(),
                })
                if success:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ Registration Blocked: {msg}")


# ==============================================================================
# PAGE C: VISUAL EVIDENCE LAB
# ==============================================================================
elif nav_selection == "Visual Evidence Lab":
    render_header(
        title="Visual Evidence & Correspondence Lab",
        description="OpenCV computer-vision engine extracting ORB keypoints, Lowe's ratio test descriptors, and RANSAC homography geometric consensus.",
        badge_text="OPENCV GEOMETRIC CONSENSUS",
    )

    render_scientific_guardrail()

    all_saplings = storage.get_all_saplings()
    sapling_options = ["-- Unlinked Manual Audit --"] + [f"{s['id']} — {s['species']} ({s['plot_name']})" for s in all_saplings]

    col_target, col_params = st.columns([2, 1])
    with col_target:
        selected_sapling_str = st.selectbox("Target Sapling Record:", sapling_options)
    with col_params:
        max_features = st.slider("ORB Max Feature Budget", min_value=500, max_value=3000, value=1500, step=250)

    selected_sapling_id = None
    preloaded_ref_path = None
    if selected_sapling_str != "-- Unlinked Manual Audit --":
        selected_sapling_id = selected_sapling_str.split(" — ")[0]
        s_rec = storage.get_sapling_by_id(selected_sapling_id)
        if s_rec and s_rec.get("baseline_photo_path"):
            preloaded_ref_path = s_rec["baseline_photo_path"]

    # Image Upload / Selection Columns
    col_ref, col_fol = st.columns(2)

    with col_ref:
        st.markdown(
            """
            <div class="fp-lab-card">
                <div class="fp-lab-card-header">
                    <h4 class="fp-lab-card-title">📸 1. Baseline Evidence</h4>
                    <span class="fp-mono">REFERENCE</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        use_sample_ref = st.checkbox("Load Sample Baseline (ref_sapling_A.png)", value=(preloaded_ref_path is None and not all_saplings), key="cb_ref_sample")
        ref_source = None
        if use_sample_ref:
            ref_source = "sample_data/ref_sapling_A.png"
            if os.path.exists(ref_source):
                st.image(ref_source, caption="Sample: ref_sapling_A.png", use_container_width=True)
            else:
                st.warning("Sample baseline file not found.")
        elif preloaded_ref_path and os.path.exists(preloaded_ref_path):
            ref_source = preloaded_ref_path
            st.image(ref_source, caption=f"Loaded from {selected_sapling_id}", use_container_width=True)
        else:
            uploaded_ref = st.file_uploader("Select Baseline Image", type=["png", "jpg", "jpeg", "webp"], key="lab_ref_up")
            if uploaded_ref:
                st.image(uploaded_ref, caption="Uploaded Baseline Photo", use_container_width=True)
                ref_source = uploaded_ref
        st.markdown("</div>", unsafe_allow_html=True)

    with col_fol:
        st.markdown(
            """
            <div class="fp-lab-card">
                <div class="fp-lab-card-header">
                    <h4 class="fp-lab-card-title">📸 2. Follow-up Audit</h4>
                    <span class="fp-mono">INSPECTION</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        sample_fol_choice = st.selectbox(
            "Demo Follow-up Scenario:",
            ["Custom File Upload", "Sample Matched (followup_sapling_A_matched.png)", "Sample Distinct (followup_sapling_B_distinct.png)", "Sample Non-Foliage (non_sapling_brick.png)"],
            key="sb_fol_choice",
        )
        fol_source = None
        if sample_fol_choice == "Sample Matched (followup_sapling_A_matched.png)":
            fol_source = "sample_data/followup_sapling_A_matched.png"
            st.image(fol_source, caption="Sample Positive Matched Foliage", use_container_width=True)
        elif sample_fol_choice == "Sample Distinct (followup_sapling_B_distinct.png)":
            fol_source = "sample_data/followup_sapling_B_distinct.png"
            st.image(fol_source, caption="Sample Distinct Sapling (Negative Control)", use_container_width=True)
        elif sample_fol_choice == "Sample Non-Foliage (non_sapling_brick.png)":
            fol_source = "sample_data/non_sapling_brick.png"
            st.image(fol_source, caption="Sample Non-Foliage Scene (Negative Control)", use_container_width=True)
        else:
            uploaded_fol = st.file_uploader("Select Follow-up Audit Image", type=["png", "jpg", "jpeg", "webp"], key="lab_fol_up")
            if uploaded_fol:
                st.image(uploaded_fol, caption="Uploaded Follow-up Photo", use_container_width=True)
                fol_source = uploaded_fol
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    can_execute_cv = bool(ref_source and fol_source)

    exec_col, info_col = st.columns([1, 3])
    with exec_col:
        run_compare_btn = st.button("🔬 Execute Comparison", type="primary", disabled=not can_execute_cv, use_container_width=True)

    with info_col:
        if not can_execute_cv:
            st.caption("Provide both baseline and follow-up images to enable computer vision analysis.")

    if run_compare_btn and can_execute_cv:
        with st.spinner("Decoding images, detecting ORB keypoints, running Lowe's ratio test, and computing RANSAC homography consensus..."):
            cv_ref = load_image_to_cv2(ref_source)
            cv_fol = load_image_to_cv2(fol_source)

            if cv_ref is None or cv_fol is None:
                st.error("Failed to decode image buffers. Ensure files are valid images.")
            else:
                results = compare_images(cv_ref, cv_fol, max_features=max_features)
                st.session_state["cv_output"] = results
                st.session_state["cv_ref_img"] = cv_ref
                st.session_state["cv_fol_img"] = cv_fol
                st.session_state["cv_ref_src"] = ref_source
                st.session_state["cv_fol_src"] = fol_source
                st.session_state["cv_sapling_target"] = selected_sapling_id or f"SPL-UNLINKED-{datetime.date.today().strftime('%Y%m%d')}"

    # Display Results
    if "cv_output" in st.session_state:
        res = st.session_state["cv_output"]
        status = res.get("status", INVALID_IMAGE)
        inliers = res.get("inliers_count", 0)
        good_matches = res.get("good_matches_count", 0)
        inlier_ratio = res.get("inlier_ratio", 0.0)
        ref_kps = res.get("ref_keypoints_count", 0)
        fol_kps = res.get("fol_keypoints_count", 0)
        ref_blur = res.get("ref_blur_metric", 0.0)
        fol_blur = res.get("fol_blur_metric", 0.0)
        explanation = res.get("explanation", "")
        requires_review = res.get("requires_human_review", True)

        st.markdown("### 📊 Verification Analysis Verdict")
        st.markdown(render_status_pill(status), unsafe_allow_html=True)

        if requires_review:
            st.warning(f"**Manual Field Review Warranted:** {explanation}")
        else:
            st.success(f"**Geometric Consistency Established:** {explanation}")

        # Metrics Bar
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            render_kpi("RANSAC Inliers", inliers, "Geometric consensus", "📐")
        with m2:
            render_kpi("Good Matches", good_matches, "Lowe's ratio test", "🔗")
        with m3:
            render_kpi("Inlier Share", f"{round(inlier_ratio*100, 1)}%", "Inliers / Good matches", "📈")
        with m4:
            render_kpi("Keypoints", f"{ref_kps} / {fol_kps}", "Ref / Fol extracted", "✨")
        with m5:
            render_kpi("Review Status", "FLAGGED" if requires_review else "CLEARED", "Botanical audit", "🛡️")

        # Feature correspondence visualization
        vis_img = res.get("visualization_image")
        if vis_img is not None:
            st.markdown("#### 🔍 Feature Correspondence Visualization")
            vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            st.image(
                vis_rgb,
                caption="Green lines denote inliers satisfying RANSAC homography constraints. Orange lines indicate candidate matches.",
                use_container_width=True,
            )

        # Quality diagnostics & blur
        with st.expander("🔬 Optical Quality Diagnostics", expanded=False):
            qd1, qd2 = st.columns(2)
            with qd1:
                st.write(f"**Reference Laplacian Variance:** `{ref_blur}`")
                st.caption("✅ Sharpness acceptable" if ref_blur >= 40.0 else "⚠️ Image blur detected (<40.0)")
            with qd2:
                st.write(f"**Follow-up Laplacian Variance:** `{fol_blur}`")
                st.caption("✅ Sharpness acceptable" if fol_blur >= 40.0 else "⚠️ Image blur detected (<40.0)")

        # Record in Permanent Audit Trail
        st.markdown("#### 💾 Save to Permanent Audit Trail")
        s_col1, s_col2 = st.columns([2, 1])
        with s_col1:
            save_notes = st.text_input("Reviewer Notes & Context:", value=explanation, key="save_audit_notes")
        with s_col2:
            save_officer = st.text_input("Auditor / Observer Name:", value="Field Auditor", key="save_audit_officer")

        if st.button("📥 Save This Audit to Database", type="primary", use_container_width=True):
            target_id = st.session_state.get("cv_sapling_target")
            if not storage.get_sapling_by_id(target_id):
                storage.register_sapling({
                    "id": target_id,
                    "species": "Field Sample",
                    "planting_date": str(datetime.date.today()),
                    "plot_name": "Field Sector Unassigned",
                })

            vis_path = None
            if vis_img is not None:
                _, vis_enc = cv2.imencode(".png", vis_img)
                vis_path = storage.save_image_file(vis_enc.tobytes(), filename_prefix=f"vis_{target_id}")

            ok_aud, aud_id = storage.save_audit({
                "sapling_id": target_id,
                "audit_date": str(datetime.date.today()),
                "reference_photo_path": str(st.session_state.get("cv_ref_src")) if isinstance(st.session_state.get("cv_ref_src"), str) else None,
                "followup_photo_path": str(st.session_state.get("cv_fol_src")) if isinstance(st.session_state.get("cv_fol_src"), str) else None,
                "vis_photo_path": vis_path,
                "status": status,
                "requires_human_review": requires_review,
                "inliers_count": inliers,
                "good_matches_count": good_matches,
                "inlier_ratio": inlier_ratio,
                "ref_keypoints_count": ref_kps,
                "fol_keypoints_count": fol_kps,
                "ref_blur_metric": ref_blur,
                "fol_blur_metric": fol_blur,
                "explanation": explanation,
                "reviewer_decision": "PENDING" if requires_review else "VERIFIED",
                "reviewer_notes": save_notes,
                "reviewed_by": save_officer,
                "reviewed_at": datetime.datetime.now().isoformat(),
            })

            if ok_aud:
                st.success(f"✅ Audit successfully stored with permanent ID: `{aud_id}`!")
            else:
                st.error(f"❌ Failed to store audit: {aud_id}")


# ==============================================================================
# PAGE D: SAPLING REGISTRY
# ==============================================================================
elif nav_selection == "Sapling Registry":
    render_header(
        title="Sapling Inventory & Registry",
        description="Searchable, filterable repository of monitored saplings, baseline imagery, GPS coordinates, and inspection histories.",
        badge_text="INVENTORY GOVERNANCE",
    )

    all_saplings = storage.get_all_saplings()

    # Search and Filter Toolbar
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        query = st.text_input("🔍 Search by Sapling ID or Species Name:", "")
    with f2:
        plot_options = ["All Plots"] + sorted(list(set(s["plot_name"] for s in all_saplings))) if all_saplings else ["All Plots"]
        filter_plot = st.selectbox("Filter by Sector / Plot:", plot_options)
    with f3:
        filter_status = st.selectbox("Filter Status:", ["All Statuses", "ACTIVE", "FLAGGED", "ARCHIVED"])

    filtered = all_saplings
    if query.strip():
        q = query.strip().lower()
        filtered = [s for s in filtered if q in s["id"].lower() or q in s["species"].lower()]
    if filter_plot != "All Plots":
        filtered = [s for s in filtered if s["plot_name"] == filter_plot]
    if filter_status != "All Statuses":
        filtered = [s for s in filtered if s.get("status") == filter_status]

    st.markdown(f"**Displaying {len(filtered)} of {len(all_saplings)} registered saplings**")

    # CSV Export
    if filtered:
        csv_buf = io.StringIO()
        w = csv.writer(csv_buf)
        w.writerow(["ID", "Species", "Planting_Date", "Plot_Name", "Latitude", "Longitude", "Field_Officer", "Campaign", "Status"])
        for s in filtered:
            w.writerow([s["id"], s["species"], s["planting_date"], s["plot_name"], s.get("latitude"), s.get("longitude"), s.get("field_officer"), s.get("campaign"), s.get("status")])
        st.download_button(
            label="📥 Export Registry CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"sapling_registry_{datetime.date.today()}.csv",
            mime="text/csv",
        )

    st.markdown("---")

    if not filtered:
        st.info("No saplings match the current search filters.")
    else:
        for s in filtered:
            sap_audits = storage.get_all_audits(sapling_id=s["id"])
            audit_count = len(sap_audits)
            status_val = s.get("status", "ACTIVE")
            status_html = render_status_pill(status_val)

            # Record Card Container
            with st.container():
                st.markdown(
                    f"""
                    <div class="fp-registry-card">
                        <div class="fp-registry-header">
                            <div>
                                <div class="fp-registry-title">🌱 {s['id']} &nbsp;·&nbsp; {s['species']}</div>
                                <div class="fp-registry-subtitle">📍 <strong>{s['plot_name']}</strong> &nbsp;|&nbsp; Campaign: {s.get('campaign', 'Western Ghats 2024')}</div>
                            </div>
                            <div>
                                {status_html}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_img, col_info, col_ops = st.columns([1.1, 2.3, 1.2])

                with col_img:
                    b_path = s.get("baseline_photo_path")
                    if b_path and os.path.exists(b_path):
                        st.image(b_path, caption="Baseline Reference Photo", use_container_width=True)
                    else:
                        st.markdown(
                            """
                            <div style="background: #F1F5F9; height: 160px; border-radius: 8px; border: 1px dashed #CBD5E1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #64748B; font-size: 0.85rem; text-align: center; padding: 10px;">
                                <span style="font-size: 1.5rem; margin-bottom: 4px;">📷</span>
                                <strong>No Photo Attached</strong>
                                <span style="font-size: 0.75rem; color: #94A3B8;">Upload in Evidence Lab</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with col_info:
                    st.markdown(
                        f"""
                        <div class="fp-field-grid">
                            <div class="fp-field-item">
                                <span class="fp-field-label">Botanical Species</span>
                                <span class="fp-field-value">{s['species']}</span>
                            </div>
                            <div class="fp-field-item">
                                <span class="fp-field-label">Planting Date</span>
                                <span class="fp-field-value">{s['planting_date']}</span>
                            </div>
                            <div class="fp-field-item">
                                <span class="fp-field-label">Plot / Catchment Sector</span>
                                <span class="fp-field-value">{s['plot_name']}</span>
                            </div>
                            <div class="fp-field-item">
                                <span class="fp-field-label">Field Officer</span>
                                <span class="fp-field-value">{s.get('field_officer', 'Field Agent')}</span>
                            </div>
                            <div class="fp-field-item">
                                <span class="fp-field-label">GPS Geolocation</span>
                                <span class="fp-field-value"><span class="fp-mono">{f"{s['latitude']:.5f}, {s['longitude']:.5f}" if (s.get('latitude') and s.get('longitude')) else "Unassigned (Text Sector Only)"}</span></span>
                            </div>
                            <div class="fp-field-item">
                                <span class="fp-field-label">Inspection History</span>
                                <span class="fp-field-value"><strong>{audit_count}</strong> Verified Audits</span>
                            </div>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.85rem; color: #334155; line-height: 1.45; background: #F8FAF9; padding: 8px 12px; border-radius: 6px; border: 1px solid #E2E8F0;">
                            <strong style="color: #0F172A;">Field Observations:</strong> {s.get('notes') if s.get('notes') else "Standard planting protocol verified. No special anomalies reported."}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_ops:
                    st.markdown("<span class='fp-field-label'>Record Actions</span>", unsafe_allow_html=True)
                    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

                    with st.popover("✏️ Edit Record", use_container_width=True):
                        st.markdown(f"**Edit Sapling {s['id']}**")
                        edit_species = st.text_input("Botanical Species", value=s["species"], key=f"ed_sp_{s['id']}")
                        edit_plot = st.text_input("Plot / Sector", value=s["plot_name"], key=f"ed_pl_{s['id']}")
                        edit_notes = st.text_area("Field Notes", value=s.get("notes") or "", key=f"ed_nt_{s['id']}")
                        if st.button("Save Record Changes", key=f"btn_save_ed_{s['id']}", type="primary", use_container_width=True):
                            ok_up, msg_up = storage.update_sapling(s["id"], {"species": edit_species, "plot_name": edit_plot, "notes": edit_notes})
                            if ok_up:
                                st.success("Record updated successfully.")
                                st.rerun()
                            else:
                                st.error(msg_up)

                    with st.popover("🗑️ Delete Sapling", use_container_width=True):
                        st.markdown(f"⚠️ **Confirm Deletion of `{s['id']}`?**")
                        st.caption("This action is permanent and also purges associated audit records.")
                        if st.button("Confirm Permanent Deletion", key=f"btn_confirm_del_{s['id']}", type="primary", use_container_width=True):
                            ok_del, msg_del = storage.delete_sapling(s["id"])
                            if ok_del:
                                st.success(msg_del)
                                st.rerun()
                            else:
                                st.error(msg_del)

                st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)


# ==============================================================================
# PAGE E: AUDIT HISTORY & DECISIONS
# ==============================================================================
elif nav_selection == "Audit History":
    render_header(
        title="Audit History & Decision Workflow",
        description="Comprehensive inspection timeline linking photographic evidence, geometric verification outcomes, and reviewer decisions.",
        badge_text="AUDIT TRAIL & DECISION GOVERNANCE",
    )

    render_scientific_guardrail()

    all_audits = storage.get_all_audits()

    # Filter Controls
    af1, af2, af3 = st.columns(3)
    with af1:
        f_status = st.selectbox("Filter by CV Outcome:", ["All Outcomes", MATCH_EVIDENCE_FOUND, NO_RELIABLE_MATCH, INSUFFICIENT_EVIDENCE])
    with af2:
        f_dec = st.selectbox("Reviewer Decision:", ["All Decisions", "PENDING", "VERIFIED", "REJECTED"])
    with af3:
        f_review = st.selectbox("Ground Review Flag:", ["All", "Requires Review (FLAGGED)", "Cleared"])

    filtered_audits = all_audits
    if f_status != "All Outcomes":
        filtered_audits = [a for a in filtered_audits if a["status"] == f_status]
    if f_dec != "All Decisions":
        filtered_audits = [a for a in filtered_audits if a.get("reviewer_decision") == f_dec]
    if f_review == "Requires Review (FLAGGED)":
        filtered_audits = [a for a in filtered_audits if a.get("requires_human_review") == 1]
    elif f_review == "Cleared":
        filtered_audits = [a for a in filtered_audits if a.get("requires_human_review") == 0]

    st.markdown(f"**Displaying {len(filtered_audits)} of {len(all_audits)} recorded audits**")

    # CSV Export
    if filtered_audits:
        c_buf = io.StringIO()
        w = csv.writer(c_buf)
        w.writerow(["Audit_ID", "Sapling_ID", "Audit_Date", "Status", "Inliers", "Good_Matches", "Inlier_Ratio", "Decision", "Notes", "Reviewed_By"])
        for a in filtered_audits:
            w.writerow([a["id"], a["sapling_id"], a["audit_date"], a["status"], a.get("inliers_count"), a.get("good_matches_count"), a.get("inlier_ratio"), a.get("reviewer_decision"), a.get("reviewer_notes"), a.get("reviewed_by")])
        st.download_button(
            label="📥 Download Audit History CSV",
            data=c_buf.getvalue().encode("utf-8"),
            file_name=f"audit_history_{datetime.date.today()}.csv",
            mime="text/csv",
        )

    st.markdown("---")

    if not filtered_audits:
        st.info("No audits found matching the selected filters.")
    else:
        for a in filtered_audits:
            with st.expander(f"🔍 {a['id']} — {a['sapling_id']} ({a['audit_date']}) | Decision: {a.get('reviewer_decision', 'PENDING')}", expanded=False):
                ca1, ca2, ca3 = st.columns([1, 1, 1])

                with ca1:
                    st.write(f"**Sapling ID:** `{a['sapling_id']}`")
                    st.write(f"**Audit Date:** {a['audit_date']}")
                    st.markdown(f"**CV Verdict:** {render_status_pill(a['status'])}", unsafe_allow_html=True)
                    st.write(f"**RANSAC Inliers:** `{a.get('inliers_count', 0)}`")
                    st.write(f"**Good Matches:** `{a.get('good_matches_count', 0)}`")
                    st.write(f"**Inlier Share:** `{round(a.get('inlier_ratio', 0.0)*100, 1)}%`")

                with ca2:
                    v_path = a.get("vis_photo_path")
                    if v_path and os.path.exists(v_path):
                        st.image(v_path, caption="Correspondence Visualization", use_container_width=True)
                    else:
                        st.info("No visualization image cached.")

                with ca3:
                    st.markdown("#### ✍️ Reviewer Decision Action")
                    curr_dec = a.get("reviewer_decision", "PENDING")
                    st.write(f"**Current Decision:** `{curr_dec}`")
                    st.write(f"**Reviewed By:** {a.get('reviewed_by') or 'Pending'}")
                    st.write(f"**Notes:** {a.get('reviewer_notes') or 'None'}")

                    new_dec = st.selectbox("Set Decision:", ["VERIFIED", "REJECTED", "PENDING"], index=["VERIFIED", "REJECTED", "PENDING"].index(curr_dec) if curr_dec in ["VERIFIED", "REJECTED", "PENDING"] else 2, key=f"d_sel_{a['id']}")
                    new_notes = st.text_input("Reviewer Notes:", value=a.get("reviewer_notes") or "", key=f"d_nt_{a['id']}")
                    new_by = st.text_input("Auditor Name:", value=a.get("reviewed_by") or "Field Supervisor", key=f"d_by_{a['id']}")

                    if st.button("Confirm Decision", key=f"d_btn_{a['id']}", type="primary"):
                        ok_up, msg_up = storage.update_audit_decision(a["id"], new_dec, new_notes, new_by)
                        if ok_up:
                            st.success(msg_up)
                            st.rerun()
                        else:
                            st.error(msg_up)


# ==============================================================================
# PAGE F: SITES & GEOSPATIAL MAP
# ==============================================================================
elif nav_selection == "Sites & Map":
    render_header(
        title="Sites & Geospatial Monitoring",
        description="Geographic distribution of monitoring plots, sapling GPS coordinates, and regional audit compliance.",
        badge_text="GEOSPATIAL INTELLIGENCE",
    )

    all_saplings = storage.get_all_saplings()
    geo_saplings = [s for s in all_saplings if s.get("latitude") and s.get("longitude")]

    col_map, col_plots = st.columns([3, 1])

    with col_plots:
        st.markdown("#### 📍 Monitored Plots")
        plots_summary = storage.get_plots_summary()
        for p in plots_summary:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;">
                    <div style="font-weight: 700; color: #0F172A; font-size: 0.92rem;">{p['plot_name']}</div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">
                        Saplings: <strong>{p['sapling_count']}</strong> &nbsp;|&nbsp; Audits: <strong>{p['audit_count']}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_map:
        if geo_saplings:
            avg_lat = sum(s["latitude"] for s in geo_saplings) / len(geo_saplings)
            avg_lon = sum(s["longitude"] for s in geo_saplings) / len(geo_saplings)

            folium_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=10, tiles="CartoDB positron")

            for s in geo_saplings:
                popup_html = f"""
                <div style="font-family: sans-serif; font-size: 12px;">
                    <strong>{s['id']}</strong><br/>
                    Species: {s['species']}<br/>
                    Plot: {s['plot_name']}<br/>
                    Planted: {s['planting_date']}
                </div>
                """
                folium.Marker(
                    location=[s["latitude"], s["longitude"]],
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{s['id']} ({s['species']})",
                    icon=folium.Icon(color="green", icon="leaf"),
                ).add_to(folium_map)

            # Render Folium map cleanly via HTML container (independent of pandas DLL)
            try:
                map_html = folium_map.get_root().render()
                components_v1.html(map_html, height=530, scrolling=False)
            except Exception as map_err:
                st.warning(f"Interactive map component fallback: {map_err}")
                st.markdown(
                    """
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.2rem;">
                        <h4 style="margin: 0 0 6px 0; color: #0F172A;">🗺️ Geospatial Coordinates Summary</h4>
                        <p style="font-size: 0.85rem; color: #64748B; margin: 0 0 10px 0;">Coordinates registered for field telemetry:</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for s in geo_saplings[:5]:
                    st.write(f"🌱 **{s['id']}** ({s['species']}): `{s['latitude']:.5f}, {s['longitude']:.5f}` — *{s['plot_name']}*")
        else:
            st.info("No GPS coordinates recorded yet on registered saplings. Register saplings with GPS latitude and longitude, or load demo data.")


# ==============================================================================
# PAGE G: ANALYTICS & REPORTS
# ==============================================================================
elif nav_selection == "Analytics & Reports":
    render_header(
        title="Analytics & Verification Governance",
        description="Audit volume analysis, geometric match ratios, and ground review workloads across afforestation campaigns.",
        badge_text="SCIENTIFIC AUDIT ANALYTICS",
    )

    render_scientific_guardrail()

    kpis = storage.get_kpi_summary()
    all_audits = storage.get_all_audits()

    # Metric Row
    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi("Total Audits Completed", kpis["audits_completed"], "Inspection sessions logged", "📑")
    with c2:
        v_share = round((kpis["matched_audits"] / max(kpis["audits_completed"], 1)) * 100, 1)
        render_kpi("Geometric Match Share", f"{v_share}%", "High inlier correspondence", "🎯")
    with c3:
        r_share = round((kpis["requiring_review"] / max(kpis["audits_completed"], 1)) * 100, 1)
        render_kpi("Review Workload", f"{r_share}%", "Mandates on-site field check", "🛡️")

    st.markdown("---")
    st.markdown("#### 📊 Outcome Distributions")

    if all_audits:
        status_counts = {}
        for a in all_audits:
            status_counts[a["status"]] = status_counts.get(a["status"], 0) + 1

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("##### CV Engine Outcomes")
            for st_name, count in status_counts.items():
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                        <span>{render_status_pill(st_name)}</span>
                        <span style="font-weight: 700; color: #0F172A; font-size: 1rem;">{count} Audits</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with b2:
            st.markdown("##### Reviewer Supervisory Decisions")
            dec_counts = {}
            for a in all_audits:
                d = a.get("reviewer_decision", "PENDING")
                dec_counts[d] = dec_counts.get(d, 0) + 1

            for dec_name, count in dec_counts.items():
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; color: #334155;">{dec_name}</span>
                        <span style="font-weight: 700; color: #0F172A; font-size: 1rem;">{count}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No audit history available to calculate analytics.")


# ==============================================================================
# PAGE H: SETTINGS & DATA MANAGEMENT
# ==============================================================================
elif nav_selection == "Settings & Data":
    render_header(
        title="Settings & Data Management",
        description="Demonstration environments, verification thresholds, database administration, and system governance.",
        badge_text="SYSTEM GOVERNANCE",
    )

    st.markdown("#### 📦 Demo Scenarios & Sample Seeding")
    st.write("Populate the database with realistic afforestation field records, Western Ghats species, GPS coordinates, and diverse CV comparison outcomes.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🌱 Load Realistic Demo Records", type="primary", use_container_width=True):
            storage.seed_demo_data(force=True)
            st.success("✅ Demo environment loaded successfully!")
            st.rerun()

    with col_s2:
        if st.button("⚠️ Reset All Records (Clean Database)", use_container_width=True):
            storage.reset_database()
            st.warning("Database has been reset to an empty state.")
            st.rerun()

    st.markdown("---")
    st.markdown("#### ⚙️ Computer Vision Parameters & Scientific Thresholds")
    st.markdown(
        """
        - **Minimum ORB Keypoints:** `15`
        - **Lowe's Ratio Test Threshold:** `0.75`
        - **Minimum Good Matches:** `10`
        - **Minimum RANSAC Homography Inliers:** `10`
        - **Minimum Inlier Ratio Threshold:** `0.25 (25%)`
        - **Laplacian Variance Blur Threshold:** `40.0`
        """
    )
    st.caption("Engine: OpenCV 4.x Headless | Streamlit 1.65+ | Python 3.11 | SQLite 3")
