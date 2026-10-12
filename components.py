"""components.py

Forest Intelligence Design System for SaplingProof AI.
Synthesizes design principles from:
- Linear: High-contrast typography scale, subtle neutral borders, refined spacing, dark-pine sidebar.
- Stripe: Clear form controls, legible field labels, high-contrast badges, operational feedback, cards.
- ArcGIS Field Maps: Real GPS coordinate pins, location telemetry, field-audit inspection workflows.

Typography Scale:
- Page Title: 28-32px, weight 750
- Section Heading: 20-22px, weight 700
- Card Title: 17-19px, weight 700
- Form Labels: 14-15px, weight 650
- Body Text: 14-15px, weight 450-500
- Metadata / Badges: 12-13px, weight 600
- Tabular Monospace: JetBrains Mono for Coordinates, IDs, and Metrics
"""

from typing import Any, Dict, List, Optional
import streamlit as st


def inject_forest_intelligence_theme():
    """Injects high-contrast, linear/stripe-inspired styling rules."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

        /* Global Reset & Base Typography */
        *, *::before, *::after {
            box-sizing: border-box;
        }

        html, body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #0F172A;
            -webkit-font-smoothing: antialiased;
            overflow-x: hidden;
        }

        .stApp {
            background-color: #F8FAF9 !important;
            overflow-x: hidden;
        }

        /* Protect Material Icon font ligatures from being turned into literal text like 'upload' */
        [data-testid="stIconMaterial"], 
        .material-symbols-rounded, 
        .material-symbols-outlined,
        [data-testid*="Icon"] {
            font-family: "Material Symbols Rounded", sans-serif !important;
            font-feature-settings: 'liga' 1;
            -webkit-font-feature-settings: 'liga' 1;
        }

        /* Top Command Header */
        .fp-command-header {
            background: linear-gradient(135deg, #091E13 0%, #123824 55%, #1B4C33 100%);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 12px;
            padding: 1.6rem 2.2rem;
            color: #FFFFFF !important;
            margin-bottom: 1.6rem;
            box-shadow: 0 4px 20px -2px rgba(9, 30, 19, 0.2);
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
        }

        .fp-command-header * {
            box-sizing: border-box;
        }

        .fp-badge-live {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(88, 176, 129, 0.22);
            border: 1px solid rgba(130, 209, 163, 0.4);
            color: #A7F3D0 !important;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }

        .fp-pulse-dot {
            width: 7px;
            height: 7px;
            background-color: #34D399;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px #34D399;
        }

        .fp-header-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #FFFFFF !important;
            margin: 0 0 0.35rem 0;
            line-height: 1.2;
        }

        .fp-header-desc {
            font-size: 0.95rem;
            color: #D1FAE5 !important;
            margin: 0;
            line-height: 1.45;
            max-width: 820px;
        }

        /* Scientific Guardrail Alert */
        .fp-guardrail {
            background-color: #FFFBEB;
            border: 1px solid #FCD34D;
            border-left: 4px solid #D97706;
            border-radius: 8px;
            padding: 1rem 1.3rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }

        .fp-guardrail-icon {
            font-size: 1.25rem;
            color: #D97706;
            line-height: 1.2;
        }

        .fp-guardrail-title {
            font-weight: 750;
            color: #92400E !important;
            margin-bottom: 3px;
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.6px;
        }

        .fp-guardrail-text {
            font-size: 0.88rem;
            color: #78350F !important;
            line-height: 1.5;
        }

        .fp-guardrail-text strong {
            color: #92400E !important;
            font-weight: 700;
        }

        /* High-Density KPI Cards */
        .fp-kpi-container {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            transition: all 0.15s ease-in-out;
            height: 100%;
        }

        .fp-kpi-container:hover {
            border-color: #1B4D3E;
            box-shadow: 0 4px 14px -2px rgba(18, 56, 36, 0.08);
            transform: translateY(-2px);
        }

        .fp-kpi-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
        }

        .fp-kpi-label {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            color: #475569 !important;
        }

        .fp-kpi-icon {
            font-size: 1.2rem;
        }

        .fp-kpi-num {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.035em;
            color: #0F172A !important;
            line-height: 1.1;
        }

        .fp-kpi-sub {
            font-size: 0.8rem;
            color: #64748B !important;
            margin-top: 0.4rem;
            font-weight: 500;
        }

        /* Status Pills with high WCAG contrast */
        .fp-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.4px;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace !important;
        }

        .fp-pill-MATCH_EVIDENCE_FOUND {
            background-color: #ECFDF5 !important;
            color: #065F46 !important;
            border: 1px solid #6EE7B7 !important;
        }

        .fp-pill-NO_RELIABLE_MATCH {
            background-color: #FEF2F2 !important;
            color: #991B1B !important;
            border: 1px solid #FCA5A5 !important;
        }

        .fp-pill-INSUFFICIENT_EVIDENCE {
            background-color: #FFFBEB !important;
            color: #92400E !important;
            border: 1px solid #FCD34D !important;
        }

        .fp-pill-INVALID_IMAGE {
            background-color: #F1F5F9 !important;
            color: #334155 !important;
            border: 1px solid #CBD5E1 !important;
        }

        .fp-pill-ACTIVE {
            background-color: #ECFDF5 !important;
            color: #065F46 !important;
            border: 1px solid #6EE7B7 !important;
        }

        .fp-pill-FLAGGED {
            background-color: #FFFBEB !important;
            color: #92400E !important;
            border: 1px solid #FCD34D !important;
        }

        .fp-pill-ARCHIVED {
            background-color: #F1F5F9 !important;
            color: #475569 !important;
            border: 1px solid #CBD5E1 !important;
        }

        /* Review Badges */
        .fp-badge-verified {
            background: #E0F2FE !important;
            color: #0369A1 !important;
            border: 1px solid #7DD3FC !important;
            padding: 3px 9px;
            border-radius: 5px;
            font-size: 0.74rem;
            font-weight: 750;
            text-transform: uppercase;
        }

        .fp-badge-pending {
            background: #FEF3C7 !important;
            color: #B45309 !important;
            border: 1px solid #FCD34D !important;
            padding: 3px 9px;
            border-radius: 5px;
            font-size: 0.74rem;
            font-weight: 750;
            text-transform: uppercase;
        }

        .fp-badge-rejected {
            background: #FEE2E2 !important;
            color: #B91C1C !important;
            border: 1px solid #FCA5A5 !important;
            padding: 3px 9px;
            border-radius: 5px;
            font-size: 0.74rem;
            font-weight: 750;
            text-transform: uppercase;
        }

        /* Dark Pine Sidebar */
        [data-testid="stSidebar"] {
            background-color: #081C10 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #F8FAFC !important;
        }

        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
            color: #CBD5E1 !important;
        }

        [data-testid="stSidebar"] .stRadio label {
            font-size: 0.9rem !important;
            font-weight: 550 !important;
            padding: 5px 0 !important;
            color: #E2E8F0 !important;
        }

        /* Visible, High-Contrast Form Controls & Labels */
        .stTextInput label, .stSelectbox label, .stDateInput label, .stTextArea label, .stSlider label {
            font-weight: 650 !important;
            font-size: 0.92rem !important;
            color: #0F172A !important;
            margin-bottom: 4px !important;
        }

        .stTextInput input, .stSelectbox select, .stDateInput input, .stTextArea textarea {
            border-radius: 8px !important;
            border: 1.5px solid #CBD5E1 !important;
            font-size: 0.92rem !important;
            color: #0F172A !important;
            background-color: #FFFFFF !important;
            padding: 0.5rem 0.75rem !important;
        }

        .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
            border-color: #1B4D3E !important;
            box-shadow: 0 0 0 3px rgba(27, 77, 62, 0.16) !important;
        }

        /* Action Buttons */
        .stButton>button {
            border-radius: 8px !important;
            font-weight: 650 !important;
            font-size: 0.9rem !important;
            padding: 0.5rem 1.25rem !important;
            transition: all 0.15s ease !important;
            border: 1.5px solid #CBD5E1 !important;
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }

        .stButton>button:hover {
            border-color: #1B4D3E !important;
            color: #1B4D3E !important;
            background-color: #F8FAF9 !important;
            transform: translateY(-1px) !important;
        }

        .stButton>button[kind="primary"] {
            background: #143D27 !important;
            color: #FFFFFF !important;
            border: 1.5px solid #143D27 !important;
            box-shadow: 0 2px 6px rgba(20, 61, 39, 0.25) !important;
        }

        .stButton>button[kind="primary"]:hover {
            background: #1B4D3E !important;
            border-color: #1B4D3E !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(27, 77, 62, 0.35) !important;
        }

        /* Tabular Monospace Coordinates & Data Tags */
        .fp-mono {
            font-family: 'JetBrains Mono', monospace !important;
            background: #F1F5F9;
            color: #0F172A !important;
            padding: 3px 7px;
            border-radius: 5px;
            font-size: 0.84rem;
            font-weight: 600;
            border: 1px solid #E2E8F0;
        }

        /* Production Registry Card Component */
        .fp-registry-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.4rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            transition: all 0.15s ease-in-out;
        }

        .fp-registry-card:hover {
            border-color: #94A3B8;
            box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.08);
        }

        .fp-registry-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 1px solid #F1F5F9;
            padding-bottom: 0.9rem;
            margin-bottom: 1rem;
        }

        .fp-registry-title {
            font-size: 1.15rem;
            font-weight: 750;
            color: #0F172A !important;
            margin: 0 0 0.25rem 0;
            line-height: 1.3;
        }

        .fp-registry-subtitle {
            font-size: 0.86rem;
            font-weight: 550;
            color: #475569 !important;
            margin: 0;
        }

        .fp-field-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.8rem 1.4rem;
        }

        .fp-field-item {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .fp-field-label {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #64748B !important;
        }

        .fp-field-value {
            font-size: 0.92rem;
            font-weight: 600;
            color: #0F172A !important;
        }

        /* Expander Title Enhancement */
        [data-testid="stExpander"] details summary {
            font-weight: 700 !important;
            font-size: 0.98rem !important;
            color: #0F172A !important;
            padding: 0.6rem 0.8rem !important;
        }

        /* Headings scale (excluding dark command header) */
        h1:not(.fp-header-title), .stApp h1:not(.fp-header-title) {
            font-size: 1.85rem !important;
            font-weight: 800 !important;
            color: #0F172A !important;
            letter-spacing: -0.025em !important;
        }
        h2, .stApp h2 {
            font-size: 1.45rem !important;
            font-weight: 750 !important;
            color: #0F172A !important;
            letter-spacing: -0.02em !important;
        }
        h3, .stApp h3 {
            font-size: 1.25rem !important;
            font-weight: 700 !important;
            color: #0F172A !important;
        }
        h4, .stApp h4 {
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            color: #0F172A !important;
        }

        /* Streamlit Native File Uploader Polish */
        [data-testid="stFileUploader"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 1.5px dashed #94A3B8 !important;
            border-radius: 10px !important;
            background-color: #FFFFFF !important;
            padding: 1.2rem 1.4rem !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #1B4D3E !important;
            background-color: #F8FAF9 !important;
        }

        /* Ensure dropzone button has clean font and clear single label */
        [data-testid="stFileUploaderDropzone"] button {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            color: #0F172A !important;
            border-radius: 6px !important;
            padding: 0.4rem 1rem !important;
            border: 1.5px solid #CBD5E1 !important;
        }

        [data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded', sans-serif !important;
        }

        /* Ensure dropzone instructions text is crisp and readable */
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: #64748B !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
        }

        /* Card Container Utility */
        .fp-lab-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.4rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            height: 100%;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            width: 100%;
            max-width: 100%;
        }

        .fp-lab-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #F1F5F9;
            padding-bottom: 0.75rem;
            margin-bottom: 1rem;
        }

        .fp-lab-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, description: str, badge_text: str = "FOREST INTELLIGENCE COMMAND CENTER"):
    """Renders the top command banner."""
    st.markdown(
        f"""
        <div class="fp-command-header">
            <div class="fp-badge-live">
                <span class="fp-pulse-dot"></span> {badge_text}
            </div>
            <h1 class="fp-header-title">{title}</h1>
            <p class="fp-header-desc">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scientific_guardrail():
    """Renders the mandatory scientific policy notice."""
    st.markdown(
        """
        <div class="fp-guardrail">
            <div class="fp-guardrail-icon">⚠️</div>
            <div class="fp-guardrail-body">
                <div class="fp-guardrail-title">Scientific Correspondence Protocol</div>
                <div class="fp-guardrail-text">
                    SaplingProof AI measures <strong>2D optical feature correspondences (ORB descriptors + RANSAC homography consensus)</strong>.
                    Visual similarity reflects shared camera-angle textures, stem outlines, or markers.
                    <strong>Optical feature correspondence does NOT prove botanical identity, continuous survival, or physiological health.</strong>
                    Any flagged or ambiguous observation mandates ground-truth forestry inspection.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: Any, subtext: str = "", icon: str = "📊"):
    """Renders a single high-density KPI metric card."""
    st.markdown(
        f"""
        <div class="fp-kpi-container">
            <div class="fp-kpi-header">
                <span class="fp-kpi-label">{label}</span>
                <span class="fp-kpi-icon">{icon}</span>
            </div>
            <div class="fp-kpi-num">{value}</div>
            <div class="fp-kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(status: str) -> str:
    """Returns HTML for a status badge pill."""
    clean_name = status.replace("_", " ")
    return f'<span class="fp-pill fp-pill-{status}">{clean_name}</span>'


# Backward compatibility aliases
render_hero = render_header
render_kpi_card = render_kpi
