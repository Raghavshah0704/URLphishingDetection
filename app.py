"""
Streamlit Web App — PhiUSIIL Phishing URL Detector
===================================================
A clean, modern dashboard for phishing detection using the PhiUSIIL dataset
trained models. Supports three input modes:
  1. Enter a URL directly (auto-extracts features)
  2. Paste JSON/CSV feature values
  3. Manual sliders
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.predictor import PhishingPredictor
from src.url_feature_extractor import extract_features_from_url

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhiUSIIL Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# INJECT CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.4); }

/* Verdict banners */
.verdict-phishing {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 2px;
    box-shadow: 0 0 40px rgba(255,65,108,0.5);
    animation: pulse 1.5s infinite;
}
.verdict-legit {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 2px;
    box-shadow: 0 0 40px rgba(56,239,125,0.4);
}
@keyframes pulse {
    0%   { box-shadow: 0 0 40px rgba(255,65,108,0.5); }
    50%  { box-shadow: 0 0 70px rgba(255,65,108,0.9); }
    100% { box-shadow: 0 0 40px rgba(255,65,108,0.5); }
}

/* Model row */
.model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.badge-phish { background:#ff416c; color:#fff; border-radius:8px; padding:4px 12px; font-weight:600; }
.badge-legit { background:#38ef7d; color:#000; border-radius:8px; padding:4px 12px; font-weight:600; }

/* Progress bar custom */
.stProgress > div > div { background: linear-gradient(90deg, #11998e, #38ef7d); border-radius: 8px; }

/* Feature table */
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 24px;
    font-size: 0.85rem;
    padding: 12px;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}
.feature-item {
    display: flex;
    justify-content: space-between;
    padding: 4px 8px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.feature-name { color: #a0a0c0; }
.feature-val { color: #e0e0f0; font-weight: 600; font-family: monospace; }

h1, h2, h3 { color: #ffffff; }
label, .stTextInput label { color: #c0c0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return PhishingPredictor()

try:
    predictor = load_predictor()
    feature_cols = predictor.feature_cols
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load models. Please run `python train.py` first.\n\nError: {e}")
    models_loaded = False
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — mode selection
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ PhiUSIIL Detector")
    st.markdown("Powered by ensemble ML trained on **20,174** URLs")
    st.divider()

    st.markdown("### 📋 Input Mode")
    input_mode = st.radio(
        "Choose how to provide features",
        ["🔗 Enter URL", "📥 JSON / CSV Row", "🎛️ Manual Sliders"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("""
**Dataset:** PhiUSIIL (Clean)  
**Rows:** 20,174  
**Features:** 19 (leaky + zero-variance removed)  
**Models:** RF · ET · GB · LR · KNN  
**Best:** Gradient Boosting (AUC 0.9356)
""")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — render verdict
# ─────────────────────────────────────────────────────────────────────────────
def render_verdict(result, show_url=None):
    """Render the prediction verdict and per-model breakdown."""
    st.divider()
    st.markdown("### 🎯 Verdict")

    if show_url:
        st.markdown(f"**URL:** `{show_url}`")

    if result["final_label"] == 1:
        st.markdown(f'<div class="verdict-phishing">⚠️ PHISHING DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-legit">✅ LEGITIMATE</div>', unsafe_allow_html=True)

    st.markdown(f"**Confidence (phishing prob):** `{result['confidence']:.1%}`")
    st.progress(result["confidence"])

    st.markdown("### 🤖 Per-Model Predictions")
    for mname, mres in result["per_model"].items():
        badge = f'<span class="badge-phish">PHISHING</span>' if mres["label"] == 1 else f'<span class="badge-legit">LEGIT</span>'
        st.markdown(
            f'<div class="model-row"><span>🔹 <b>{mname.replace("_"," ").title()}</b></span>'
            f'<span>{badge} &nbsp; <code>{mres["prob"]:.4f}</code></span></div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ PhiUSIIL Phishing URL Detector")
st.markdown("*Ensemble ML classifier trained on the PhiUSIIL dataset — 20K+ URLs, 19 features*")
st.divider()

# ── MODE: URL Input ───────────────────────────────────────────────────────────
if input_mode == "🔗 Enter URL":
    st.markdown("### 🔗 Enter a URL to Check")
    st.markdown("Paste any URL below — features will be automatically extracted from the live page.")

    url_input = st.text_input(
        "URL",
        placeholder="https://example.com/login",
        label_visibility="collapsed"
    )

    col_detect, col_info = st.columns([1, 2])
    with col_detect:
        detect_btn = st.button("🔍 Detect Phishing", use_container_width=True, type="primary")
    with col_info:
        st.caption("⏱️ Takes 5-15 seconds (fetches the page & extracts 19 features)")

    if detect_btn:
        if not url_input or not url_input.strip():
            st.warning("Please enter a URL first.")
            st.stop()

        url_clean = url_input.strip()
        if not url_clean.startswith(("http://", "https://")):
            url_clean = "http://" + url_clean

        # Extract features
        with st.spinner(f"🌐 Fetching and analyzing `{url_clean}` ..."):
            feature_dict = extract_features_from_url(url_clean)

        # Show extracted features in expander
        with st.expander("📊 Extracted Features (19)", expanded=False):
            feat_html = '<div class="feature-grid">'
            for fname, fval in feature_dict.items():
                display_val = f"{fval:.4f}" if isinstance(fval, float) else str(fval)
                feat_html += f'<div class="feature-item"><span class="feature-name">{fname}</span><span class="feature-val">{display_val}</span></div>'
            feat_html += '</div>'
            st.markdown(feat_html, unsafe_allow_html=True)

        # Predict
        with st.spinner("Running ensemble prediction…"):
            result = predictor.predict(feature_dict)

        render_verdict(result, show_url=url_clean)

# ── MODE: JSON / CSV ──────────────────────────────────────────────────────────
elif input_mode == "📥 JSON / CSV Row":
    st.markdown("### 📥 Paste Feature Values (JSON or CSV Row)")
    st.markdown("Paste a raw dictionary of feature values, or a single CSV row (with header).")

    sample = {c: 0 for c in feature_cols}
    sample_str = json.dumps(sample, indent=2)

    json_input = st.text_area("Feature JSON", value=sample_str, height=300)

    url_hint = st.text_input("URL (for display only, not used in prediction)", placeholder="https://example.com/page")

    if st.button("🔍 Detect", use_container_width=True, type="primary"):
        try:
            feature_dict = json.loads(json_input)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        with st.spinner("Running ensemble prediction…"):
            result = predictor.predict(feature_dict)

        render_verdict(result, show_url=url_hint if url_hint else None)

# ── MODE: Manual Sliders ─────────────────────────────────────────────────────
else:
    st.markdown("### 🎛️ Adjust Feature Values")
    st.info("Move the sliders to set feature values, then click **Detect**.")

    feature_dict = {}

    # Group features by type for nicer UI
    url_features = [
        "domain_length", "has_hyphen", "url_shortener", "suspicious_words",
        "domain_age_category", "dns_record", "domain_entropy",
    ]
    html_features = [
        "iframe", "popup_window", "onmouseover", "num_forms",
        "external_scripts", "external_images", "hidden_elements",
        "internal_links", "external_links", "null_links",
        "external_ratio", "script_length",
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔗 URL-Based Features")
        for feat in url_features:
            if feat in feature_cols:
                if feat in ("has_hyphen", "url_shortener", "dns_record"):
                    feature_dict[feat] = st.selectbox(feat, [0, 1], key=feat)
                elif feat == "domain_age_category":
                    feature_dict[feat] = st.selectbox(feat, [0, 1, 2], key=feat,
                        help="0=new/unknown, 1=medium, 2=established")
                elif feat == "domain_entropy":
                    feature_dict[feat] = st.slider(feat, 0.0, 5.0, 2.5, key=feat)
                elif feat == "domain_length":
                    feature_dict[feat] = st.number_input(feat, min_value=1, value=10, key=feat)
                elif feat == "suspicious_words":
                    feature_dict[feat] = st.number_input(feat, min_value=0, value=0, key=feat)

    with col2:
        st.markdown("#### 🌐 HTML / Content Features")
        for feat in html_features:
            if feat in feature_cols:
                if feat in ("onmouseover",):
                    feature_dict[feat] = st.selectbox(feat, [0, 1], key=feat)
                elif feat == "external_ratio":
                    feature_dict[feat] = st.slider(feat, 0.0, 1.0, 0.0, key=feat)
                else:
                    feature_dict[feat] = st.number_input(feat, min_value=0, value=0, key=feat)

    if st.button("🔍 Detect Phishing", use_container_width=True, type="primary"):
        with st.spinner("Running ensemble prediction…"):
            result = predictor.predict(feature_dict)
        render_verdict(result)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='text-align:center; color:#666; font-size:0.8rem;'>
  PhiUSIIL Phishing Detector · Ensemble ML · 5 Models · 20,174 training samples · 19 features
</p>
""", unsafe_allow_html=True)
