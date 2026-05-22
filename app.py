import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion Classifier AI",
    page_icon="👗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #f0eee8;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #12101a 50%, #0e0c18 100%);
    min-height: 100vh;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(90deg, #e8c4f0, #c4d4f0, #f0e8c4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.hero-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #6b6880;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Upload Area */
.upload-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #9b96b0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: block;
}

/* Result Card */
.result-card {
    background: linear-gradient(135deg, rgba(232,196,240,0.08), rgba(196,212,240,0.05));
    border: 1px solid rgba(232,196,240,0.15);
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #e8c4f0, #c4d4f0, #f0e8c4);
}

.prediction-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #6b6880;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.prediction-value {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem;
}
.confidence-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    background: rgba(232,196,240,0.1);
    border: 1px solid rgba(232,196,240,0.25);
    color: #e8c4f0;
}

/* Class icon mapping */
.class-icon { font-size: 2rem; }

/* Prob bar */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.6rem 0;
}
.prob-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #9b96b0;
    width: 70px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
}
.prob-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #6b6880;
    width: 42px;
    text-align: right;
}

/* History */
.history-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #6b6880;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 2rem 0 0.75rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 1.5rem;
}
.history-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.history-thumb {
    width: 40px; height: 40px;
    border-radius: 8px;
    object-fit: cover;
    background: rgba(255,255,255,0.05);
}
.history-meta {
    flex: 1;
}
.history-cls { font-size: 0.85rem; font-weight: 600; }
.history-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #6b6880;
}

/* Streamlit overrides */
.stFileUploader > div {
    border: 1px dashed rgba(232,196,240,0.2) !important;
    border-radius: 12px !important;
    background: rgba(232,196,240,0.03) !important;
}
.stFileUploader > div:hover {
    border-color: rgba(232,196,240,0.4) !important;
}
div[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 100px !important;
}
div[data-testid="stProgress"] > div > div {
    border-radius: 100px !important;
}
.stSpinner > div { color: #e8c4f0 !important; }

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #3d3a50;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('fashion_model.h5')
    with open('classes.json') as f:
        class_names = json.load(f)
    return model, class_names

CLASS_ICONS = {
    'T-Shirt': '👕',
    'Dress':   '👗',
    'Pants':   '👖',
}
BAR_COLORS = {
    'T-Shirt': 'linear-gradient(90deg, #c4d4f0, #8aaae8)',
    'Dress':   'linear-gradient(90deg, #e8c4f0, #c08ae8)',
    'Pants':   'linear-gradient(90deg, #f0e8c4, #e8c08a)',
}

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Fashion Classifier</div>
    <div class="hero-subtitle">MobileNetV2 · Transfer Learning · 94.84% Accuracy</div>
</div>
""", unsafe_allow_html=True)

# ─── Load model ─────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, class_names = load_model()

# ─── Session state for history ───────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Upload ─────────────────────────────────────────────────────────────────
st.markdown('<span class="upload-label">Upload a clothing image</span>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label="upload",
    type=['jpg', 'jpeg', 'png', 'webp'],
    label_visibility="collapsed"
)

# ─── Prediction ─────────────────────────────────────────────────────────────
if uploaded:
    img = Image.open(uploaded).convert('RGB')

    col1, col2 = st.columns([1, 1.4], gap="large")
    with col1:
        st.image(img, use_column_width=True)

    with col2:
        with st.spinner("Analysing..."):
            img_arr = np.array(img.resize((224, 224))) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            pred = model.predict(img_arr, verbose=0)

        idx      = str(np.argmax(pred))
        cls_name = class_names[idx]
        conf     = pred[0][int(idx)] * 100
        icon     = CLASS_ICONS.get(cls_name, "🧥")

        # Result card
        st.markdown(f"""
        <div class="result-card">
            <div class="prediction-label">Predicted Class</div>
            <div class="prediction-value">{icon} {cls_name}</div>
            <div class="confidence-badge">⚡ {conf:.1f}% confidence</div>

            <div style="margin-top:1.5rem">
                <div class="prediction-label" style="margin-bottom:0.75rem">All Probabilities</div>
        """, unsafe_allow_html=True)

        # Probability bars
        for i, name in class_names.items():
            prob  = float(pred[0][int(i)])
            pct   = prob * 100
            color = BAR_COLORS.get(name, 'linear-gradient(90deg,#aaa,#888)')
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-name">{name}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
                </div>
                <div class="prob-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Accuracy tag
        if conf >= 90:
            st.success("✅ High confidence prediction")
        elif conf >= 70:
            st.warning("⚠️ Moderate confidence — try a clearer image")
        else:
            st.error("❌ Low confidence — image may be unclear")

    # Save to history
    st.session_state.history.insert(0, {
        "name": cls_name,
        "conf": conf,
        "icon": icon,
    })
    if len(st.session_state.history) > 5:
        st.session_state.history = st.session_state.history[:5]

# ─── History ─────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="history-title">Recent Predictions</div>', unsafe_allow_html=True)
    for item in st.session_state.history:
        st.markdown(f"""
        <div class="history-item">
            <div style="font-size:1.5rem">{item['icon']}</div>
            <div class="history-meta">
                <div class="history-cls">{item['name']}</div>
                <div class="history-conf">{item['conf']:.1f}% confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Clear History", type="secondary"):
        st.session_state.history = []
        st.rerun()

# ─── Stats Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.metric("Test Accuracy",  "94.84%")
    st.metric("Val Accuracy",   "94.84%")
    st.metric("Train Accuracy", "98.14%")
    st.metric("Total Classes",  "3")
    st.metric("Training Images","1,440")
    st.markdown("---")
    st.markdown("**Classes**")
    st.markdown("👕 T-Shirt\n\n👗 Dress\n\n👖 Pants")
    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown("MobileNetV2 + Transfer Learning + Fine-tuning")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with TensorFlow · MobileNetV2 · Streamlit — Clothing Dataset by agrigorev
</div>
""", unsafe_allow_html=True)
