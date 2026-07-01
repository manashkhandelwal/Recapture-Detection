"""
main.py  –  Streamlit demo for the screen-recapture detector.

Lets a user either take a photo with their device camera or upload an
existing image, runs it through predict.py's pipeline, and displays the
score, REAL/SCREEN label, confidence, latency, and a per-signal
breakdown.

Run with
--------
    streamlit run main.py

Requires `weights.json` (produced by train.py) and `features.py` /
`predict.py` in the same directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from predict import predict_array, SIGNAL_NAMES  # noqa: E402

WEIGHTS_PATH = _HERE / "weights.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_uploaded_image(uploaded_file) -> np.ndarray | None:
    """
    Decode a Streamlit UploadedFile (from camera_input or file_uploader)
    into a BGR numpy array, the format predict_array expects.

    Returns None if decoding fails (corrupted file, unsupported format).
    """
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr  # None if cv2 couldn't decode it


def render_result(result: dict) -> None:
    """Render the prediction result: label, score, confidence, latency, signals."""
    label       = result["label"]
    score       = result["score"]
    confidence  = result["confidence"]
    latency_ms  = result["latency_ms"]

    if label == "SCREEN":
        st.error(f"🖥️ **SCREEN** detected  —  score {score:.3f}")
    else:
        st.success(f"📷 **REAL** photo  —  score {score:.3f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Score (0=real, 1=screen)", f"{score:.3f}")
    col2.metric("Confidence", f"{confidence:.0%}")
    col3.metric("Latency", f"{latency_ms:.1f} ms")

    st.progress(min(max(score, 0.0), 1.0))
    st.caption(
        f"Decision threshold: {result['threshold']:.2f}  ·  "
        f"Resolution: {result['resolution']}"
    )

    with st.expander("Per-signal breakdown"):
        st.caption(
            "Each signal scores 0–1 (higher = more screen-like). "
            "Contribution = signal score × its trained weight."
        )
        for name in SIGNAL_NAMES:
            sig_val = result["signals"][name]
            weight  = result["weights"][name]
            contrib = sig_val * weight
            bar_col, val_col = st.columns([4, 1])
            with bar_col:
                st.write(f"**{name}**  (weight {weight:.3f})")
                st.progress(min(max(sig_val, 0.0), 1.0))
            with val_col:
                st.write(f"{sig_val:.3f}")
                st.caption(f"→ {contrib:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Spot the Fake Photo", page_icon="🔍", layout="centered")

st.title("🔍 Spot the Fake Photo")
st.caption(
    "Classical signal-based screen-recapture detector. Take a photo or "
    "upload one — no image is uploaded anywhere, everything runs locally "
    "in this app's process."
)

if not WEIGHTS_PATH.exists():
    st.warning(
        "`weights.json` not found next to main.py — falling back to the "
        "built-in default weights baked into predict.py. Run `train.py` "
        "first for the calibrated, dataset-trained weights."
    )

tab_camera, tab_upload = st.tabs(["📷 Use camera", "📁 Upload a file"])

img_bgr = None

with tab_camera:
    st.caption("Take a photo directly — works on desktop and mobile browsers.")
    camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_file is not None:
        img_bgr = decode_uploaded_image(camera_file)
        if img_bgr is None:
            st.error("Could not decode the captured photo. Try again.")

with tab_upload:
    st.caption("Upload an existing image instead.")
    upload_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if upload_file is not None:
        img_bgr = decode_uploaded_image(upload_file)
        if img_bgr is None:
            st.error("Could not decode the uploaded file — unsupported "
                     "format or corrupted file.")

st.divider()

if img_bgr is not None:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Input image", use_container_width=True)

    with st.spinner("Analysing…"):
        try:
            result = predict_array(img_bgr, weights_path=str(WEIGHTS_PATH))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            result = None

    if result is not None:
        render_result(result)
else:
    st.info("Take a photo or upload an image above to get a prediction.")

st.divider()
st.caption(
    "Score convention: 0.0 = confidently real, 1.0 = confidently a screen "
    "recapture. Threshold and weights come from train.py's grid search on "
    "the labelled real/ and screen/ datasets — see report.txt for the "
    "full training report."
)