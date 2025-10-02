import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import pyttsx3
from ultralytics import YOLO   # ✅ keep Ultralytics

def load_class_labels(label_path="labels.txt"):
    """Load class labels from file and clean them up for TTS"""
    labels = {}
    with open(label_path, "r") as f:
        for line in f.readlines():
            idx, name = line.strip().split(maxsplit=1)
            idx = int(idx)

            # Clean up whitespace and format for TTS
            name = name.strip()
            if name.lower() != "background":
                labels[idx] = f"{name} Rupees"
            else:
                labels[idx] = "Background"
    return labels

# Load labels once
CLASS_LABELS = load_class_labels("labels.txt")

@st.cache_resource
def load_model(model_path="best.pt"):
    """Load YOLO model"""
    return YOLO(model_path)

def text_to_speech(text):
    """Speak detection result"""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"TTS error: {e}")

# Streamlit UI
st.title("💵 Money Detector for the Blind (Web/PWA)")

st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

uploaded_file = st.file_uploader("Upload an image of currency note", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temp file for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Load YOLO model
    model = load_model("best.pt")

    # Run inference
    results = model(temp_path)

    # Draw and display results
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf >= confidence_threshold:
                label = CLASS_LABELS.get(cls, "Unknown")
                st.success(f"✅ Detected: {label} (Confidence: {conf:.2f})")
                text_to_speech(f"Detected {label}")

    # Show annotated image
    annotated_frame = results[0].plot()
    st.image(annotated_frame, caption="Detections", use_container_width=True)

st.sidebar.write("### Supported Classes")
for idx, name in CLASS_LABELS.items():
    st.write(f"{idx}: {name}")
