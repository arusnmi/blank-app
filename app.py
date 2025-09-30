import os
from pathlib import Path
import streamlit as st
from ultralytics import YOLO
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from gtts import gTTS
import io
import time
import numpy as np

# 1. PATH FIX: Get the absolute path to the root of the deployed app
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_money_model.pt"
LABELS_PATH = BASE_DIR / "labels.txt"

st.set_page_config(layout="wide")
st.title("Money Detection App")

# 2. LOAD RESOURCES: Use st.cache_resource to load expensive resources once
@st.cache_resource
def load_yolo_model(path):
    """Load the YOLO model using the corrected absolute path."""
    try:
        model = YOLO(str(path))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_class_labels(path):
    """Load class labels from the text file using the corrected absolute path."""
    labels = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                # The line format is confirmed to be "ID Label" (e.g., "0 0", "1 1") 
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Map the class ID (int) to the label (string)
                    labels[int(parts[0])] = parts[1]
        return labels
    except Exception as e:
        st.error(f"Error loading labels file at {path}: {e}")
        # Return an empty dictionary if loading fails
        return {}

# Load the model and labels using the fixed paths
model = load_yolo_model(MODEL_PATH)
class_labels = load_class_labels(LABELS_PATH)

# Check if resources loaded successfully
if model is None or not class_labels:
    st.error("Application setup failed. Check logs for model/labels loading errors.")
    st.stop()


# 3. VIDEO TRANSFORMER CLASS
class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels
        self.detected_class = None
        # Initialize gTTS only once per session
        self.tts = None 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform detection
        results = self.model.predict(img, verbose=False, conf=0.5)
        
        # Check if any objects were detected
        if results and results[0].boxes:
            box = results[0].boxes[0]
            class_id = int(box.cls.item())
            
            # Use the corrected class_labels dictionary
            self.detected_class = self.labels.get(class_id, f"Unknown Class {class_id}")
            confidence = box.conf.item() * 100

            # Draw bounding box and label
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            label = f"{self.detected_class}: {confidence:.1f}%"
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label background
            cv2.rectangle(img, (x1, y1 - 20), (x1 + len(label) * 10, y1), (0, 255, 0), -1)
            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            self.detected_class = None

        return img

# 4. STREAMLIT UI AND WEBRTC SETUP
st.subheader("Live Money Detection via Webcam")

ctx = webrtc_streamer(
    key="money-detection-key",
    mode=WebRtcMode.SENDRECV,
    # Pass the model and labels to the VideoTransformer
    video_transformer_factory=lambda: VideoTransformer(model, class_labels),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 5. TEXT-TO-SPEECH ANNOUNCEMENT LOGIC
if ctx.video_transformer:
    detected_class = ctx.video_transformer.detected_class
    if detected_class:
        # Create a unique key for the audio file to prevent caching issues
        audio_key = f"tts_audio_{detected_class}"
        
        @st.cache_data(ttl=3600, show_spinner=False)
        def create_and_cache_audio(text):
            """Generates audio bytes and caches them."""
            try:
                tts = gTTS(text=f"Detected {text}", lang='en', slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.read()
            except Exception as e:
                # Handle gTTS errors gracefully
                st.warning(f"Error generating audio: {e}")
                return None

        audio_bytes = create_and_cache_audio(detected_class)
        
        if audio_bytes:
            # Display the audio player
            st.audio(audio_bytes, format='audio/mp3', autoplay=True, loop=False)
            
        # Optional: Display the last detected item for debugging
        st.info(f"Last Detected Item: **{detected_class}**")