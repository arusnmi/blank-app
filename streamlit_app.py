import os
from pathlib import Path

# CRITICAL FIX: Set environment variables BEFORE any OpenCV imports
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
import cv2

# CRITICAL FIX: Patch cv2.imshow for headless environment (before ultralytics import)
cv2.imshow = lambda *args: None
cv2.waitKey = lambda *args: None
cv2.destroyAllWindows = lambda *args: None

from ultralytics import YOLO
# Note: Use VideoProcessorBase and WebRtcMode.SENDRECV for maximum stability
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase 
from gtts import gTTS
import io
import time
import numpy as np

# 1. PATH FIX: Get the absolute path to the root of the deployed app
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_money_model.pt"
LABELS_PATH = BASE_DIR / "labels.txt"

# --- CRITICAL OPTIMIZATION CONSTANTS ---

# CRITICAL SPEED FIX: Width for YOLO inference only (640 is a common YOLO input size).
YOLO_INPUT_SIZE_W = 640 

# LOCAL-ONLY FIX: Minimal WebRTC configuration for stability
RTC_CONFIGURATION = {
    "iceServers": [], 
    "iceTransportPolicy": "all", 
}

# FINAL FIX: Target a lower resolution for the camera itself (e.g., 480p)
CAMERA_CONSTRAINTS = {
    "video": {
        "width": 640,
        "height": 480,
    },
    "audio": False
}
# ----------------------------------

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
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[int(parts[0])] = parts[1]
        return labels
    except Exception as e:
        st.error(f"Error loading labels file at {path}: {e}")
        return {}

# Load the model and labels using the fixed paths
model = load_yolo_model(MODEL_PATH)
class_labels = load_class_labels(LABELS_PATH)

# Check if resources loaded successfully
if model is None or not class_labels:
    st.error("Application setup failed. Check logs for model/labels loading errors.")
    st.stop()


# 3. VIDEO PROCESSOR CLASS (Optimized for CPU and Stability)
class VideoProcessor(VideoProcessorBase):
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels
        self.detected_class = None
        self.tts = None 
        
        # FINAL FIX: Try 1.0 second delay (1 FPS). Lower resolution should allow this to stabilize.
        self.time_threshold = 1 / 2
        self.last_run_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        original_h, original_w = img.shape[:2]

        # SPEED FIX: Frame skipping logic (Active at 1 FPS)
        current_time = time.time()
        if current_time - self.last_run_time < self.time_threshold:
            return img
        self.last_run_time = current_time

        # CRITICAL OPTIMIZATION 1: Downscale image for fast YOLO inference.
        # Note: This is now less drastic since the input frame is already smaller.
        scale_factor = original_w / YOLO_INPUT_SIZE_W
        processing_img = cv2.resize(img, (YOLO_INPUT_SIZE_W, int(original_h / scale_factor)))
        
        # CRITICAL OPTIMIZATION 2: Convert BGR to RGB 
        processing_img_rgb = cv2.cvtColor(processing_img, cv2.COLOR_BGR2RGB)
        
        # Perform detection on the smaller image
        # TARGET CPU ('cpu') remains the most stable setting.
        results = self.model.predict(processing_img_rgb, verbose=False, conf=0.5, device='cpu') 
        
        # Check if any objects were detected
        if results and results[0].boxes:
            box = results[0].boxes[0]
            class_id = int(box.cls.item())
            
            self.detected_class = self.labels.get(class_id, f"Unknown Class {class_id}")
            confidence = box.conf.item() * 100

            # Draw bounding box and label
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1_scaled, y1_scaled, x2_scaled, y2_scaled = xyxy

            # CRITICAL OPTIMIZATION 3: Rescale coordinates back to the original image size for accurate drawing
            x1 = int(x1_scaled * scale_factor)
            y1 = int(y1_scaled * scale_factor)
            x2 = int(x2_scaled * scale_factor)
            y2 = int(y2_scaled * scale_factor)
            
            label = f"{self.detected_class}: {confidence:.1f}%"
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label background
            cv2.rectangle(img, (x1, y1 - 20), (x1 + len(label) * 10, y1), (0, 255, 0), -1)
            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            self.detected_class = None

        return img # Return the modified frame

# 4. STREAMLIT UI AND WEBRTC SETUP
st.subheader("Live Money Detection via Webcam")

# CRITICAL FIX 4: Initialize session state for audio debounce
if 'last_announced_class' not in st.session_state:
    st.session_state['last_announced_class'] = None

# CRITICAL FIX 6: Use SENDRECV mode for stable video display
ctx = webrtc_streamer(
    key="money-detection-key",
    mode=WebRtcMode.SENDONLY, 
    video_processor_factory=lambda: VideoProcessor(model, class_labels),
    # FINAL FIX: Use the new, lower camera constraints
    media_stream_constraints=CAMERA_CONSTRAINTS, 
    async_processing=True,
    rtc_configuration=RTC_CONFIGURATION,
)

# 5. TEXT-TO-SPEECH ANNOUNCEMENT LOGIC (Debounced)
if ctx.video_processor and ctx.video_processor.detected_class is not None:
    detected_class = ctx.video_processor.detected_class
    
    # CRITICAL FIX: Only proceed if the detected class is NEW to prevent stream interruption.
    if detected_class != st.session_state['last_announced_class']:
        st.session_state['last_announced_class'] = detected_class
        
        @st.cache_data(ttl=3600, show_spinner=False)
        def create_and_cache_audio(text):
            try:
                tts = gTTS(text=f"Detected {text}", lang='en', slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.read()
            except Exception as e:
                st.warning(f"Error generating audio: {e}")
                return None

        audio_bytes = create_and_cache_audio(detected_class)
        
        if audio_bytes:
            st.audio(audio_bytes, format='audio/mp3', autoplay=True, loop=False)
            
        st.info(f"Last Detected Item: **{detected_class}**")

elif ctx.video_processor:
    st.info("Video stream is running. Hold an object in view for detection.")