import streamlit as st
import numpy as np
import cv2
import av
import time
import io
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from gtts import gTTS
from collections import deque

# --- Configuration ---
MODEL_PATH = "best_money_model.pt"
LABELS_PATH = "labels.txt"

# --- Utility Functions ---

@st.cache_resource
def load_class_labels(path):
    """Loads class labels from the provided text file."""
    labels = {}
    try:
        # Load content of labels.txt (content fetch id: uploaded:labels.txt)
        # Note: The provided file content is: "0 0\r\n1 1\r\n2 2\r\n..."
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    # Assuming format is "index name"
                    parts = line.split()
                    if len(parts) >= 2:
                        index = int(parts[0])
                        # The name is the rest of the line (in this case, parts[1])
                        name = ' '.join(parts[1:])
                        labels[index] = name
                    else:
                        st.warning(f"Skipping malformed line in labels.txt: {line.strip()}")
        st.info(f"Loaded {len(labels)} class labels. (e.g., 0: '{labels.get(0)}')")
    except Exception as e:
        st.error(f"Error loading labels from {path}: {e}")
        # Default labels if loading fails
        labels = {i: str(i) for i in range(8)}
    
    return labels

def text_to_audio_bytes(text):
    """Converts text to MP3 audio bytes using gTTS."""
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# --- Video Processing Class for Real-Time Inference ---

class YOLOVideoProcessor(VideoProcessorBase):
    """
    Processes video frames using the loaded YOLO model for object detection 
    and updates the detection state for the main thread.
    """
    def __init__(self, labels_map, detection_state):
        try:
            # Model and labels loaded from the main thread
            self.model = st.session_state.yolo_model
            self.labels_map = labels_map
            # Thread-safe object to store latest detections for the main thread
            self.detection_state = detection_state 
            st.success("YOLO model and labels initialized.")
        except Exception as e:
            st.error(f"Error initializing processor: {e}")
            self.model = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        if self.model:
            # 1. Run Detection
            results = self.model(image, verbose=False, stream=False)
            
            # 2. Get Detected Classes and Update State
            detected_classes = set()
            # Iterate through all detected boxes
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                # Get the class name, falling back to the ID if not found
                class_name = self.labels_map.get(class_id, str(class_id))
                detected_classes.add(class_name)

            # Update the thread-safe deque with the latest unique detections
            self.detection_state.clear()
            if detected_classes:
                self.detection_state.extend(sorted(list(detected_classes)))

            # 3. Annotate Frame
            annotated_frame = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        else:
            return frame

# --- Streamlit Application ---

st.title("💸 Real-Time Money Detection with YOLOv8 & TTS")
st.caption(f"Using model: **{MODEL_PATH}** and labels from **{LABELS_PATH}**")
st.write("---")

# 1. Load Model and Labels
if 'yolo_model' not in st.session_state:
    try:
        st.session_state.yolo_model = YOLO(MODEL_PATH)
        st.session_state.class_labels = load_class_labels(LABELS_PATH) #
        
        # Initialize thread-safe objects for communication and TTS cooldown
        # Deque is used here for a simple thread-safe list of strings.
        st.session_state.detections_queue = deque() 
        # Cooldown prevents speaking on every single frame
        st.session_state.last_spoken_time = 0.0
        st.session_state.tts_cooldown_s = 3.0 
        
    except Exception as e:
        st.error(f"Failed to load model or labels: {e}")
        st.stop()

# 2. Start WebRTC Streamer
st.subheader("Live Webcam Feed (Click START to begin detection)")

webrtc_ctx = webrtc_streamer(
    key="yolo-detector",
    video_processor_factory=lambda: YOLOVideoProcessor(
        st.session_state.class_labels,
        st.session_state.detections_queue
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 3. Main Thread Loop for TTS (runs constantly while the stream is active)
if webrtc_ctx.state.playing:
    # Get the latest detected classes from the shared deque
    detected_items = list(st.session_state.detections_queue)
    current_time = time.time()
    
    # Check if anything is detected AND the TTS cooldown is over
    if detected_items and (current_time - st.session_state.last_spoken_time > st.session_state.tts_cooldown_s):
        
        # Construct the text to be spoken
        text_to_speak = "Detected: " + ", ".join(detected_items)
        
        # Generate audio bytes
        audio_bytes = text_to_audio_bytes(text_to_speak)
        
        if audio_bytes:
            # Play the audio
            st.info(f"Speaking: **{text_to_speak}**")
            # st.audio requires a file or bytes. This is the simplest way.
            st.audio(audio_bytes, format='audio/mp3', start_time=0)
            
            # Update the last spoken time
            st.session_state.last_spoken_time = current_time
            
            # Rerun the Streamlit app to trigger the st.audio player update
            st.experimental_rerun()
            
    # Display detected classes in the sidebar/main area
    if detected_items:
        st.sidebar.markdown("### Latest Detections")
        st.sidebar.success(f"Detected Classes: **{', '.join(detected_items)}**")
    else:
        st.sidebar.info("No objects detected yet.")
        
st.info(
    "TTS will announce detected objects every few seconds. Autoplay may be blocked by your browser until you interact with the page."
)



