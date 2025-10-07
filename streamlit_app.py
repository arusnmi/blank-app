import os
from pathlib import Path
import base64

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
from gtts import gTTS
import io
import numpy as np
from PIL import Image

# 1. PATH FIX: Get the absolute path to the root of the deployed app
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_money_model.pt"
LABELS_PATH = BASE_DIR / "labels.txt"

st.set_page_config(layout="wide", page_title="Money Detection App", page_icon="💵")
st.title("💵 Money Detection App")

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
with st.spinner("Loading AI model..."):
    model = load_yolo_model(MODEL_PATH)
    class_labels = load_class_labels(LABELS_PATH)

# Check if resources loaded successfully
if model is None or not class_labels:
    st.error("Application setup failed. Check logs for model/labels loading errors.")
    st.stop()

# Initialize session state
if 'last_detected' not in st.session_state:
    st.session_state['last_detected'] = None
if 'detection_count' not in st.session_state:
    st.session_state['detection_count'] = 0

# Add CSS for larger camera button
st.markdown("""
    <style>
        .stCamera > button {
            transform: scale(2.0);
            margin: 20px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.subheader("Take a Photo with Your Camera")
camera_image = st.camera_input("Capture money/currency", label_visibility="collapsed", key="camera")

if camera_image:
    uploaded_file = camera_image
    input_source = "camera"
else:
    uploaded_file = None
    input_source = None

# 4. PROCESS AND DISPLAY RESULTS
if uploaded_file is not None:
    # Read and display original image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="📷 Original Image", use_container_width=True)
    
    # Perform detection
    with st.spinner("🔍 Analyzing image..."):
        results = model.predict(img_bgr, verbose=False, conf=0.75, device='cpu')
    
    # Process results
    detected_items = []
    annotated_img = img_bgr.copy()
    
    if results and results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            detected_class = class_labels.get(class_id, f"Unknown Class {class_id}")
            confidence = box.conf.item() * 100
            
            detected_items.append({
                'class': detected_class,
                'confidence': confidence
            })
            
            # Draw bounding box
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            label = f"{detected_class}: {confidence:.1f}%"
            
            # Draw box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                annotated_img, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                (0, 255, 0), 
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_img, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 0), 
                2
            )
    
    # Convert back to RGB for display
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.image(annotated_img_rgb, caption="🎯 Detection Results", use_container_width=True)
    
    # Display results summary
    st.divider()
    
    if detected_items:
        st.success(f"✅ Successfully detected {len(detected_items)} item(s)!")
        
        # Increment detection count
        st.session_state['detection_count'] += 1
        
        # Create detection summary with columns
        cols = st.columns(min(len(detected_items), 3))
        
        for idx, item in enumerate(detected_items):
            with cols[idx % 3]:
                st.metric(
                    label=f"💵 Detection {idx + 1}",
                    value=item['class'],
                    delta=f"{item['confidence']:.1f}% confidence"
                )
        
        # Text-to-speech for first detection
        non_background_detections = [item for item in detected_items if item['class'].lower() != 'background']
        
        if non_background_detections:
            first_detection = non_background_detections[0]['class']
            
            try:
                # Create audio directly without caching
                tts = gTTS(text=f"Detected {first_detection}", lang='en', slow=False)
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)
                
                # Convert audio to base64 for HTML audio element
                audio_bytes = audio_fp.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                
                # Create HTML audio element with autoplay
                audio_html = f"""
                    <audio autoplay="true">
                        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                    </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"Audio generation failed: {str(e)}")

        # Show all detections in an expander
        with st.expander("📋 Detailed Detection Information"):
            for idx, item in enumerate(detected_items, 1):
                st.write(f"**{idx}.** {item['class']} - Confidence: {item['confidence']:.2f}%")
        
    else:
        st.warning("⚠️ No money detected in the image")
        st.info("💡 **Tips to improve detection:**\n- Ensure good lighting\n- Keep money flat and visible\n- Avoid shadows or reflections\n- Try a different angle")
    
    # Show statistics
    if st.session_state['detection_count'] > 0:
        st.sidebar.metric("Total Detections", st.session_state['detection_count'])

else:
    # Welcome message when no image is loaded
    st.info("👆 **Get Started:** Use your camera to take a photo of money")
    
    # Show supported labels in sidebar
    if class_labels:
        with st.sidebar:
            st.subheader("🏷️ Supported Denominations")
            for class_id, label in sorted(class_labels.items()):
                st.write(f"• {label}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This app uses YOLO AI to detect and identify money denominations in images. Perfect for accessibility and currency verification!")