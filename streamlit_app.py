import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile
import os
import pyttsx3


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
def load_tflite_model(model_path="model.tflite"):
    """Load a TensorFlow Lite model"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image, input_size=(640, 640)):
    """Resize and normalize image for model"""
    img = cv2.resize(image, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def run_inference(interpreter, input_data):
    """Run inference on TFLite model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    outputs = [interpreter.get_tensor(out['index']) for out in output_details]
    return outputs


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

    # Convert to OpenCV format
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Load model
    interpreter = load_tflite_model("model.tflite")

    # Preprocess
    input_data = preprocess_image(img_bgr)

    # Run inference
    outputs = run_inference(interpreter, input_data)

    # Post-process (simple argmax classification for now)
    preds = outputs[0]
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    if confidence >= confidence_threshold:
        label = CLASS_LABELS.get(class_id, "Unknown")
        st.success(f"✅ Detected: {label} (Confidence: {confidence:.2f})")
        text_to_speech(f"Detected {label}")
    else:
        st.warning("⚠️ No confident detection")


st.sidebar.write("### Supported Classes")
for idx, name in CLASS_LABELS.items():
    st.write(f"{idx}: {name}")
