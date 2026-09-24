import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Page config
st.set_page_config(
    page_title="SignSpeak",
    page_icon="🤟",
    layout="centered"
)

# Title
st.title("🤟 SignSpeak")
st.markdown("*Real-Time Sign Language Recognition*")

# Load model (cached)
@st.cache_resource
def load_model():
    # Try multiple paths for flexibility
    possible_paths = [
        "models/baseline_cnn_final.keras",
        "checkpoints/baseline_cnn_best.keras",
        "baseline_cnn_best.keras",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
    raise FileNotFoundError("Model file not found")

# Class names
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "SignSpeak classifies American Sign Language (ASL) "
    "hand signs into 29 classes (A-Z + del + nothing + space).\n\n"
    "**Model:** Custom CNN\n"
    "**Accuracy:** 96.59%"
)

# Main
uploaded_file = st.file_uploader(
    "Upload an image of a hand sign",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    model = load_model()
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", width="stretch")
    
    # Preprocess
    img_resized = image.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    with st.spinner("Classifying..."):
        predictions = model.predict(img_array, verbose=0)[0]
        pred_idx = np.argmax(predictions)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = predictions[pred_idx]
    
    # Show result
    with col2:
        st.markdown("### Result")
        st.markdown(f"## {pred_class}")
        st.metric("Confidence", f"{confidence*100:.2f}%")
    
    # Top 5 predictions
    st.markdown("### Top 5 Predictions")
    top5_idx = np.argsort(predictions)[-5:][::-1]
    for idx in top5_idx:
        st.progress(float(predictions[idx]), text=f"{CLASS_NAMES[idx]}: {predictions[idx]*100:.2f}%")

st.markdown("---")
st.caption("Research project only. Not for real-world accessibility use without validation.")
