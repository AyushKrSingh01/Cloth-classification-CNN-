"""
Streamlit web app for the Lab-4 clothing image classifier.
Run with:  streamlit run app.py
"""

import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------------------------
# Config — must match what the model was trained with in the notebook
# ----------------------------------------------------------------------
IMG_SIZE = 128
MODEL_PATH = "model/clothing_cnn_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"

st.set_page_config(page_title="Clothing Image Classifier", page_icon="👕", layout="centered")


# ----------------------------------------------------------------------
# Cache the model and class names so they load only once per session
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, convert to RGB, rescale to [0,1], add batch dimension."""
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def predict(model, class_names, image: Image.Image):
    arr = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]  # shape (num_classes,)
    top_idx = np.argsort(preds)[::-1][:3]
    top3 = [(class_names[i], float(preds[i])) for i in top_idx]
    return top3


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("👕 Clothing Image Classifier")
st.write(
    "Upload an image of a clothing item and the trained CNN model "
    "(from Lab-4) will predict its category."
)

model, class_names = load_model()

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying..."):
        top3 = predict(model, class_names, image)

    best_label, best_conf = top3[0]

    st.subheader("Prediction")
    st.markdown(f"**Predicted class:** `{best_label}`")
    st.markdown(f"**Confidence:** `{best_conf * 100:.2f}%`")

    st.subheader("Top-3 Predictions")
    for label, conf in top3:
        st.write(f"{label} — {conf * 100:.2f}%")
        st.progress(min(int(conf * 100), 100))
else:
    st.info("👆 Upload an image to get a prediction.")