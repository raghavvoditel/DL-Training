# Upload a picture of a handwritten digit and the CNN guesses which digit it is.
# First run train_cnn.py, then:  streamlit run predict_app.py

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

MODEL_PATH = Path(__file__).parent / "digit_cnn.keras"


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def prepare(image):
    """Make the uploaded picture look like MNIST: 28x28, grayscale, white digit on black."""
    img = image.convert("L").resize((28, 28))  # "L" = grayscale
    pixels = np.array(img) / 255.0
    if pixels.mean() > 0.5:      # mostly white background? then invert it
        pixels = 1 - pixels
    return pixels.reshape(1, 28, 28, 1)


st.title("Handwritten Digit Recognizer")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run  python train_cnn.py  first.")
    st.stop()

model = load_model()
uploaded = st.file_uploader("Upload an image of one digit", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, width=150)
    probs = model.predict(prepare(image), verbose=0)[0]
    st.subheader(f"Prediction: {probs.argmax()}")
    st.write(f"Confidence: {probs.max() * 100:.1f}%")
    st.bar_chart(probs)
