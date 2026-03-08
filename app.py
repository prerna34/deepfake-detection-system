import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("deepfake_model.h5")

st.set_page_config(page_title="Deepfake Detector", page_icon="🧠")

st.title("🧠 Deepfake Detection System")
st.write("Upload a face image and the model will predict whether it is **Real** or **Fake**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    with col2:

        st.subheader("Prediction Result")

        if prediction > 0.5:
            st.error("⚠️ Fake Image Detected")
        else:
            st.success("✅ Real Image")

        st.write("### Confidence Score")

        confidence = float(prediction)

        if prediction > 0.5:
            st.progress(confidence)
            st.write(f"Fake Probability: {confidence:.2f}")
        else:
            st.progress(1-confidence)
            st.write(f"Real Probability: {(1-confidence):.2f}")