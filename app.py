import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model("skin_disease_model.h5")

st.title("Skin Disease Detection")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (224,224)) / 255.0
    img = np.reshape(img, (1,224,224,3))

    pred = model.predict(img)
    st.write("Prediction:", np.argmax(pred))