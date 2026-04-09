import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load trained model
model = load_model("skin_disease_model.h5")

# Disease labels (CHANGE according to your dataset classes)
class_names = [
    "Acne",
    "Eczema",
    "Psoriasis",
    "Melanoma",
    "Ringworm",
    "Healthy Skin"
]

# Page settings
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="wide"
)

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 Skin Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image or use your camera to detect possible skin conditions</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📋 Instructions")
st.sidebar.write("""
1. Upload a clear image of the skin area
2. Or use your device camera
3. Wait for prediction
4. Check confidence score
""")

st.sidebar.warning("⚠️ This is AI-based prediction and not a medical diagnosis.")

# Input method selection
option = st.radio("Choose Input Method", ["📁 Upload Image", "📷 Use Camera"])

image = None

# Upload image
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload skin image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# Camera input
elif option == "📷 Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)

# Prediction section
if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    # Convert image for prediction
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    pred = model.predict(img)
    predicted_class = np.argmax(pred)
    confidence = np.max(pred) * 100

    disease_name = class_names[predicted_class]

    with col2:
        st.subheader("🔍 Prediction Result")
        st.success(f"Detected Disease: **{disease_name}**")
        st.info(f"Confidence Score: **{confidence:.2f}%**")

        # Show all class probabilities
        st.subheader("📊 Prediction Probabilities")
        for i, disease in enumerate(class_names):
            st.progress(float(pred[0][i]))
            st.write(f"{disease}: {pred[0][i]*100:.2f}%")

    # Health suggestion
    st.subheader("💡 Suggestion")
    st.warning(f"If symptoms of **{disease_name}** persist, please consult a dermatologist.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit + TensorFlow")