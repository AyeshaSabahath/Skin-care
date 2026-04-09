import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("skin_disease_model.h5")

# Class labels
classes = [
    "Melanoma",
    "Nevus",
    "Basal cell carcinoma",
    "Actinic keratosis",
    "Benign keratosis",
    "Dermatofibroma",
    "Vascular lesion"
]

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    print("Prediction:", classes[class_index])

# Test
predict_image("test.jpg")