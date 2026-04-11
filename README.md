# Derma-Web | AI Skin Disease Detection System

An intelligent **AI-powered web application** that detects and classifies various skin diseases from uploaded images using **Deep Learning and Computer Vision**.

This project helps users identify possible skin conditions by analyzing skin lesion images with a trained **Convolutional Neural Network (CNN)** model.

---

## Project Overview

Skin diseases are one of the most common medical issues worldwide. Early detection can help in timely treatment and reduce risks.

This system uses **Machine Learning + Deep Learning** techniques to classify skin disease images into multiple categories such as:

- Melanoma
- Basal Cell Carcinoma (BCC)
- Benign Keratosis (BKL)
- Nevus
- Vascular Lesions
- Dermatofibroma
- Actinic Keratosis

The user uploads an image, and the model predicts the disease type with confidence score.

---

## Features

- Image-based skin disease detection
- Deep Learning CNN model
- User-friendly web interface
- Real-time image upload and prediction
- Confidence score display
- Training and testing pipeline
- Preprocessing and classification support
- Multi-class disease prediction

---

## Tech Stack

### Programming Language
- Python

### Libraries / Frameworks
- TensorFlow
- Keras
- OpenCV
- NumPy
- Streamlit / Flask
- Scikit-learn

### Frontend
- HTML
- CSS
- JavaScript

---

## Project Structure

```bash
Derma-web/
│── app.py
│── Predictmodel.py
│── Trainmodel.py
│── dataset/
│   ├── train/
│   └── test/
│── skin_disease_model.h5
│── requirements.txt
│── README.md
```

---

## Dataset

The dataset is organized into training and testing folders with class-wise subfolders.

```bash
dataset/
│── train/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── melanoma/
│   ├── nevus/
│   └── vasc/
│
│── test/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── melanoma/
│   ├── nevus/
│   └── vasc/
```

---

## Model Training

Run the following command to train the model:

```bash
python Trainmodel.py
```

This will generate the trained model file:

```bash
skin_disease_model.h5
```

---

## Run the Application

Use the following command:

```bash
python app.py
```

If using Streamlit:

```bash
streamlit run app.py
```

---

## How It Works

1. User uploads skin image
2. Image is preprocessed
3. CNN model extracts features
4. Disease class is predicted
5. Result is shown with confidence score

---

## Future Enhancements

- Doctor consultation integration
- Medical report generation
- Mobile app support
- Severity detection
- Prescription recommendation
- Cloud deployment
- Real-time camera detection

---

## Use Cases

- Healthcare assistance
- Preliminary skin screening
- Dermatology support systems
- Medical college projects
- AI healthcare research

---

## Author

**Ayesha Sabahath**  
AI / ML Developer | Major Project  
GitHub: https://github.com/AyeshaSabahath
