# OncoVision 🔬

**OncoVision** is a prototype AI diagnostic dashboard and Convolutional Neural Network (CNN) designed to automate the morphological assessment of breast histopathology slides. It classifies cell patches to detect Invasive Ductal Carcinoma (IDC).

This project features a fully functioning deep learning pipeline, a high-performance FastAPI backend, and a modern, glassmorphism-styled frontend UI.

## Features
* **Custom CNN Architecture:** Trained on the Kaggle Breast Histopathology dataset to classify H&E stained tissue patches as either *Benign* or *Malignant*.
* **FastAPI Backend (`app.py`):** A lightweight API that handles image uploads, preprocesses data, and serves the TensorFlow model predictions.
* **Out-Of-Distribution (OOD) Protection:** The backend includes a custom color-profile heuristic filter that automatically rejects non-medical images (like screenshots or random photos) that lack the standard pink/purple H&E stain signature.
* **Diagnostic UI (`code.html`):** A premium, dark-themed responsive dashboard built with TailwindCSS. Features include drag-and-drop uploads, animated confidence gauges, and automatic dynamic medical flagging.

## Tech Stack
* **Machine Learning:** TensorFlow / Keras, NumPy, Pandas, Scikit-Learn
* **Backend:** FastAPI, Uvicorn, Python-Multipart
* **Frontend:** HTML5, Vanilla JavaScript, TailwindCSS
* **Security:** SlowAPI (Rate Limiting)

## How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
If you want to train the model from scratch, run the training script. It will automatically download the required dataset from Kaggle via the `opendatasets` library.
```bash
python train_cnn.py
```

### 3. Start the Backend Server
Launch the FastAPI server to handle predictions.
```bash
python app.py
```
*The server will start on `http://127.0.0.1:8000`.*

### 4. Open the Dashboard
Simply double-click on `stitch_ocnovision_breast_cancer_detection/code.html` to open the interface in your web browser. Drag and drop a histopathology scan to see the model in action!

---
*Disclaimer: This is a prototype application built for educational and demonstration purposes. It is not intended for actual clinical diagnosis or medical use.*
