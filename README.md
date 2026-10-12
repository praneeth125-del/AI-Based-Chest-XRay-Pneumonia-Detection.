# AI-Based Chest X-Ray Pneumonia Detection

A deep learning project that uses a Convolutional Neural Network (CNN) to classify chest X-ray images into **NORMAL** and **PNEUMONIA** categories.

## Features
- Upload chest X-ray images for analysis.
- Preprocess images to 224 × 224 pixels.
- Predict NORMAL or PNEUMONIA using a trained CNN.
- Display prediction results through a Streamlit web application.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- scikit-learn
- Streamlit

## Model Performance
The current model achieved **88.46% test accuracy** on 624 test images.

## Dataset
Kaggle Chest X-Ray Images (Pneumonia):  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Run Locally
1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required packages using `pip install -r requirements.txt`.
4. Obtain the trained model file `pneumonia_cnn_improved.keras` and place it in the project root.
5. Run the application using `streamlit run app.py`.

**Note:** The dataset, trained model files, and `requirements.txt` are not currently included in this repository.

## Disclaimer
This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used to make medical decisions.
