import io
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------- CONFIG ----------------

MODEL_PATH = "pneumonia_cnn_improved.keras"
IMG_SIZE = (224, 224)
THRESHOLD = 0.50  # fixed decision threshold: PNEUMONIA if probability >= 0.50

# Edit these if you retrain the model
MODEL_STATS = {
    "Test Accuracy": "88.46%",
    "Macro F1-Score": "0.88",
    "Pneumonia Recall": "91%",
    "Architecture": "CNN",
}

st.set_page_config(
    page_title="AI Chest X-Ray Detection",
    page_icon="🩺",
    layout="wide",
)


# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model '{MODEL_PATH}'. Make sure it is in the same folder as app.py.")
    st.exception(e)
    st.stop()


# ---------------- HELPERS ----------------

def preprocess(image: Image.Image) -> np.ndarray:
    """Same preprocessing as training: resize, scale to 0-1, add batch dim."""
    arr = np.array(image.resize(IMG_SIZE)).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def looks_like_xray(image: Image.Image) -> bool:
    """Chest X-rays are essentially grayscale. Colourful photos are not X-rays."""
    arr = np.array(image.resize((128, 128))).astype("float32")
    channel_gap = np.mean(np.abs(arr[..., 0] - arr[..., 1])) + np.mean(np.abs(arr[..., 1] - arr[..., 2]))
    return channel_gap < 12.0


def find_last_conv_layer(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam(m, img_array, layer_name):
    """Grad-CAM heatmap showing regions that push the model towards PNEUMONIA."""
    grad_model = tf.keras.models.Model(
        m.inputs, [m.get_layer(layer_name).output, m.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.nn.relu(tf.reduce_sum(conv_out[0] * pooled, axis=-1))
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def jet(x):
    """Small jet-like colormap (avoids needing matplotlib)."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float):
    base = np.array(image.resize(IMG_SIZE)).astype("float32") / 255.0
    heat = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE, Image.BILINEAR)
    heat = np.array(heat).astype("float32") / 255.0
    colored = jet(heat)
    blended = (1 - alpha) * base + alpha * colored
    return np.uint8(np.clip(blended, 0, 1) * 255)


def build_report(filename, result, confidence, p_pneu, p_norm, threshold):
    return (
        "AI CHEST X-RAY ANALYSIS REPORT\n"
        "==============================\n"
        f"Date/Time   : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Image       : {filename}\n"
        f"Prediction  : {result}\n"
        f"Confidence  : {confidence:.2f}%\n"
        f"NORMAL      : {p_norm:.2f}%\n"
        f"PNEUMONIA   : {p_pneu:.2f}%\n"
        f"Threshold   : {threshold:.2f}\n\n"
        "DISCLAIMER: Educational/research use only. This is not a medical\n"
        "diagnosis. Consult a qualified medical professional.\n"
    )


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = THRESHOLD  # fixed, not user-adjustable
    show_cam = st.toggle("Show Grad-CAM heatmap", value=True)
    cam_alpha = st.slider("Heatmap strength", 0.1, 0.9, 0.45, 0.05, disabled=not show_cam)

    st.divider()
    st.header("📊 Model Performance")
    for k, v in MODEL_STATS.items():
        st.metric(k, v)

# ---------------- HEADER ----------------

st.title("🩺 AI-Based Chest X-Ray Disease Detection")
st.subheader("Deep Learning Based Pneumonia Classification System")
st.write(
    "Upload a chest X-ray image and submit it for AI-based "
    "classification as NORMAL or PNEUMONIA."
)
st.divider()

tab_analysis, tab_about = st.tabs(["🔍 Analysis", "🔧 Model Details"])

# ---------------- ANALYSIS TAB ----------------

with tab_analysis:
    uploaded_file = st.file_uploader(
        "📤 Select a chest X-ray image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload an X-ray image to begin.")
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")
            st.stop()

        if min(image.size) < 100:
            st.warning("Image resolution is very low; results may be unreliable.")
        if not looks_like_xray(image):
            st.warning(
                "This image appears to be a colour photo, not a chest X-ray. "
                "The prediction may be meaningless."
            )

        left, right = st.columns(2)
        with left:
            st.subheader("🖼️ Uploaded Image")
            st.image(image, use_container_width=True)

        submit = st.button("🔍 Submit for Analysis", type="primary", use_container_width=True)

        # Run the model only on click; keep the result so slider changes don't re-run it
        if submit:
            with st.spinner("Analyzing chest X-ray..."):
                img_array = preprocess(image)
                prob = float(model.predict(img_array, verbose=0)[0][0])

                cam_error = None
                st.session_state["heat"] = None
                layer_name = find_last_conv_layer(model)
                if layer_name is None:
                    cam_error = "No convolutional layer found for Grad-CAM."
                else:
                    try:
                        heat = make_gradcam(model, tf.convert_to_tensor(img_array), layer_name)
                        st.session_state["heat"] = heat
                    except Exception as e:
                        cam_error = f"Grad-CAM unavailable: {e}"
                        st.session_state["heat"] = None

            st.session_state["analysis"] = {
                "file": uploaded_file.name,
                "prob": prob,
                "cam_error": cam_error,
            }

        analysis = st.session_state.get("analysis")
        if analysis and analysis["file"] == uploaded_file.name:
            prob = analysis["prob"]
            p_pneu, p_norm = prob * 100, (1 - prob) * 100

            if prob >= threshold:
                result, confidence = "PNEUMONIA", p_pneu
            else:
                result, confidence = "NORMAL", p_norm

            with right:
                st.subheader("📋 Analysis Result")
                if result == "PNEUMONIA":
                    st.error(f"⚠️ Prediction: {result}")
                else:
                    st.success(f"✅ Prediction: {result}")

                st.metric("Confidence", f"{confidence:.2f}%")

                if abs(prob - threshold) < 0.10:
                    st.warning(
                        "Borderline result: the model is not confident. "
                        "Treat this prediction with extra caution."
                    )

                st.write("**Prediction probabilities**")
                st.write(f"NORMAL — {p_norm:.2f}%")
                st.progress(int(round(p_norm)))
                st.write(f"PNEUMONIA — {p_pneu:.2f}%")
                st.progress(int(round(p_pneu)))

            if show_cam:
                st.divider()
                st.subheader("🔥 Grad-CAM: Where the Model Looked")
                heat = st.session_state.get("heat")
                if heat is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(image.resize(IMG_SIZE), caption="Original", use_container_width=True)
                    with c2:
                        st.image(
                            overlay_heatmap(image, heat, cam_alpha),
                            caption="Highlighted regions (red = stronger influence towards PNEUMONIA)",
                            use_container_width=True,
                        )
                    st.caption(
                        "The heatmap shows image regions that influenced the model's output. "
                        "It is an aid for understanding the model, not a clinical finding."
                    )
                else:
                    st.info(analysis["cam_error"] or "Grad-CAM could not be generated.")

            st.download_button(
                "⬇️ Download Report (.txt)",
                data=build_report(uploaded_file.name, result, confidence, p_pneu, p_norm, threshold),
                file_name="xray_analysis_report.txt",
                mime="text/plain",
            )

# ---------------- MODEL DETAILS TAB ----------------

with tab_about:
    st.write("**Model:** Convolutional Neural Network (CNN)")
    st.write("**Input Size:** 224 × 224 pixels")
    st.write("**Classes:** NORMAL / PNEUMONIA")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Optimizer:** Adam")
    st.write("**Loss Function:** Binary Cross-Entropy")
    st.write("**Data Augmentation:** Flip, Rotation and Zoom")
    st.write("**Class Imbalance Handling:** Class Weighting")
    st.write("**Regularization:** Dropout")
    st.write("**Explainability:** Grad-CAM")

# ---------------- DISCLAIMER ----------------

st.divider()
st.warning(
    "⚠️ Medical Disclaimer: This application is developed "
    "for educational and research purposes. The prediction "
    "should not be considered a medical diagnosis or a "
    "substitute for professional medical evaluation."
)
st.caption(
    "AI-Based Chest X-Ray Disease Detection • "
    "Deep Learning • CNN • TensorFlow • Streamlit"
)