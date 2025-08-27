import streamlit as st
from PIL import Image
import numpy as np
import os
from model import *
from numba import cuda

# =========================
# Config
# =========================
st.set_page_config(page_title="Traffic Sign Classification", page_icon="🚦", layout="centered")

# Mapping from class index to traffic sign label
CLASS_NAMES = {
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',
    5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)',
    7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',
    9:'No passing',
    10:'No passing veh over 3.5 tons',
    11:'Right-of-way at intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'Vehicles > 3.5 tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve left',
    20:'Dangerous curve right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End speed + passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End no passing veh > 3.5 tons'
}

NUM_CLASSES = len(CLASS_NAMES)


# ===============================
# Image preprocessing
# ===============================
def preprocess_image_for_display(image: Image.Image):
    """
    Preprocess an image for display (keep as uint8).
    Args:
        image (PIL.Image): Input image.
    Returns:
        ndarray: Raw image as numpy array.
    """
    return np.array(image)

def preprocess_image_for_model(image: Image.Image):
    """
    Preprocess an image for model input.
    - Resize to 32x32
    - Normalize to [-1,1]
    - Add batch dimension
    Args:
        image (PIL.Image): Input image.
    Returns:
        ndarray: Preprocessed image (1,H,W,C).
    """
    img = image.resize((32, 32))
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return np.expand_dims(img, axis=0)

# ===============================
# Prediction wrapper
# ===============================
def predict(model, arr: np.ndarray):
    """
    Predicts labels for a batch of images.
    Args:
        model (SimpleCNN_Sequential_Vec): CNN model.
        arr (ndarray): Batch of images (N,H,W,C).
    Returns:
        list[str]: Predicted class names.
    """
    preds = model.predict(arr)
    labels = [CLASS_NAMES.get(int(i), f"Class {int(i)}") for i in preds]
    return labels


def load_params(model, filepath):
    """
    Load model parameters from a .npz file.
    Args:
        model (SimpleCNN_Sequential_Vec): CNN model.
        filepath (str): Path to the parameter file.
    """
    data = np.load(filepath)
    for name in model.params.keys():
        model.params[name] = data[name]
    print(f"[INFO] Loaded model parameters from {filepath}")

# =======================================
# Load model and parameters
# =======================================
if cuda.is_available():
    model = SimpleCNN_Final()
else:
    model = SimpleCNN_Sequential_Vec()

# Get the base directory of the current .py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Build the full path to the parameter file
param_path = os.path.join(BASE_DIR, "best_params.npz")
# Load parameters into the model
load_params(model, param_path)

# ===============================
# Streamlit App
# ===============================
st.title("🚦 Traffic Sign Classification")

# Initialize session state
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "allow_predict" not in st.session_state:
    st.session_state.allow_predict = False

# File uploader (key changes when reset)
uploaded_files = st.file_uploader(
    "Upload Traffic Sign Images",
    type=["jpg", "jpeg", "png", "ppm"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key
)

if uploaded_files:
    images_display, images_model = [], []

    # Preprocess all uploaded images
    for uploaded in uploaded_files:
        image = Image.open(uploaded).convert("RGB")
        images_display.append(preprocess_image_for_display(image))
        images_model.append(preprocess_image_for_model(image))

    arr_batch = np.vstack(images_model)

    # Add classify button
    if not st.session_state.allow_predict:
        if st.button("Classify Images"):
            st.session_state.allow_predict = True

    # Perform prediction
    if st.session_state.allow_predict:
        with st.spinner("Processing and predicting..."):
            results = predict(model, arr_batch)

        col = st.columns([4, 1])
        with col[0]:
            st.subheader("📌 Classification Results")
        with col[1]:
            clear_button = st.button("Clear all")

        if clear_button:
            # Reset state and uploader
            st.session_state.allow_predict = False
            st.session_state.uploader_key += 1
            st.rerun()

        # Display results in a grid (4 images per row)
        cols_per_row = 4
        cols = st.columns(cols_per_row)

        for i, (img, label) in enumerate(zip(images_display, results)):
            col = cols[i % cols_per_row]
            with col:
                st.image(img, use_container_width=True)
                st.markdown(
                    f"""
                    <div style='text-align:center;
                                font-size:14px;
                                font-weight:bold;
                                color:green;
                                margin-top:3px;
                                background-color:#F0F8FF;
                                padding:4px;
                                border-radius:6px;'>
                        {label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Add spacing
            st.markdown("<div style='margin-bottom:1px;'></div>", unsafe_allow_html=True)
            
            if (i + 1) % cols_per_row == 0 and (i + 1) < len(results):
                cols = st.columns(cols_per_row)
