import os
import streamlit as st
from PIL import Image
import torch
import base64
from model.model import load_model, predict_and_visualize
from utils.preprocess import preprocess_image

st.set_page_config(page_title="TheEyeCatchers", layout="centered", page_icon="🖼️")

st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: pink;
    }
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        color: #90EE90;
        font-size: 24px;
        padding: 15px 0;
        background-color: black;
    }
    img.logo {
        width: 50px;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model, class_names = load_model()

logo_path = r"C:\Users\Fatima Noor\Downloads\DR_Detection\Logo.png"  
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_image = f.read()
    logo_base64 = base64.b64encode(logo_image).decode("utf-8")
    st.markdown(
        f"""
        <div class="title-container">
            <img src="data:image/png;base64,{logo_base64}" class="logo">
            <h1>TheEyeCatchers</h1>
        </div>
        <h2 style="text-align: center;">Diabetic Retinopathy Detection with Grad-CAM</h2>
        """,
        unsafe_allow_html=True,
    )
else:
    st.error("Logo file not found. Please ensure it's placed in the correct directory.")

uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(image_path).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = preprocess_image(image)
    
    predicted_label, _, grad_cam_image = predict_and_visualize(model, input_tensor, image_path, class_names)

    st.subheader("Prediction:")
    prediction_text = "Diabetic Retinopathy Detected" if predicted_label == "DR" else "No Diabetic Retinopathy Detected"
    st.write(f"**{prediction_text}**")

    st.subheader("Grad-CAM Visualization:")
    st.image(grad_cam_image, caption="Grad-CAM Overlay", use_container_width=True)

st.markdown(
    """
    <div class="footer">
        © 2024 TheEyeCatchers: Research, Development, & Innovation
    </div>
    """,
    unsafe_allow_html=True,
)
