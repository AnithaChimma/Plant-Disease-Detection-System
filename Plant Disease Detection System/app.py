import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import tempfile
from PIL import Image

# Load the model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# Load and preprocess the image
def model_predict(image_path):
    img = cv2.imread(image_path)  # Read the file and convert into an array
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0  # Rescaling
    img = img.reshape(1, H, W, C)  # Reshaping

    preds = model.predict(img)
    prediction = np.argmax(preds, axis=-1)[0]
    confidence = np.max(preds)
    return prediction, confidence

# Sidebar
st.sidebar.title("Plant Disease Detection System")
st.sidebar.image(r"D:/Plant Disease Detection System/home_image.jpeg",caption= "Machine Learning Meets Mother Nature", use_container_width=True)
app_mode = st.sidebar.radio("Navigation", ["HOME", "DISEASE RECOGNITION", "ABOUT"])

# Main Page
if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True,
    )
    st.image(r"D:/Plant Disease Detection System/image1.png", caption="Healthy Plants, Healthy Life", use_container_width=True)
    st.markdown("### Features:")
    st.write("1. Real-Time Predictions: Upload images of plant leaves for disease detection and get instant results with confidence scores.")
    st.write("2. Comprehensive Disease Detection: Covers 38 different plant disease classes, including healthy crops.")
    st.write('''3. User-Friendly Interface: Empower sustainable agriculture with AI-driven insights."
''')

elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection")

    test_image = st.file_uploader("Upload an Image of a Plant Leaf:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(test_image.read())
            temp_path = temp_file.name
    
        st.image(test_image, caption="Uploaded Image", use_container_width=True)

        # Predict button
        if st.button("Predict Disease"):
            with st.spinner("Analyzing the image..."):
                result_index, confidence = model_predict(temp_path)
                try:
                    result_index, confidence = model_predict(temp_path)
                finally:
                    os.remove(temp_path)
            st.success("Analysis Complete!")

            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            predicted_label = class_name[result_index]
            st.success(f"Prediction: {predicted_label} (Confidence: {confidence:.2%})")

            # Add this part to make it clear:
            if "healthy" in predicted_label.lower():
                st.success("🟢 The plant is healthy!")
            else:
                st.error("🔴 The plant is affected by a disease!")
            
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.image(r"D:/Plant Disease Detection System/image2.png", caption="Detect Early, Protect Fully", use_container_width=True)

elif app_mode == "ABOUT":
    st.header("About This Application")
    st.write("""This application leverages deep learning to identify plant diseases from images of plant leaves,
             providing farmers with quick and accurate diagnoses to promote sustainable agriculture. The goal is to 
             empower farmers with AI-driven tools that help reduce crop losses, optimize yields, and minimize reliance 
             on harmful chemical treatments.""")
    st.markdown("### Technologies Used:")
    st.write("- **TensorFlow**: To train and deploy a state-of-the-art Convolutional Neural Network (CNN) for multi-class classification.")
    st.write("- **Streamlit**: For building an intuitive, interactive web application interface.")
    st.write("- **OpenCV**: For preprocessing images to ensure accurate predictions.")
    st.markdown("### Challenges Addressed:")
    st.write("- **Early Detection**: Detecting diseases early helps prevent the spread and save crops.")
    st.write("- **Accessibility**: Making advanced technology accessible to non-technical users.")
    st.write("- **Sustainability**: Reducing environmental impact through precise disease management.")
    st.markdown("### Developed By:")
    st.markdown("##### Anitha Chimma")
    st.write("- B.Tech in AIML")
    st.markdown("### Acknowledgments:")
    st.write("""Inspired by the drive to improve agricultural productivity through AI. Thanks to open-source contributions, 
             research papers, and datasets that made this project possible. Special acknowledgment to the agricultural
             community for their continuous efforts in ensuring food security.""")
    st.write("")
    st.write("")
    st.image(r"D:/Plant Disease Detection System/image3.png", caption="Growing Smarter, Farming Stronger", use_container_width=True)

