
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Title and Header
st.title("Skin Cancer Detection")
st.markdown("### Upload an image to detect skin cancer using AI.")

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resizing to model input size
    image = np.array(image) / 255.0  # Normalizing
    image = np.expand_dims(image, axis=0)  # Adding batch dimension
    return image

# Uploading Image
uploaded_file = st.file_uploader("Upload an image (JPG/PNG format)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Load the trained model (update model path)
    try:
        model = tf.keras.models.load_model('skin_cancer_detection_model.h5')  # Replace with actual model path
        prediction = model.predict(preprocessed_image)
        class_names = ["Benign", "Malignant"]  # Replace with your actual class names
        result = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        st.error("Error loading the model. Please ensure the model file is present and correct.")
        st.error(str(e))

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit.")
