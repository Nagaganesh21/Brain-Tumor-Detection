import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page title
st.set_page_config(page_title="Brain Tumor Detection App")

# Set page title and subtitles
st.title("Hello!")
st.title("Brain Tumor Detection")

image = Image.open('C:\\Users\\MBR\\Pictures\\background\\health.jfif')
st.image(image)

loaded_model = tf.keras.models.load_model(r'C:\Users\MBR\Downloads/my_model.h5')

def preprocess_image(image):
    img_size = 224  # Specify the desired input size for the model

    # Convert image to RGB and remove alpha channel if present
    image = image.convert("RGB")

    # Resize the image to match the input shape of the model
    image = image.resize((img_size, img_size))

    # Normalize the image pixels between 0 and 1
    image = np.array(image) / 255.0

    # Add batch dimension and expand dimensions
    image = np.expand_dims(image, axis=0)

    return image

# Function to make predictions
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = loaded_model.predict(preprocessed_image)
    return prediction[0][0]

# Display file uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction on the uploaded image
    prediction = predict_image(image)
    print(prediction)

    # Display the prediction result
    if prediction < 0.5:
        st.markdown("## Prediction: Class 0")
    else:
        st.markdown("## Prediction: Class 1")

