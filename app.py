import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image
import io

# Download the trained model
file_id = "1rf0ixYg3vvgxh03vmx4JagRP14y7jSua"
url = "https://drive.google.com/uc?id=" + file_id
model_path = "trained_plant_disase_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

model_path = "trained_plant_disase_model.keras"

def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            # Reading Labels
            class_name = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
