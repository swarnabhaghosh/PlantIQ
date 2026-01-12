import streamlit as st

st.set_page_config(
    page_title="PlantIQ",
    page_icon="🌿",
    layout="wide",
)

import pandas as pd
import numpy as np
import pickle
import os
import json
from PIL import Image
import tensorflow as tf

# ---- Sidebar Navigation ----
st.sidebar.title("Plant IQ Navigation")
app_mode = st.sidebar.radio(
    "Choose the app",
    ["Crop Recommendation", "Plant Disease Classifier"]
)

# Crop Recommendation
if app_mode == "Crop Recommendation":
    st.title("🌾 Crop Recommendation System")
    st.markdown(
        "Enter the soil and weather conditions to get a recommendation "
        "for the most suitable crop to cultivate."
    )

    # Load model
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "NBClassifier.pkl"
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    def soil_validation(input_df):
        N = input_df.loc[0, "N"]
        P = input_df.loc[0, "P"]
        K = input_df.loc[0, "K"]
        ph = input_df.loc[0, "ph"]
        rainfall = input_df.loc[0, "rainfall"]

        if ph < 4.5 or ph > 9.0:
            return False, "❌ Extreme pH detected. Soil not suitable."

        if N < 10 or P < 10 or K < 10:
            return False, "❌ Very low soil nutrients."

        if rainfall < 20:
            return False, "❌ Insufficient rainfall."

        return True, ""

    # Sidebar inputs
    st.sidebar.header("Input Parameters")

    def user_input_features():
        return pd.DataFrame({
            "N": [st.sidebar.number_input("Nitrogen (N)", 0, 140, 50)],
            "P": [st.sidebar.number_input("Phosphorus (P)", 0, 145, 50)],
            "K": [st.sidebar.number_input("Potassium (K)", 0, 300, 50)],
            "temperature": [st.sidebar.number_input("Temperature (°C)", 9.0, 43.0, 25.0)],
            "humidity": [st.sidebar.number_input("Humidity (%)", 15.0, 99.0, 70.0)],
            "ph": [st.sidebar.number_input("pH Level", 3.6, 9.9, 6.5)],
            "rainfall": [st.sidebar.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)],
        })

    input_df = user_input_features()
    st.dataframe(input_df, use_container_width=True)

    if st.button("Get Crop Recommendation"):
        valid, msg = soil_validation(input_df)
        if not valid:
            st.error(msg)
        else:
            pred = model.predict(input_df.values)[0]
            st.success(f"🌱 Recommended crop: **{pred.capitalize()}**")

    st.markdown("---") 
    st.info("**Disclaimer:** Always consult local agricultural experts.")

# Plant Disease Classifier
elif app_mode == "Plant Disease Classifier":
    st.title("🪴 Plant Disease Classifier")

    model_path = os.path.join("models", "best_mobilenet_model.keras")
    class_indices = json.load(open("class_indices.json"))

    model = tf.keras.models.load_model(model_path)

    def preprocess(img):
        img = Image.open(img).resize((224, 224))
        return np.expand_dims(np.array(img) / 255.0, axis=0)

    file = st.file_uploader("Upload plant image", ["jpg", "png", "jpeg"])

    if file and st.button("Classify"):
        pred = np.argmax(model.predict(preprocess(file)), axis=1)[0]
        st.success(f"🧪 Prediction: {class_indices[str(pred)]}")
