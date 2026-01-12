import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from PIL import Image
import tensorflow as tf

# --- Sidebar Navigation ---
st.sidebar.title("Plant IQ Navigation")
app_mode = st.sidebar.radio("Choose the app", ["Crop Recommendation", "Plant Disease Classifier"])

# --- Crop Recommendation System ---
if app_mode == "Crop Recommendation":
    st.set_page_config(page_title="Crop Recommendation System", page_icon="🌿", layout="wide")

    st.title("🌾 Crop Recommendation System")
    st.markdown("Enter the soil and weather conditions to get a recommendation for the most suitable crop to cultivate.")

    # Use absolute path relative to the current script
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'NBClassifier.pkl')

    def load_model(path):
        try:
            with open(path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            st.error(f"Model loading error: {e}")
            return None

    model = load_model(model_path)

    # Sidebar input
    st.sidebar.header("Input Parameters")
    def user_input_features():
        n = st.sidebar.number_input('Nitrogen (N)', 5, 140, 50)
        p = st.sidebar.number_input('Phosphorus (P)', 5, 145, 50)
        k = st.sidebar.number_input('Potassium (K)', 5, 300, 50)
        temp = st.sidebar.number_input('Temperature (°C)', 9.0, 43.0, 25.0, step=0.1)
        humidity = st.sidebar.number_input('Humidity (%)', 15.0, 99.0, 70.0, step=0.1)
        ph = st.sidebar.number_input('pH Level', 3.6, 9.9, 6.5, step=0.1)
        rainfall = st.sidebar.number_input('Rainfall (mm)', 21.0, 298.0, 100.0, step=0.1)
        data = {'N': n, 'P': p, 'K': k, 'temperature': temp, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()
    st.subheader("Your Input Parameters")
    st.dataframe(input_df, use_container_width=True)

    # Prediction
    if model:
        if st.button("Get Crop Recommendation"):
            try:
                prediction = model.predict(input_df.values)
                st.success(f"**Recommended crop: {prediction[0].capitalize()}**")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.warning("Model not loaded. Check path.")

    st.markdown("---")
    st.info("**Disclaimer:** Always consult local agricultural experts.")

# --- Plant Disease Classifier ---
elif app_mode == "Plant Disease Classifier":
    st.set_page_config(page_title="Plant Disease Classifier", page_icon="🪴", layout="wide")

    st.title("Plant Disease Classifier")

    # Paths
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(working_dir, "models", "best_mobilenet_model.keras")
    class_indices_path = os.path.join(working_dir, "class_indices.json")

    # Load model and class indices
    model = tf.keras.models.load_model(model_path)
    class_indices = json.load(open(class_indices_path))

    # Functions
    def load_and_preprocess_image(image, target_size=(224, 224)):
        img = Image.open(image).resize(target_size)
        img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
        return img_array

    def predict_image_class(model, image, class_indices):
        img_array = load_and_preprocess_image(image)
        pred_index = np.argmax(model.predict(img_array), axis=1)[0]
        return class_indices[str(pred_index)].replace("_", " ").strip()

    # File upload
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, width=150)
        with col2:
            if st.button('Classify'):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {prediction}')
