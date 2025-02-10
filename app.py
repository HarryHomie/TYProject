import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore

TF_ENABLE_ONEDNN_OPTS=0

# Load the trained model
model = tf.keras.models.load_model("Plant_Model.h5")


# Title for the web app
st.title("Plant Nutrition detector App")


# Input fields for all the data 
soil_moisture= st.number_input("Enter soil_moisture (%):",value=20, step=1.0)
ambient_temp = st.number_input("Enter Ambient Temperature (°C):", value=25.0, step=0.1)
humidity = st.number_input("Enter Humidity (%):", value=60.0, step=0.1)
light_intensity = st.number_input("Enter Light Intensity (Lux):", value=500.0, step=1.0)
nitrogen = st.number_input("Enter Nitrogen Level:", value=30.0, step=0.1)
phosphorus = st.number_input("Enter Phosphorus Level:", value=25.0, step=0.1)
potassium = st.number_input("Enter Potassium Level:", value=20.0, step=0.1)

def get_care_advice(health_status):
    advice = {
        "Healthy": "Your plant is in great condition! Maintain current care.",
        "Moderate Stress": "Adjust watering and check for nutrient imbalances.",
        "High Stress": "Immediate attention needed! Ensure hydration, check for pests, and balance nutrients."
    }
    return advice.get(health_status, "No specific advice available.")

if st.button("Predict"):
    # Prepare the input data for the model
    input_data = np.array([[soil_moisture, ambient_temp, humidity, light_intensity, nitrogen, phosphorus, potassium]])
    prediction = model.predict(input_data)

    # Get predicted class
    predicted_label = np.argmax(prediction)
    
    # Map the prediction to the plant health status
    status_labels = ["Healthy", "Moderate Stress", "High Stress"]
    predicted_status = status_labels[predicted_label]
    
    # Get care advice
    care_advice = get_care_advice(predicted_status)

    # Display the result
    st.success(f"Predicted Plant Health Status: {predicted_status}")
    st.info(f"Care Advice: {care_advice}")

