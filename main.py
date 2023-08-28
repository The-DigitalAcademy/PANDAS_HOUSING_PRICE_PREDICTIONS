import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = tf.keras.models.load_model('manoko3.h5')

st.title("Loan Approval Prediction")

# Input fields for the new features
feature1 = st.number_input("longitude", value=0)
feature2 = st.number_input("latitude", value=0)
feature3 = st.number_input("housing_median_age", value=0)
feature4 = st.number_input("total_bedrooms", value=0)
feature5 = st.number_input("population", value=0)
feature6 = st.number_input("households", value=0)
feature7 = st.number_input("median_income", value=0)
feature8 = st.number_input("median_house_value", value=0)
ocean_proximity = st.selectbox("ocean_proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

# Make predictions when a button is clicked
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([feature1, feature2, feature3, feature4, feature5,
                           feature6, feature7, feature8])

    # Preprocess input_data if needed
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data.reshape(1, -1))

    # Use the loaded model to make predictions
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    st.write(f"Loan Approval Probability: {prediction[0, 0]}")
