import streamlit as st
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('model2.h5')

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

# Make predictions when a button is clicked
if st.button("Predict"):
    try:
        # Prepare the input data for prediction
        input_data = np.array([feature1, feature2, feature3, feature4, feature5,
                               feature6, feature7, feature8])

        st.write("Input Data:", input_data)  # Log the input data

        # Use the loaded model to make predictions
        prediction = model.predict(np.array([input_data]))

        st.write("Raw Prediction:", prediction)  # Log the raw prediction

        # Display the prediction
        st.write(f"Loan Approval Probability: {prediction[0, 0]}")
    except Exception as e:
        st.error("An error occurred during prediction.")
        st.exception(e)  # Log the exception details
