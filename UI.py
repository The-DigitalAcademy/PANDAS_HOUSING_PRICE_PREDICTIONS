import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the pre-trained model
model = tf.keras.models.load_model('model2 (1).h5')

# Streamlit UI
st.title("HOUSE PRICE PREDICTION")

# User input for features
st.header('Feature Input')
feature1 = st.number_input("housing_median_age", value=0)
feature2 = st.number_input("total_bedrooms", value=0)
feature3 = st.number_input("households", value=0)
feature4 = st.number_input("median_income", value=0.0)
feature5 = st.selectbox("ocean_proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

# Button for predictions
clicked = st.button('Get Predictions')

# Perform predictions when the button is clicked
if clicked:
    # Preprocess the categorical feature
    ocean_proximity_mapping = {
        "<1H OCEAN": 1,
        "INLAND": 2,
        "NEAR OCEAN": 3,
        "NEAR BAY": 0,
        "ISLAND": 4
    }
    feature5_encoded = ocean_proximity_mapping[feature5]

    # Prepare the input for prediction
    input_features = np.array([[feature1, feature2, feature3, feature4, feature5_encoded]])

    prediction = model.predict(input_features)

    # Round the prediction to two decimal places
    rounded_prediction = round(prediction[0][0], 2)

    # Multiply the rounded prediction by 2
    multiplied_prediction = rounded_prediction * 2

    # Display the multiplied prediction result
    st.header('Prediction')
    st.write(f'The predicted house price is: {multiplied_prediction}')
