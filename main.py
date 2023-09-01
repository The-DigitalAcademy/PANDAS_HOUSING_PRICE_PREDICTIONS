import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = tf.keras.models.load_model('model3.h5')

# Streamlit UI
st.title("HOUSE PRICE PREDICTION")

# User input for features
st.header('Feature Input')
feature1 = st.number_input("housing_median_age", value=0)
feature2 = st.number_input("total_bedrooms", value=0)
feature3 = st.number_input("households", value=0)
feature4 = st.number_input("median_income", value=0)

scaler = StandardScaler()
# Scale and transform user input
user_input = scaler.transform([[feature1, feature2, feature3, feature4]])

# Button for predictions
clicked = st.button('Get Predictions')

# Perform predictions when the button is clicked
if clicked:
    # Perform predictions using the pre-trained model
    prediction = model.predict(user_input)

    # Display the prediction result
    st.header('Prediction')
    st.write(f'The predicted median house value is: {prediction[0][0]}')
