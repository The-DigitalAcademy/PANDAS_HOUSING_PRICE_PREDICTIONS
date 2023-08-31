import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

Load the pre-trained model
model = tf.keras.models.load_model('model2.h5')

st.title("HOUSE PRICE PRIDICTION")

User input for features
st.header('Feature Input')
    feature1 = st.number_input("longitude", value=0)
    feature2 = st.number_input("latitude", value=0)
    feature3 = st.number_input("housing_median_age", value=0)
    feature4 = st.number_input("total_rooms", value=0)
    feature5 = st.number_input("total_bedrooms", value=0)
    feature6 = st.number_input("population", value=0)
    feature7 = st.number_input("households", value=0)
    feature8 = st.number_input("median_income", value=0)
    feature9 = st.selectbox("ocean_proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])



#Button for predictions
clicked = st.button('Get Predictions')

    # Perform predictions when the button is clicked
if clicked:
        # Perform predictions using the selected model
    prediction = model.predict([[feature1, feature2, feature3, feature4, feature5]])

        # Display the prediction result
    st.header('Prediction')
    st.write(f'The prediction result is: {prediction[0]}')

#if name == 'main':
    #main()

