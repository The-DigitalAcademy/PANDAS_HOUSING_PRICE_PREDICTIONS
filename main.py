import pandas as pd
import streamlit as st
import h5py
from keras.models import load_model  # Assuming Keras model was used

import pandas as pd
import streamlit as st
import requests
from keras.models import load_model

# Function to download models from GitHub
def download_models():
    model2_url = 'https://github.com/The-DigitalAcademy/PANDAS_HOUSING_PRICE_PREDICTIONS/raw/main/model2.h5'
    model3_url = 'https://github.com/The-DigitalAcademy/PANDAS_HOUSING_PRICE_PREDICTIONS/raw/main/model3.h5'
    
    response_model2 = requests.get(model2_url)
    response_model3 = requests.get(model3_url)
    
    with open('model2.h5', 'wb') as f:
        f.write(response_model2.content)
    
    with open('model3.h5', 'wb') as f:
        f.write(response_model3.content)

def load_models():
    download_models()
    model2 = load_model('model2.h5')
    model3 = load_model('model3.h5')
    
    return model2, model3


# Rest of the code remains the same


def main():
    # Title of the web app
    st.title("Model Predictions App")
    st.write("Enter the following features to get predictions:")

    # Load the models
    model2, model3 = load_models()

    # User input for features
    st.header('Feature Input')
    
    # Create input fields for each feature
    feature1 = st.number_input("longitude", value=0)
    feature2 = st.number_input("latitude", value=0)
    feature3 = st.number_input("housing_median_age", value=0)
    feature4 = st.number_input("total_bedrooms", value=0)
    feature5 = st.number_input("population", value=0)
    feature6 = st.number_input("households", value=0)
    feature7 = st.number_input("median_income", value=0)
    feature8 = st.number_input("median_house_value", value=0)
    feature9 = st.selectbox("ocean_proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])  # Dropdown for ocean_proximity
    
    # Selection box for the model to use
    selected_model = st.selectbox("Choose a model", ["model2", "model3"])  # Corrected typo

    # Button for predictions
    clicked = st.button('Get Predictions')

    # Perform predictions when the button is clicked
    import numpy as np

# ...

    if clicked:
        if selected_model == "model2":
            model = model2
        elif selected_model == "model3":
            model = model3

        # Explicitly convert input features to NumPy arrays
        input_features = np.array([
            feature1, feature2, feature3, feature4,
            feature5, feature6, feature7, feature8
        ], dtype=np.float32)

        # Reshape input features to match the model's input shape
        input_features = input_features.reshape(1, -1)

        # Perform predictions using the selected model
        prediction = model.predict(input_features)

        # Display the prediction result
        st.header('Prediction')
        st.write(f'The prediction result is: {prediction[0]}')

# ...

        prediction = model.predict(input_features)

        # Display the prediction result
        st.header('Prediction')
        st.write(f'The prediction result is: {prediction[0]}')

if __name__ == '__main__':
    main()
