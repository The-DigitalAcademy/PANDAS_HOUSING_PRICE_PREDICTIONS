import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

# Load your trained model (replace with the path to your model file)
model = tf.keras.models.load_model('model3 (1).h5')

# Load your scaler (replace with the path to your scaler file)
scaler = joblib.load('scaler.pkl')

# Define column names in the same order as your training data
columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']

# Set page configuration and title
st.title("Loan Approval Prediction")

# Sidebar
with st.sidebar:
    # Add options in the sidebar to navigate to different pages
    page_selection = st.selectbox("Navigation", ["Project Overview", "Loan Approval Prediction"])

# Define a function to display the "Overview" page
def project_overview():
    # Set page configuration and title
    st.title("Loan Approval Prediction")
    st.title("Project Overview")

    st.write("This project is aimed at predicting loan approval using machine learning.")
    st.write("It uses a deep learning model to predict the loan amount, which can be used to make a decision about loan approval.")
    st.write("Please navigate to other pages for more details about the team and predictions.")

# Main content
if page_selection == "Loan Approval Prediction":
    # Create a DataFrame from the input variables
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = [0] * len(columns)  # Initialize with zeros, you can replace these with your desired default values

    # Create input fields for each column
    for column in columns:
        input_df[column] = st.number_input(f"{column.replace('_', ' ').title()}", value=input_df[column].values[0])

    # Make predictions when a button is clicked
    if st.button("Predict"):
        # Standardize the input data using the loaded scaler
        input_data = input_df.values  # Convert DataFrame to array
        input_data = scaler.transform(input_data)

        # Use the loaded model to make predictions
        prediction = model.predict(input_data)
        
        # Assuming the prediction is a single continuous value representing loan amount
        predicted_loan_amount = prediction[0]  # Adjust this according to your regression target
        
        # Display the prediction result
        st.write(f"Predicted House Price: {predicted_House_Price:.2f}")
        # You can add additional logic here to determine loan approval based on the predicted amount
elif page_selection == "Project Overview":
    project_overview()

