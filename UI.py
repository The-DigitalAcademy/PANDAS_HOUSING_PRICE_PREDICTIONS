import pandas as pd
import streamlit as st
import joblib

# Load the saved models
model2 = joblib.load('model2.pkl')
model3 = joblib.load('model3.pkl')

def main():
    # Title of the web app
    st.title("Model Predictions App")
    st.write("Enter the following features to get predictions:")

    # User input for features
    st.header('Feature Input')

    # Create input fields for each feature
    # For numerical features, use st.number_input()
    feature1 = st.number_input("longitude", value=0)
    feature2 = st.number_input("latitude", value=0)
    feature3 = st.number_input("housing_median_age", value=0)
    feature4 = st.number_input("total_bedrooms", value=0)
    feature5 = st.number_input("population", value=0)
    feature6 = st.number_input("households", value=0)
    feature7 = st.number_input("median_income", value=0)
    feature8 = st.number_input("median_house_value", value=0)
    feature9 = st.selectbox("ocean_proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

    # Selection box for the model to use
    selected_model = st.selectbox("Choose a model", ["model2", "model3"])

    # Button for predictions
    clicked = st.button('Get Predictions')

    # Perform predictions when the button is clicked
    if clicked:
        if selected_model == "model2":
            model = model3
        elif selected_model == "model3":
            model = model2
        else:
            st.warning("Please select a valid model.")
            return

        # Perform predictions using the selected model
        input_features = pd.DataFrame({
            "longitude": [feature1],
            "latitude": [feature2],
            "housing_median_age": [feature3],
            "total_bedrooms": [feature4],
            "population": [feature5],
            "households": [feature6],
            "median_income": [feature7],
            "median_house_value": [feature8],
            "ocean_proximity": [feature9]
        })
        prediction = model.predict(input_features)

        # Display the prediction result
        st.header('Prediction')
        st.write(f'The prediction result is: {prediction[0]}')

if __name__ == '__main__':
    main()

