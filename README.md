# Housing Price Prediction Project

This repository contains code and analysis for a housing price prediction project. The goal of this project is to analyze a dataset of housing-related features and build a deep learning model to predict median house prices.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Analysis](#analysis)
- [Deep Learning Model](#deep-learning-model)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Project Overview

In this project, we aim to predict median house prices based on various features using a deep learning model. The project includes data preprocessing, exploratory data analysis, feature engineering, model building, and analysis of results.

## Data

The dataset used for this project includes features like housing age, total rooms, total bedrooms, population, households, median income, and ocean proximity. The dataset was cleaned and preprocessed to prepare it for analysis and modeling.

## Analysis

The project involved the following steps:

1. **Data Preprocessing**: Categorical variables like "ocean proximity" were mapped to numerical values. Features were scaled using StandardScaler to ensure consistent training.

2. **Exploratory Data Analysis**: Visualizations were created to explore the relationships between different features and median house prices. Insights were gained regarding the influence of factors like location, income, and age on housing values.

3. **Feature Selection**: Features were selected based on their relevance to the prediction task. SelectKBest with f_regression was used to choose the top features.

4. **Deep Learning Model**: A deep learning model with regularization (dropout and L2 regularization) was built using Keras. The model's architecture was designed with multiple layers to capture complex relationships in the data.

5. **Model Training and Evaluation**: The model was trained on the selected features and evaluated on a validation set. Different hyperparameters were tested to achieve optimal performance.

## Deep Learning Model

The deep learning model was built using Keras with the TensorFlow backend. The architecture includes several dense layers with dropout and L2 regularization. The model aims to predict median house prices based on the selected features.

## Usage

To run the code and reproduce the analysis:

1. Clone this repository.
2. Ensure you have the required libraries installed (Keras, TensorFlow, scikit-learn, etc.).
3. Run the provided Python scripts in your preferred development environment.

## Conclusion

Through data analysis and the deep learning model, we aimed to predict median house prices based on a set of features. The project involved thorough preprocessing, exploratory analysis, and model building. The results provided insights into the factors that influence housing values and demonstrated the potential of deep learning for predictive tasks in the real estate domain.



