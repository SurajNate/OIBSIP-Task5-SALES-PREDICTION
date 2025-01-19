import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set Streamlit page configuration for full width
st.set_page_config(page_title="Email Spam Detector", layout="wide")

# Loading the model
model = joblib.load('sales_prediction_surajnate_model.pkl')

# Loading the dataset
data = pd.read_csv('Advertising.csv')

# Preprocessing
data = data.drop_duplicates()  # Remove duplicate rows
data = data.drop(data.columns[0], axis=1) # Removeing First Column
data = data.fillna(data.mean())  # Fill missing values with mean

st.title('Sales Prediction using Advertising Data - [@suraj_nate](https://www.instagram.com/suraj_nate/) 👀')
st.header('Make Predictions')
st.write("Enter values for the following columns to predict Sales:")

# Input fields for user to input values
tv = st.number_input("TV Advertising Budget", min_value=0, max_value=500, value=100)
radio = st.number_input("Radio Advertising Budget", min_value=0, max_value=500, value=50)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0, max_value=500, value=60)

# Prepare the input data for prediction
input_data = np.array([[tv, radio, newspaper]])

# Make the prediction when button is clicked
if st.button('Predict Sales'):
    # Make the prediction
    predicted_sales = model.predict(input_data)
    st.success(f"Predicted Sales: **{predicted_sales[0]:.2f}**")

# Data Overview
st.header('Data Overview')

# Displaying the first 5 rows
st.subheader('Dataset:')
st.write(data)

# Displaying summary of statistics
st.subheader('Summary Statistics:')
st.write(data.describe())

# Check for missing values
st.subheader('Missing values in each column:')
st.write(data.isnull().sum())

# Check the shape of the dataset
st.write(f"Shape of the dataset (rows, columns): {data.shape}")

# Visualizations : Pairplot
st.subheader('Pairplot of the Data:')
fig = sns.pairplot(data)
st.pyplot(fig)


# Footer
st.write("---")
st.markdown('<center><a href="https://www.instagram.com/suraj_nate/" target="_blank" style="color:white;text-decoration:none">&copy; 2025 @suraj_nate All rights reserved.</a></center>', unsafe_allow_html=True)