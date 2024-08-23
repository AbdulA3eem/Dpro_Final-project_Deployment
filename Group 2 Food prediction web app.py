#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading the necessary libraries
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Loading the model and scaler
model = joblib.load('Group2 food price prediction.pkl')
scaler = joblib.load('scaler')

# State mapping dictionary
state_mapping = {
    'Abia': 1, 'Anambra': 1, 'Ebonyi': 1, 'Enugu': 1, 'Imo': 1,  # South-East
    'Akwa Ibom': 2, 'Bayelsa': 2, 'Cross River': 2, 'Delta': 2, 'Edo': 2, 'Rivers': 2,  # South-South
    'Ekiti': 3, 'Lagos': 3, 'Ogun': 3, 'Ondo': 3, 'Osun': 3, 'Oyo': 3,  # South-West
    'Adamawa': 4, 'Bauchi': 4, 'Borno': 4, 'Gombe': 4, 'Taraba': 4, 'Yobe': 4,  # North-East
    'Benue': 5, 'FCT': 5, 'Kogi': 5, 'Kwara': 5, 'Nasarawa': 5, 'Niger': 5, 'Plateau': 5,  # North-Central
    'Jigawa': 6, 'Kaduna': 6, 'Kano': 6, 'Katsina': 6, 'Kebbi': 6, 'Sokoto': 6, 'Zamfara': 6,  # North-West
}

# Function to map state input from the user
def map_state(state_name):
    return state_mapping.get(state_name, -1)  # -1 for unknown states

# Defining the target names 
target_names = [
    'Bread (350g-500g)', 'Cassava Meal (100kg)', 'Cowpeas (100kg)',
    'Gari (100kg)', 'Groundnuts (100kg)', 'Maize (100kg)',
    'Millet (100kg)', 'Rice (50kg)', 'Sorghum (100kg)'
]

# User inputs
fuel_diesel = st.number_input('Enter Diesel Fuel Price', min_value=0.0)
USD_NGN_Price = st.number_input('Enter USD-NGN Exchange Rate', min_value=0.0)
price_year = st.number_input('Enter year in full; e.g., 2024', min_value=2020)
fuel_petrol_gasoline = st.number_input('Enter Petrol Fuel Price', min_value=0.0)
state = st.selectbox('Select State', list(state_mapping.keys()))

# Mapping the state as it was done in the model
mapped_state = map_state(state)

# the input features
feature_values = [[fuel_diesel, USD_NGN_Price, price_year, fuel_petrol_gasoline, mapped_state]]

# Feature names in correct order
feature_names = ["fuel_diesel", "USD_NGN _Price", "price_year", "fuel_petrol_gasoline", "state"]

# Creating a DataFrame with the correct order and values
new_data = pd.DataFrame(feature_values, columns=feature_names)

# Scaling the features
scaled_data = scaler.transform(new_data)

# Making predictions
predictions = model.predict(scaled_data)

#  Displaying the prediction with target names
st.write("Predicted Food Prices:")
for name, prediction in zip(target_names, predictions[0]):  # Iterate through the first row of predictions
    st.write(f"{name}: {prediction:.2f}")

