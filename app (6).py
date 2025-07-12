import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("/content/xgb_house_price_model.pkl")

st.title("Property Valuation App")

# Create input fields for each feature
region = st.number_input("Region")
price_min = st.number_input("Price Min")
price_max = st.number_input("Price Max")
neighborhood = st.number_input("Neighborhood")
price_category = st.number_input("Price Category")

# Predict
if st.button("Predict"):
    features = np.array([[region, price_min, price_max, neighborhood, price_category]])
    prediction = model.predict(features)
    st.write(f"Prediction: {'price_avg' if prediction[0] == 1 else 'No Prediction'}")


