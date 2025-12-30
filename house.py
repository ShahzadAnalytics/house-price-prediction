import streamlit as st
import numpy as np
import pickle 
# import pickle  # agar aapne trained model aur scaler save kiya hai

# Load trained model and scaler
# model=pickle.load(open('house.pkl','rb'))
model = pickle.load(open(r"C:\Users\E6440\Desktop\House_price_prediction\india_1\house.pkl", "rb"))
scaler = pickle.load(open(r"C:\Users\E6440\Desktop\House_price_prediction\india_1\scaler.pkl", "rb"))
# model = pickle.load(open("house.pkl", "rb"))
# scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 House Price Prediction App")

# User inputs
square_footage = st.number_input("Square Footage (sq ft)", min_value=300, max_value=10000, value=2000)
num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2010)
lot_size = st.number_input("Lot Size (acres)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
garage_size = st.number_input("Garage Size (cars)", min_value=0, max_value=5, value=1)
neighborhood_quality = st.slider("Neighborhood Quality (1-10)", min_value=1, max_value=10, value=7)

# Predict button
if st.button("Predict House Price"):
    # Prepare input
    new_house = np.array([[square_footage, num_bedrooms, num_bathrooms, year_built,
                           lot_size, garage_size, neighborhood_quality]])
    
    # Scale input
    new_house_scaled = scaler.transform(new_house)
    
    # Prediction
    predicted_price = model.predict(new_house_scaled)[0]
    
    st.success(f"Predicted House Price: RS {predicted_price:,.2f}")
