import streamlit as st
import pickle
import numpy as np

# Load semua object
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
poly = pickle.load(open("poly.pkl", "rb"))

st.title("Prediksi Harga Rumah")

sqft_living = st.number_input("Luas Rumah (sqft)", min_value=0.0)
sqft_above = st.number_input("Luas Atas", min_value=0.0)
sqft_basement = st.number_input("Luas Basement", min_value=0.0)
bedrooms = st.number_input("Jumlah Kamar", min_value=0.0)
bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=0.0)
floors = st.number_input("Jumlah Lantai", min_value=0.0)
view = st.number_input("View (0-4)", min_value=0.0)
condition = st.number_input("Condition (1-5)", min_value=0.0)
yr_built = st.number_input("Tahun Dibangun", min_value=1900.0)
waterfront = st.number_input("Waterfront (0/1)", min_value=0.0)

if st.button("Prediksi Harga"):

    data = np.array([[
        sqft_living,
        sqft_above,
        sqft_basement,
        bedrooms,
        bathrooms,
        floors,
        view,
        condition,
        yr_built,
        waterfront
    ]])

    # Transform sama seperti training
    data_scaled = scaler.transform(data)
    data_poly = poly.transform(data_scaled)

    log_prediction = model.predict(data_poly)
    prediction = np.exp(log_prediction)  # karena kamu pakai log target

    st.success(f"Perkiraan Harga Rumah: ${prediction[0]:,.0f}")
