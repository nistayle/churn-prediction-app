import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# 📌 Load Model dan Scaler
model = pickle.load(open('model_churn.pkl', 'rb'))
scaler = pickle.load(open('scaler_churn.pkl', 'rb'))

# 📌 Judul Web App
st.title("Customer Churn Prediction App")

# 📌 Form Input Data Pelanggan
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 60, 30)
tenure = st.slider("Tenure (tahun)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3])
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", 30000.0, 150000.0, 75000.0)

# 📌 Convert Input ke Bentuk Array
if st.button("Predict Churn Probability"):
    gender = 1 if gender == "Male" else 0
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0

    data_baru = np.array([[gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])

    # Scaling
    data_baru_scaled = scaler.transform(data_baru)

    # Predict
    prob = model.predict_proba(data_baru_scaled)[:, 1]

    st.success(f"Prediksi Probabilitas Churn: {prob[0]*100:.2f}%")
