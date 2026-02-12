import streamlit as st
import joblib
import numpy as np

st.title("⚡ Quick-Witted Fraud Detection System")

model = joblib.load("models/fraud_model.pkl")

input_data = st.text_input(
    "Enter transaction features (comma-separated)"
)

if st.button("Check Transaction"):
    values = np.array([float(x) for x in input_data.split(",")]).reshape(1, -1)
    prob = model.predict_proba(values)[0][1]

    if prob > 0.7:
        st.error(f"Fraud Detected 🚨 (Probability: {prob:.2f})")
    else:
        st.success(f"Transaction Safe ✅ (Probability: {prob:.2f})")
