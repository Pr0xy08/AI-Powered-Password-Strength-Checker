# app.py
import streamlit as st
import joblib
import pandas as pd
from model import extract_features

# Load trained model
model = joblib.load("password_strength_classifier.pkl")

# App title
st.title("Password Strength Checker")

# Password input
password = st.text_input("Enter your password:", type="password")

if password:
    # Extract features
    features = extract_features(password)

    # Predict
    pred_class = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    # Map strength labels
    strength_labels = {0: "Weak", 1: "Medium", 2: "Strong"}
    strength = strength_labels[pred_class]

    # Show result
    st.subheader(f"Prediction: {strength}")
    st.subheader(f"Probability of Correctness: {pred_proba}%")

    # Feature insights
    st.subheader("Password Features")
    st.write(features.T)
