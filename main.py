import streamlit as st
import joblib
import pandas as pd
from model import extract_features

# Load trained model
model = joblib.load("password_strength_classifier.pkl")

# title
st.set_page_config(page_title="Password Strength Checker", layout="centered")
st.title("Password Strength Checker")
st.write("Check the strength of your password using a machine learning model and get tips to improve it.")

# Password input
password = st.text_input("Enter your password:", type="password", placeholder="Enter a password...")

if password:
    # Extract features
    features = extract_features(password)

    # Predict
    pred_class = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    # Map strength labels
    strength_labels = {0: "Weak", 1: "Medium", 2: "Strong"}
    strength_colors = {"Weak": "red", "Medium": "orange", "Strong": "green"}
    strength = strength_labels[pred_class]

    # show predication
    st.markdown("---")
    st.subheader("Prediction Result:")

    st.markdown(
        f"**Strength:** <span style='color:{strength_colors[strength]}; font-size:20px;'>{strength}</span>",
        unsafe_allow_html=True,
    )

    st.progress(int(pred_proba[pred_class] * 100))

    st.caption(f"Model confidence: **{pred_proba[pred_class]*100:.2f}%**")

    st.caption(f"Password entropy: **{features["shannon_entropy"].iloc[0]} bits**")

    # Feature Insights
    with st.expander("View Extracted Password Features:"):
        st.write(features.T)

    # Suggestions
    tips = []
    if features["length"].iloc[0] < 8:
        tips.append("Increase the length to at least **8 characters**")
    if not features["is_mixed_case"].iloc[0]:
        tips.append("Use a mix of **uppercase and lowercase letters**")
    if not features["is_alphanum"].iloc[0]:
        tips.append("Include a mix of **letters and numbers**")
    if not features["special_count"].iloc[0]:
        tips.append("Add **special characters** (!, @, #, etc.)")
    if not features["digit_count"].iloc[0]:
        tips.append("Include at least **one digit**")
    if features["is_common_password"].iloc[0]:
        tips.append("This is a **common password** – avoid using it!")
    if features["contains_year"].iloc[0]:
        tips.append("Remove obvious **years or dates** from your password")

    if tips:
        st.subheader("Suggestions to Improve Your Password: ")
        for t in tips:
            st.markdown(f"- {t}")
    else:
        st.success("Great job! Your password is strong.")

# Feature Importance
if st.button("Show Model Feature Importance Graph:"):
    st.image("feature_importance.png", caption="Feature Importance (LightGBM)", use_container_width=True)
    st.write("This graph shows which features the model has learnt to prioritise the most.")
