# main.py
import joblib
from model import extract_features

# Load trained model
model = joblib.load("password_strength_classifier.pkl")

# Example usage
pwd = input("Enter a password: ")
features = extract_features(pwd)
prediction = model.predict(features)[0]

print(f"Predicted strength for '{pwd}': {prediction}")

