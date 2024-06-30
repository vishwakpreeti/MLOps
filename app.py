import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the trained model (assuming it's saved in MLflow)
logged_model = 'model.pkl'
model = joblib.load(logged_model)

# Load model
# model = mlflow.sklearn.load_model(logged_model)

# Define categorical features and their distinct values
categorical_features = ['employment_type', 'job_category', 'experience_level',
                        'employee_residence', 'remote_ratio', 'company_location', 'company_size']

distinct_values = {
    'experience_level': ['Senior-level/Expert','Mid-level/Intermediate', 'Entry-level/Junior',
 'Executive-level/Director'],  # Replace with actual distinct values
    'employment_type': ['Full-time', 'Contractor', 'Freelancer', 'Part-time'],  # Replace with actual distinct values
    'employee_residence': ['ES', 'US', 'CA', 'DE', 'GB', 'NG', 'IN', 'HK', 'PT', 'NL', 'CH', 'CF', 'FR', 'AU',
 'FI', 'UA', 'IE', 'IL', 'GH', 'AT', 'CO', 'SG', 'SE', 'SI', 'MX', 'UZ', 'BR', 'TH',
 'HR', 'PL', 'KW', 'VN', 'CY', 'AR', 'AM', 'BA', 'KE', 'GR', 'MK', 'LV', 'RO', 'PK',
 'IT', 'MA', 'LT', 'BE', 'AS', 'IR', 'HU', 'SK', 'CN', 'CZ', 'CR', 'TR', 'CL', 'PR',
 'DK', 'BO', 'PH', 'DO', 'EG', 'ID', 'AE', 'MY', 'JP', 'EE', 'HN', 'TN', 'RU', 'DZ',
 'IQ', 'BG', 'JE', 'RS', 'NZ', 'MD', 'LU', 'MT'],  # Replace with actual distinct values
    'remote_ratio': ['Full-Remote', 'On-Site', 'Half-Remote'],  # Replace with actual distinct values
    'company_location': ['ES', 'US', 'CA', 'DE', 'GB', 'NG', 'IN', 'HK', 'NL', 'CH', 'CF', 'FR', 'FI', 'UA',
 'IE', 'IL', 'GH', 'CO', 'SG', 'AU', 'SE', 'SI', 'MX', 'BR', 'PT', 'RU', 'TH', 'HR',
 'VN', 'EE', 'AM', 'BA', 'KE', 'GR', 'MK', 'LV', 'RO', 'PK', 'IT', 'MA', 'PL', 'AL',
 'AR', 'LT', 'AS', 'CR', 'IR', 'BS', 'HU', 'AT', 'SK', 'CZ', 'TR', 'PR', 'DK', 'BO',
 'PH', 'BE', 'ID', 'EG', 'AE', 'LU', 'MY', 'HN', 'JP', 'DZ', 'IQ', 'CN', 'NZ', 'CL',
 'MD', 'MT'],  # Replace with actual distinct values
    'company_size': ['LARGE', 'SMALL', 'MEDIUM'],  # Replace with actual distinct values
    'job_category': ['Other', 'Machine Learning', 'Data Science', 'Data Engineering',
 'Data Architecture', 'Management']  # Replace with actual distinct values
}

# Load the label encoders for each categorical feature
encoders = {feature: LabelEncoder().fit(values) for feature, values in distinct_values.items()}

# Streamlit app
st.title("Salary Prediction")

# User input
user_input = {}
for feature in categorical_features:
    # user_input[feature] = st.selectbox(f"Select {feature}", distinct_values[feature])
    user_input[feature] = st.selectbox(f"Select {feature}",distinct_values[feature])

# Encode the user input
encoded_input = [encoders[feature].transform([user_input[feature]])[0] for feature in categorical_features]

# Prediction
if st.button("Predict Salary Range"):
    encoded_input = np.array(encoded_input).reshape(1, -1)
    prediction = model.predict(encoded_input)

    # Decoding the prediction (if the output is encoded)
    salary_labels = ['low', 'low-mid', 'mid', 'mid-high', 'high', 'very-high', 'Top']
    # st.write(f"Predicted Salary Range: {salary_labels[prediction[0]]}")
    st.write(f"Predicted Salary Range: {prediction}")
 