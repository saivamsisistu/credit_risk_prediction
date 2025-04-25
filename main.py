# main.py
import streamlit as st
import pandas as pd
import pickle
from process import process_input

# Load model and scaler
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

selected_features = [
    'Checking account', 'Saving accounts', 'Has_Saving_Account', 'Job',
    'Housing_own', 'Housing_rent', 'Credit amount', 'Duration', 'Credit_Age_Ratio'
]

st.title("Credit Risk Prediction App")

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    job_options = {
        "Unskilled and non-resident": 0,
        "Unskilled and resident": 1,
        "Skilled": 2,
        "Highly skilled": 3
    }

    # Show job names in UI
    job_label = st.selectbox("Job Type", list(job_options.keys()))

    # Get corresponding number
    job = job_options[job_label]
    housing = st.selectbox("Housing", ["own", "rent", "free"])
    saving = st.selectbox("Saving Account", ["unknown", "little", "moderate", "quite rich", "rich"])
    checking = st.selectbox("Checking Account", ["unknown", "little", "moderate", "rich"])
    credit_amount = st.number_input("Credit Amount", min_value=0, step=100)
    duration = st.number_input("Duration (in months)", min_value=1)
    purpose = st.selectbox("Purpose", ["radio/tv", "education", "furniture/equipment", "car", "business", "domestic appliance", "repairs", "vacation", "retraining", "others"])

    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = {
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving],
        'Checking account': [checking],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [purpose]
    }

    input_df = pd.DataFrame(user_input)
    processed = process_input(input_df, scaler, selected_features)
    prediction = model.predict(processed)

    st.success(f"Prediction: {'Good Credit Risk' if prediction[0] == 1 else 'Bad Credit Risk'}")
