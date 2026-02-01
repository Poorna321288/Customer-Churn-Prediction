import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and columns
model = joblib.load("models/churn_pipeline.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Prediction")
st.caption("XGBoost • Recall-optimized • Threshold 0.30")

st.write("Fill in customer details below.")

# ── Inputs ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 1000.0)

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.subheader("Additional details")
col_a, col_b, col_c = st.columns(3)
with col_a:
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    senior = st.checkbox("Senior Citizen", value=False)
with col_b:
    partner = st.checkbox("Has Partner", value=False)
    dependents = st.checkbox("Has Dependents", value=False)
with col_c:
    phoneservice = st.checkbox("Phone Service", value=True)
    paperless = st.checkbox("Paperless Billing", value=True)

# ── Prepare input ─────────────────────────────────────────────────────────
input_dict = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,

    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,

    'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
    'InternetService_No': 1 if internet_service == "No" else 0,

    'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,

    'gender_Male': 1 if gender == "Male" else 0,
    'SeniorCitizen': 1 if senior else 0,
    'Partner_Yes': 1 if partner else 0,
    'Dependents_Yes': 1 if dependents else 0,
    'PhoneService_Yes': 1 if phoneservice else 0,
    'PaperlessBilling_Yes': 1 if paperless else 0,
}

df_input = pd.DataFrame([input_dict])
df_input = df_input.reindex(columns=feature_columns, fill_value=0)

# ── Prediction ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn Risk", type="primary"):
    with st.spinner("Predicting..."):
        prob = model.predict_proba(df_input)[0][1]
        prediction = 1 if prob >= 0.30 else 0

    st.divider()
    if prediction == 1:
        st.error(f"### ⚠️ HIGH CHURN RISK ({prob:.1%})")
        st.write("This customer is likely to churn. Consider retention offers.")
    else:
        st.success(f"### ✅ LOW CHURN RISK ({prob:.1%})")
        st.write("This customer looks stable.")