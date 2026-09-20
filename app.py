import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# CUSTOM DESIGN
# -----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #808080;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }

    .footer {
        text-align: center;
        color: #808080;
        margin-top: 40px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "Models" / "logistic_model.pkl")
scaler = joblib.load(BASE_DIR / "Models" / "scaler.pkl")
model_columns = joblib.load(BASE_DIR / "Models" / "model_columns.pkl")

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based telecom customer retention analysis'
    '</div>',
    unsafe_allow_html=True
)

# Information cards
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("ML Model", "Logistic Regression")

with c2:
    st.metric("Model Accuracy", "≈ 80%")

with c3:
    st.metric("Dataset Records", "7,032")

st.divider()

# -----------------------------
# CUSTOMER INPUT FORM
# -----------------------------
st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0, 72, 12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

total_charges = st.number_input(
    "Total Charges ($)",
    min_value=0.0,
    value=float(monthly_charges * tenure)
)

# -----------------------------
# PREDICTION
# -----------------------------
st.divider()

if st.button(
    "🔍 Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Create dataframe
    input_df = pd.DataFrame([customer])

    # Same encoding used during model training
    input_encoded = pd.get_dummies(input_df)

    # Match training columns
    input_encoded = input_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Scale data
    input_scaled = scaler.transform(input_encoded)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader("📈 Prediction Result")

    r1, r2 = st.columns(2)

    with r1:
        st.metric(
            "Churn Probability",
            f"{probability:.1f}%"
        )

    with r2:
        if probability < 30:
            risk = "Low Risk"
        elif probability < 60:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        st.metric("Risk Level", risk)

    st.progress(min(int(probability), 100))

    if prediction == 1:
        st.error(
            "⚠️ Customer is likely to CHURN.\n\n"
            "Retention action may be required."
        )
    else:
        st.success(
            "✅ Customer is likely to STAY.\n\n"
            "The model predicts a low likelihood of churn."
        )

# -----------------------------
# PROJECT INFORMATION
# -----------------------------
st.divider()

with st.expander("ℹ️ About This Project"):
    st.write("""
    This Customer Churn Prediction System uses machine learning
    to analyze telecom customer information and estimate the
    probability that a customer may leave the company.

    The project includes data cleaning, exploratory data analysis,
    feature preprocessing, machine learning model training,
    evaluation and web-based prediction.
    """)

    st.write("**Technologies:** Python, Pandas, Scikit-learn, Streamlit")

    st.write("**Final Model:** Logistic Regression")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    '<div class="footer">'
    'Customer Churn Analysis & Prediction • Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)