"""
Customer Churn Prediction System
---------------------------------
A colorful, professional Streamlit app for predicting telecom customer churn
using a pre-trained Logistic Regression model.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLING — vibrant, colorful theme
# =========================================================
st.markdown("""
<style>
    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #0EA5A4 0%, #2563EB 50%, #8B5CF6 100%);
        border-radius: 20px;
        padding: 2.4rem 2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(14, 165, 164, 0.35);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    .hero-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: rgba(255,255,255,0.92);
        font-weight: 500;
    }

    /* Colorful stat chips */
    .stat-row { display: flex; gap: 1rem; margin-top: -0.6rem; margin-bottom: 1.2rem; }
    .stat-chip {
        flex: 1;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    .stat-chip .label { font-size: 0.8rem; opacity: 0.9; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .stat-chip .value { font-size: 1.5rem; font-weight: 800; margin-top: 0.15rem; }
    .chip-purple { background: linear-gradient(135deg, #0EA5A4, #5EEAD4); }
    .chip-pink   { background: linear-gradient(135deg, #2563EB, #60A5FA); }
    .chip-orange { background: linear-gradient(135deg, #EA580C, #FB923C); }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
        background: linear-gradient(90deg, #0EA5A4, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }

    /* Sidebar headers */
    section[data-testid="stSidebar"] h2 {
        background: linear-gradient(90deg, #0EA5A4, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Result cards */
    .result-card {
        padding: 1.8rem;
        border-radius: 18px;
        margin-top: 1rem;
        color: white;
        box-shadow: 0 10px 26px rgba(0,0,0,0.18);
    }
    .result-card b { font-weight: 800; }
    .result-high   { background: linear-gradient(135deg, #DC2626, #F87171); }
    .result-medium { background: linear-gradient(135deg, #D97706, #FBBF24); }
    .result-low    { background: linear-gradient(135deg, #16A34A, #4ADE80); }

    /* Colorful metric replacements inside result section */
    .metric-box {
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    .metric-box .m-label { font-size: 0.78rem; opacity: 0.9; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-box .m-value { font-size: 1.6rem; font-weight: 800; margin-top: 0.2rem; }

    .footer {
        text-align: center;
        color: #9aa0a6;
        opacity: 0.8;
        margin-top: 3rem;
        font-size: 0.85rem;
    }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0EA5A4, #2563EB);
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.7rem 0;
        box-shadow: 0 6px 18px rgba(14, 165, 164, 0.4);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #0F766E, #1D4ED8);
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #0EA5A4, #2563EB, #8B5CF6) !important;
    }
    /* =========================================
   MOBILE RESPONSIVE - iPhone + Android
   ========================================= */

/* All phones */
@media screen and (max-width: 768px) {

    /* Main page */
    .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        padding-top: 4rem !important;
        max-width: 100% !important;
    }
    /* Stack columns on mobile */
div[data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
}

div[data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
}
/* Stack top statistic cards on mobile */
.stat-row {
    flex-direction: column !important;
    width: 100% !important;
}

.stat-chip {
    width: 100% !important;
    min-width: 100% !important;
    box-sizing: border-box !important;
}

    /* Hero banner */
    .hero {
        padding: 1.3rem 0.7rem !important;
        border-radius: 14px !important;
        margin-bottom: 1rem !important;
    }

    .hero-title {
        font-size: 1.6rem !important;
        line-height: 1.2 !important;
    }

    .hero-subtitle {
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
    }

    /* Headings */
    h1 {
        font-size: 1.7rem !important;
    }

    h2 {
        font-size: 1.4rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
    }

    p {
        font-size: 0.9rem !important;
    }

    /* Buttons */
    div.stButton > button {
        width: 100% !important;
        min-height: 48px !important;
        font-size: 0.95rem !important;
        border-radius: 12px !important;
    }

    /* Input fields */
    div[data-baseweb="select"],
    div[data-baseweb="input"] {
        width: 100% !important;
    }

    /* Prevent horizontal overflow */
    html, body {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    h1, h2, h3, p {
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
    }

    /* Images */
    img {
        max-width: 100% !important;
        height: auto !important;
    }
}


/* =========================================
   SMALL iPHONES
   iPhone SE / older iPhones
   approx. 320px - 390px
   ========================================= */

@media screen and (max-width: 390px) {

    .block-container {
        padding-left: 0.55rem !important;
        padding-right: 0.55rem !important;
    }

    .hero {
        padding: 1rem 0.5rem !important;
    }

    .hero-title {
        font-size: 1.35rem !important;
    }

    .hero-subtitle {
        font-size: 0.78rem !important;
    }

    div.stButton > button {
        font-size: 0.85rem !important;
    }
}


/* =========================================
   LARGE iPHONES + ANDROID PHONES
   approx. 391px - 480px
   ========================================= */

@media screen and (min-width: 391px) and (max-width: 480px) {

    .hero-title {
        font-size: 1.55rem !important;
    }

    .hero-subtitle {
        font-size: 0.88rem !important;
    }

    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
}
/* Hide Streamlit Cloud bottom-right badges */
[data-testid="stAppDeployButton"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* Hide Streamlit viewer badges */
.viewerBadge_container__1QSob,
.viewerBadge_link__1S137 {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL FILES (cached so this only runs once)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model_dir = BASE_DIR / "Models"
    model = joblib.load(model_dir / "logistic_model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    model_columns = joblib.load(model_dir / "model_columns.pkl")
    return model, scaler, model_columns


try:
    model, scaler, model_columns = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

# =========================================================
# HERO HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">💠 Customer Churn Prediction System</div>
        <div class="hero-subtitle">🤖 Machine learning powered telecom customer retention analysis ✨</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="stat-row">
        <div class="stat-chip chip-purple"><div class="label">🤖 ML Model</div><div class="value">Logistic Regression</div></div>
        <div class="stat-chip chip-pink"><div class="label">📊 Model Accuracy</div><div class="value">≈ 80%</div></div>
        <div class="stat-chip chip-orange"><div class="label">👥 Training Records</div><div class="value">7,032</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_LOADED:
    st.error(
        "⚠️ Could not load model files. Make sure a `Models/` folder containing "
        "`logistic_model.pkl`, `scaler.pkl`, and `model_columns.pkl` sits next to this script.\n\n"
        f"Details: {LOAD_ERROR}"
    )
    st.stop()

# =========================================================
# SIDEBAR — CUSTOMER INPUT FORM
# =========================================================
with st.sidebar:
    st.header("👥✨ Customer Profile")

    with st.expander("🧬👤 Demographics 🧑‍🤝‍🧑", expanded=True):
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with st.expander("💼 Account Details 📅", expanded=True):
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=float(monthly_charges * tenure), step=1.0
        )

    with st.expander("🌐 Services 📶", expanded=True):
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    predict_clicked = st.button("🔮✨ Predict Customer Churn 🚀", type="primary", use_container_width=True)

# =========================================================
# PREDICTION LOGIC
# =========================================================
def build_customer_record():
    return {
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
        "TotalCharges": total_charges,
    }


def predict_churn(customer: dict):
    input_df = pd.DataFrame([customer])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100
    return prediction, probability


# =========================================================
# MAIN AREA
# =========================================================
if predict_clicked:
    customer = build_customer_record()
    prediction, probability = predict_churn(customer)

    if probability < 30:
        risk, risk_class, icon, chip_color = "Low Risk", "result-low", "✅", "#16A34A"
    elif probability < 60:
        risk, risk_class, icon, chip_color = "Medium Risk", "result-medium", "⚠️", "#D97706"
    else:
        risk, risk_class, icon, chip_color = "High Risk", "result-high", "🚨", "#DC2626"

    st.markdown('<div class="section-header">📈 Prediction Result 🔔</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(
            f'<div class="metric-box" style="background:linear-gradient(135deg,#0EA5A4,#5EEAD4);">'
            f'<div class="m-label">🎲 Churn Probability</div><div class="m-value">{probability:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f'<div class="metric-box" style="background:{chip_color};">'
            f'<div class="m-label">⚠️ Risk Level</div><div class="m-value">{risk}</div></div>',
            unsafe_allow_html=True,
        )
    with r3:
        pred_text = "Will Churn" if prediction == 1 else "Will Stay"
        st.markdown(
            f'<div class="metric-box" style="background:linear-gradient(135deg,#2563EB,#60A5FA);">'
            f'<div class="m-label">🔮 Prediction</div><div class="m-value">{pred_text}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.progress(min(int(probability), 100))

    verdict = (
        "This customer is likely to <b>CHURN</b>. Consider proactive retention outreach "
        "(loyalty offers, contract upgrade incentives, or a support check-in)."
        if prediction == 1
        else "This customer is likely to <b>STAY</b>. The model predicts a low likelihood of churn."
    )

    st.markdown(
        f'<div class="result-card {risk_class}">{icon}&nbsp;&nbsp;{verdict}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("🔎🗒️ View submitted customer data"):
        st.dataframe(pd.DataFrame([customer]).T.rename(columns={0: "Value"}), use_container_width=True)

else:
    st.info("👈✨ Fill in the customer profile in the sidebar, then click **🔮 Predict Customer Churn**.")

# =========================================================
# ABOUT SECTION
# =========================================================
st.divider()

with st.expander("ℹ️📘 About This Project"):
    st.write("""
This Customer Churn Prediction System uses machine learning to analyze telecom
customer information and estimate the probability that a customer will leave
the company. The project covers data cleaning, exploratory data analysis,
feature preprocessing, model training, evaluation, and this web-based prediction tool.
    """)
    st.write("🛠️ **Technologies:** Python, Pandas, Scikit-learn, Streamlit")
    st.write("🧠 **Final Model:** Logistic Regression")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer">💜 Customer Churn Analysis & Prediction • Machine Learning Project 🚀</div>',
    unsafe_allow_html=True,
)