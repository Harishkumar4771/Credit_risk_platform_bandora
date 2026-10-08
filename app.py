import streamlit as st
import pandas as pd
import numpy as np
import joblib
import loan_utils as lu

st.set_page_config(page_title="Bondora Default Predictor", layout="wide", page_icon="🔮")

# Custom CSS for styling
st.markdown("""
<style>
/* Global settings */
.main {
    background-color: #0E1117;
    color: #FAFAFA;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    color: #E2E8F0;
    font-weight: 700;
}
/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 24px;
    width: 100%;
    border: none;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(118, 75, 162, 0.4);
}
/* Metric cards */
.metric-card {
    background: rgba(38, 39, 48, 0.6);
    backdrop-filter: blur(10px);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    transition: transform 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
}
.risk-high {
    color: #ff4b4b;
    font-size: 32px;
    font-weight: 800;
    text-shadow: 0 0 10px rgba(255, 75, 75, 0.3);
}
.risk-low {
    color: #00cc96;
    font-size: 32px;
    font-weight: 800;
    text-shadow: 0 0 10px rgba(0, 204, 150, 0.3);
}
/* Architecture Box */
.arch-box {
    background: #1E2129;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #764ba2;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔮 Bondora AI Credit Risk Platform")
st.markdown("Advanced machine learning platform for assessing loan applicant default probabilities. Powered by automated feature engineering and ensemble models.")

@st.cache_resource
def load_models():
    models = {}
    try: models["XGBoost (Best Model)"] = joblib.load("artifacts/xgboost_pipeline.joblib")
    except: pass
    try: models["LightGBM"] = joblib.load("artifacts/lightgbm_pipeline.joblib")
    except: pass
    try: models["Random Forest"] = joblib.load("artifacts/random_forest_pipeline.joblib")
    except: pass
    return models

@st.cache_data
def load_sample_data():
    return pd.read_csv("LoanData_Sample.csv", nrows=5000, low_memory=False)

try:
    models = load_models()
    sample_df = load_sample_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# --- Tabs ---
tab_predict, tab_compare, tab_architecture = st.tabs(["📊 Prediction Engine", "📈 Model Comparison", "🏗️ System Architecture"])

with tab_predict:
    st.sidebar.header("Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose the ML Model to use:", 
        list(models.keys()), 
        index=0  # Defaults to XGBoost
    )
    selected_model = models[selected_model_name]

    # --- Sidebar: Choose Input Method ---
    st.sidebar.header("Input Data Generation")
    input_method = st.sidebar.radio("How would you like to provide data?", ["Enter Manually", "Select Random Applicant"])

    if input_method == "Select Random Applicant":
        if st.sidebar.button("🎲 Auto-Fill Random Applicant"):
            st.session_state['random_row'] = sample_df.sample(1).iloc[0]
            
        row = st.session_state.get('random_row', sample_df.sample(1).iloc[0])
        st.info(f"Loaded applicant **{row.get('LoanId', 'Unknown')}**.")
    else:
        row = sample_df.iloc[0]

    # --- Main Form ---
    with st.form("prediction_form"):
        st.subheader("📝 Applicant Profile")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=int(row.get("Age", 30)))
            income = st.number_input("Monthly Income (€)", min_value=0.0, value=float(row.get("IncomeTotal", 2000.0)))
            new_customer = st.selectbox("New Credit Customer?", [1, 0], index=0 if row.get("NewCreditCustomer") == 1 else 1)
            country = st.selectbox("Country", ["EE", "ES", "FI", "SK"], index=["EE", "ES", "FI", "SK"].index(row.get("Country", "EE")) if row.get("Country", "EE") in ["EE", "ES", "FI", "SK"] else 0)

        with col2:
            applied_amount = st.number_input("Applied Amount (€)", min_value=0.0, value=float(row.get("AppliedAmount", 5000.0)))
            amount = st.number_input("Granted Amount (€)", min_value=0.0, value=float(row.get("Amount", 5000.0)))
            interest = st.number_input("Interest Rate (%)", min_value=0.0, value=float(row.get("Interest", 15.0)))
            duration = st.number_input("Loan Duration (Months)", min_value=1, value=int(row.get("LoanDuration", 60)))
            monthly_payment = st.number_input("Monthly Payment (€)", min_value=0.0, value=float(row.get("MonthlyPayment", 150.0)))

        with col3:
            existing_liabilities = st.number_input("Existing Liabilities", min_value=0, value=int(row.get("ExistingLiabilities", 0)))
            liabilities_total = st.number_input("Liabilities Total (€)", min_value=0.0, value=float(row.get("LiabilitiesTotal", 0.0)))
            prev_loans = st.number_input("Previous Loans Count", min_value=0, value=int(row.get("NoOfPreviousLoansBeforeLoan", 0)))
            prev_loans_amt = st.number_input("Previous Loans Amount (€)", min_value=0.0, value=float(row.get("AmountOfPreviousLoansBeforeLoan", 0.0)))
            prev_repayments = st.number_input("Previous Repayments (€)", min_value=0.0, value=float(row.get("PreviousRepaymentsBeforeLoan", 0.0)))
            prev_early_repay = st.number_input("Previous Early Repayments", min_value=0, value=int(row.get("PreviousEarlyRepaymentsCountBeforeLoan", 0)))

        st.markdown("#### Categorical Details")
        col4, col5 = st.columns(2)
        with col4:
            verification = st.selectbox("Verification Type", ["0.0", "1.0", "2.0", "3.0", "4.0", "Missing"], index=0)
            education = st.selectbox("Education", ["1.0", "2.0", "3.0", "4.0", "5.0", "Missing"], index=0)
            home_ownership = st.selectbox("Home Ownership", ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "Missing"], index=0)
        with col5:
            employment = st.selectbox("Employment Duration", ["UpTo1Year", "UpTo2Years", "UpTo3Years", "UpTo4Years", "UpTo5Years", "MoreThan5Years", "Retiree", "Other", "Missing"], index=5)
            fi_score = st.selectbox("FI AsiakasTieto Risk Grade", ["RL1", "RL2", "RL3", "RL4", "RL5", "Missing"], index=5)
            ee_score = st.selectbox("EE Mini Score", ["1000", "900", "800", "700", "600", "Missing"], index=5)

        submitted = st.form_submit_button(f"🚀 Analyze Risk using {selected_model_name}")

    if submitted:
        input_data = pd.DataFrame([{
            "Age": age, "AppliedAmount": applied_amount, "Amount": amount, "Interest": interest,
            "LoanDuration": duration, "MonthlyPayment": monthly_payment, "IncomeTotal": income,
            "ExistingLiabilities": existing_liabilities, "LiabilitiesTotal": liabilities_total,
            "NoOfPreviousLoansBeforeLoan": prev_loans, "AmountOfPreviousLoansBeforeLoan": prev_loans_amt,
            "PreviousRepaymentsBeforeLoan": prev_repayments, "PreviousEarlyRepaymentsCountBeforeLoan": prev_early_repay,
            "NewCreditCustomer": new_customer, "Country": country, "VerificationType": verification,
            "Education": education, "HomeOwnershipType": home_ownership,
            "EmploymentDurationCurrentEmployer": employment, "CreditScoreFiAsiakasTietoRiskGrade": fi_score,
            "CreditScoreEeMini": ee_score
        }])
        
        with st.spinner(f"Running inference through {selected_model_name}..."):
            # 1. Feature Engineering (Domain specific transformations)
            X_engineered = lu.engineer_features(input_data)
            
            # 2. Get predictions from the selected model
            prob_default = selected_model.predict_proba(X_engineered)[0, 1]
            is_default = prob_default > 0.5
            
            st.markdown("---")
            st.subheader(f"Results from {selected_model_name}")
            
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f'<div class="metric-card"><h3>Probability of Default</h3><div class="{"risk-high" if prob_default > 0.5 else "risk-low"}">{prob_default:.1%}</div></div>', unsafe_allow_html=True)
                
            with rc2:
                st.markdown(f'<div class="metric-card"><h3>Risk Assessment</h3><div class="{"risk-high" if is_default else "risk-low"}">{"High Risk (Reject)" if is_default else "Low Risk (Approve)"}</div></div>', unsafe_allow_html=True)
                
            # Quick Comparison Across ALL models
            st.markdown("#### Opinion from other models:")
            comp_cols = st.columns(len(models))
            for i, (m_name, m_obj) in enumerate(models.items()):
                m_prob = m_obj.predict_proba(X_engineered)[0, 1]
                with comp_cols[i]:
                    st.metric(label=m_name, value=f"{m_prob:.1%}")

with tab_compare:
    st.header("📈 Multi-Model Performance Comparison")
    st.markdown("The platform runs 3 different algorithms natively. Here are their expected performances on the Bondora test set:")
    
    comp_df = pd.DataFrame({
        "Model": ["XGBoost", "LightGBM", "Random Forest"],
        "ROC-AUC": [0.7702, 0.7692, 0.7668],
        "F1-Score": [0.8130, 0.8137, 0.8147],
        "PR-AUC": [0.8362, 0.8351, 0.8300]
    }).set_index("Model")
    
    st.dataframe(comp_df.style.highlight_max(axis=0, color='#764ba2'))
    st.info("XGBoost provides the best ROC-AUC for this highly imbalanced credit dataset, making it the default choice for the prediction engine.")

with tab_architecture:
    st.header("🏗️ System Architecture & Data Flow")
    
    st.markdown("""
    <div class="arch-box">
    <h3>1. Data Layer</h3>
    Raw applications flow from the Bondora API into the system. Historic data (110k+ seasoned loans) is used as the foundational training set.
    </div>
    
    <div class="arch-box">
    <h3>2. Feature Engineering Layer <code>(loan_utils.py)</code></h3>
    Transforms raw data into model-ready features.
    <ul>
    <li><b>Domain Rules:</b> Fixes missing monthly payments, adjusts previous loan logic.</li>
    <li><b>Ratio Features:</b> Computes <code>PaymentToIncome</code>, <code>LiabilitiesToIncome</code>, <code>GrantedRatio</code>.</li>
    <li><b>Cleansing:</b> Converts categorical codes into clean strings.</li>
    </ul>
    </div>
    
    <div class="arch-box">
    <h3>3. Preprocessing Pipeline <code>(sklearn ColumnTransformer)</code></h3>
    <ul>
    <li><b>Numeric:</b> Robust <i>Winsorization</i> (clipping 0.5% and 99.5% quantiles to prevent outlier skew), followed by <i>StandardScaling</i>.</li>
    <li><b>Categorical:</b> One-Hot Encoding with tracking for rare/unknown categories.</li>
    </ul>
    </div>
    
    <div class="arch-box">
    <h3>4. Inference Layer (Ensemble Models)</h3>
    The engineered and scaled features are passed simultaneously to three distinct tree-based classifiers:
    <ul>
    <li><b>XGBoost (Champion):</b> Gradient boosted decision trees optimized for class imbalance.</li>
    <li><b>LightGBM:</b> Leaf-wise tree growth, excellent for categorical heavy data.</li>
    <li><b>Random Forest:</b> Bagged decision trees, provides strong baseline stability.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
