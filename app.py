from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import charts
import components
import explain
import loan_utils as lu
from styles import get_css

THRESHOLD = lu.THRESHOLD
COUNTRY_THRESHOLDS = {"FI": 0.748, "ES": 0.82, "EE": 0.523, "SK": 0.65}
MODEL_PATHS = {
    "XGBoost": Path("artifacts/xgboost_pipeline.joblib"),
    "LightGBM": Path("artifacts/lightgbm_pipeline.joblib"),
    "Random Forest": Path("artifacts/random_forest_pipeline.joblib"),
}

st.set_page_config(page_title="Bondora Default Risk", layout="wide", initial_sidebar_state="collapsed")
st.markdown(get_css(), unsafe_allow_html=True)


@st.cache_resource
def load_models():
    models, messages = {}, []
    for label, path in MODEL_PATHS.items():
        if not path.exists():
            messages.append(("info", f"{label} is unavailable because {path} is not present."))
            continue
        try:
            models[label] = joblib.load(path)
        except Exception as error:
            messages.append(("warning", f"{label} could not be loaded: {error}"))
    return models, messages


@st.cache_data
def load_sample_data():
    return pd.read_csv("LoanData_Sample.csv", nrows=5000, low_memory=False)


def row_number(row, column, default, integer=False):
    value = row.get(column, default)
    if pd.isna(value):
        value = default
    return int(value) if integer else float(value)


def row_category(row, column, options):
    value = lu._cat_to_str(pd.Series([row.get(column)]))[0]
    return value if value in options[column] else ("Missing" if "Missing" in options[column] else options[column][0])


def neutral_row(model):
    options = lu.category_options(model)
    values = {
        "Age": 35, "AppliedAmount": 5000.0, "Amount": 5000.0, "Interest": 15.0, "LoanDuration": 36,
        "MonthlyPayment": 150.0, "IncomeTotal": 2500.0, "ExistingLiabilities": 0,
        "LiabilitiesTotal": 0.0, "NoOfPreviousLoansBeforeLoan": 1, "AmountOfPreviousLoansBeforeLoan": 3000.0,
        "PreviousRepaymentsBeforeLoan": 1500.0, "PreviousEarlyRepaymentsCountBeforeLoan": 0,
        "NewCreditCustomer": 0,
    }
    for column, choices in options.items():
        values[column] = "EE" if column == "Country" and "EE" in choices else ("Missing" if "Missing" in choices else choices[0])
    return pd.Series(values)


def clear_form_values():
    keys = "age income new_customer country applied_amount amount interest duration monthly_payment existing_liabilities liabilities_total prev_loans prev_loans_amt prev_repayments prev_early_repay verification education home_ownership employment fi_score ee_score".split()
    for key in keys:
        st.session_state.pop(key, None)


def choose_random_row(sample_df):
    available_countries = sample_df["Country"].dropna().astype(str).unique().tolist()
    if not available_countries:
        return sample_df.sample(1).iloc[0]
    country = np.random.default_rng().choice(available_countries)
    country_rows = sample_df.loc[sample_df["Country"].astype(str) == country]
    return country_rows.sample(1).iloc[0]


def start_app():
    st.session_state["stage"] = "app"


def start_over():
    st.session_state["stage"] = "landing"
    st.session_state.pop("last_result", None)
    clear_form_values()


def render_landing():
    st.markdown('<main class="landing-screen">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Bondora loan data</div>', unsafe_allow_html=True)
    st.title("Estimate how likely a loan is to default, and see why.")
    st.markdown('<p class="landing-copy">Enter an applicant\'s details or pick a real one from the Bondora loan book. The app gives a risk estimate and shows which factors pushed it up or down.</p>', unsafe_allow_html=True)
    st.write("")
    columns = st.columns(3)
    steps = [("1", "Choose an applicant", "Enter details yourself or draw a real example."), ("2", "Get a risk estimate", "Two trained models score the loan."), ("3", "See the reasons", "Plain-language drivers, charts and rules that fired.")]
    for column, (number, title, copy) in zip(columns, steps):
        with column:
            st.markdown(f'<div class="step-number">{number}</div><div class="step-title">{title}</div><div class="step-copy">{copy}</div>', unsafe_allow_html=True)
    st.write("")
    st.button("Start an assessment", on_click=start_app, type="primary")
    st.markdown('<div class="landing-footer">Research prototype built on public Bondora data. Not for real lending decisions.</div>', unsafe_allow_html=True)
    st.markdown('</main>', unsafe_allow_html=True)


def input_form(row, options, random_mode, form_namespace):
    field_key = lambda name: f"{form_namespace}_{name}"
    container = st.expander("Review or edit details", expanded=True) if random_mode else nullcontext()
    with container:
        with st.form("assessment_form"):
            st.markdown('<div class="section-label">About the applicant</div>', unsafe_allow_html=True)
            about_one, about_two = st.columns(2)
            with about_one:
                age = st.number_input("Age", min_value=18, max_value=100, value=row_number(row, "Age", 35, True), help="Applicant age in years.", key=field_key("age"))
                income = st.number_input("Monthly income (EUR)", min_value=0.0, value=row_number(row, "IncomeTotal", 2500.0), help="Total monthly income in euros.", key=field_key("income"))
                country = st.selectbox("Country", options["Country"], index=options["Country"].index(row_category(row, "Country", options)), help="Bondora country code.", key=field_key("country"))
            with about_two:
                customer_value = 1 if row.get("NewCreditCustomer") == 1 else 0
                new_customer = st.selectbox("Customer type", [0, 1], index=[0, 1].index(customer_value), format_func=lambda value: "New customer" if value == 1 else "Returning customer", help="Whether this is the applicant's first credit customer record.", key=field_key("new_customer"))

            st.markdown('<div class="section-label">The loan</div>', unsafe_allow_html=True)
            loan_one, loan_two = st.columns(2)
            with loan_one:
                applied_amount = st.number_input("Amount applied for (EUR)", min_value=0.0, value=row_number(row, "AppliedAmount", 5000.0), help="Requested loan amount.", key=field_key("applied_amount"))
                amount = st.number_input("Amount granted (EUR)", min_value=0.0, value=row_number(row, "Amount", 5000.0), help="Granted loan amount.", key=field_key("amount"))
                interest = st.number_input("Interest rate (%)", min_value=0.0, value=row_number(row, "Interest", 15.0), help="Annual interest rate.", key=field_key("interest"))
            with loan_two:
                duration = st.number_input("Duration (months)", min_value=1, value=row_number(row, "LoanDuration", 36, True), help="Loan duration in months.", key=field_key("duration"))
                monthly_payment = st.number_input("Monthly payment (EUR)", min_value=0.0, value=row_number(row, "MonthlyPayment", 150.0), help="Expected monthly payment.", key=field_key("monthly_payment"))

            st.markdown('<div class="section-label">Credit history</div>', unsafe_allow_html=True)
            history_one, history_two = st.columns(2)
            with history_one:
                existing_liabilities = st.number_input("Existing liabilities (count)", min_value=0, value=row_number(row, "ExistingLiabilities", 0, True), help="Number of existing liabilities.", key=field_key("existing_liabilities"))
                liabilities_total = st.number_input("Total liabilities (EUR)", min_value=0.0, value=row_number(row, "LiabilitiesTotal", 0.0), help="Total value of existing liabilities.", key=field_key("liabilities_total"))
                prev_loans = st.number_input("Previous loans (count)", min_value=0, value=row_number(row, "NoOfPreviousLoansBeforeLoan", 0, True), help="Loans held before this application.", key=field_key("prev_loans"))
            with history_two:
                prev_loans_amt = st.number_input("Previous loans amount (EUR)", min_value=0.0, value=row_number(row, "AmountOfPreviousLoansBeforeLoan", 0.0), help="Amount borrowed on previous loans.", key=field_key("prev_loans_amt"))
                prev_repayments = st.number_input("Previous repayments (EUR)", min_value=0.0, value=row_number(row, "PreviousRepaymentsBeforeLoan", 0.0), help="Amount repaid before this loan.", key=field_key("prev_repayments"))
                prev_early_repay = st.number_input("Previous early repayments", min_value=0, value=row_number(row, "PreviousEarlyRepaymentsCountBeforeLoan", 0, True), help="Count of previous early repayments.", key=field_key("prev_early_repay"))

            st.markdown('<div class="section-label">Background</div>', unsafe_allow_html=True)
            background_one, background_two = st.columns(2)
            with background_one:
                verification = st.selectbox("Verification type", options["VerificationType"], index=options["VerificationType"].index(row_category(row, "VerificationType", options)), help="Bondora category code.", key=field_key("verification"))
                education = st.selectbox("Education", options["Education"], index=options["Education"].index(row_category(row, "Education", options)), help="Bondora education category code.", key=field_key("education"))
                home_ownership = st.selectbox("Home ownership", options["HomeOwnershipType"], index=options["HomeOwnershipType"].index(row_category(row, "HomeOwnershipType", options)), help="Bondora home ownership category code.", key=field_key("home_ownership"))
            with background_two:
                employment = st.selectbox("Employment duration", options["EmploymentDurationCurrentEmployer"], index=options["EmploymentDurationCurrentEmployer"].index(row_category(row, "EmploymentDurationCurrentEmployer", options)), help="Bondora employment duration code.", key=field_key("employment"))
                fi_score = st.selectbox("FI risk grade", options["CreditScoreFiAsiakasTietoRiskGrade"], index=options["CreditScoreFiAsiakasTietoRiskGrade"].index(row_category(row, "CreditScoreFiAsiakasTietoRiskGrade", options)), help="Bondora Finnish risk grade code.", key=field_key("fi_score"))
                ee_score = st.selectbox("EE score", options["CreditScoreEeMini"], index=options["CreditScoreEeMini"].index(row_category(row, "CreditScoreEeMini", options)), help="Bondora Estonian score code.", key=field_key("ee_score"))
            submitted = st.form_submit_button("Assess this applicant", type="primary")
    values = {
        "Age": age, "AppliedAmount": applied_amount, "Amount": amount, "Interest": interest, "LoanDuration": duration, "MonthlyPayment": monthly_payment, "IncomeTotal": income,
        "ExistingLiabilities": existing_liabilities, "LiabilitiesTotal": liabilities_total, "NoOfPreviousLoansBeforeLoan": prev_loans, "AmountOfPreviousLoansBeforeLoan": prev_loans_amt,
        "PreviousRepaymentsBeforeLoan": prev_repayments, "PreviousEarlyRepaymentsCountBeforeLoan": prev_early_repay, "NewCreditCustomer": new_customer, "Country": country,
        "VerificationType": verification, "Education": education, "HomeOwnershipType": home_ownership, "EmploymentDurationCurrentEmployer": employment,
        "CreditScoreFiAsiakasTietoRiskGrade": fi_score, "CreditScoreEeMini": ee_score,
    }
    return submitted, pd.DataFrame([values])


def render_assess(models, sample_df):
    st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
    st.subheader("Who do you want to assess?")
    choices = ["Custom applicant", "Random from the Bondora sample"]
    if hasattr(st, "segmented_control"):
        mode = st.segmented_control("Applicant source", choices, default=st.session_state.get("input_mode", choices[0]), key="input_mode", label_visibility="collapsed")
    else:
        mode = st.radio("Applicant source", choices, horizontal=True, key="input_mode")
    random_mode = mode == choices[1]
    previous_mode = st.session_state.get("previous_input_mode")
    if previous_mode is not None and previous_mode != mode:
        clear_form_values()
        st.session_state.pop("last_result", None)
        st.session_state["form_revision"] = st.session_state.get("form_revision", 0) + 1
    st.session_state["previous_input_mode"] = mode
    model_default = "XGBoost" if "XGBoost" in models else next(iter(models))
    with st.expander("Advanced", expanded=False):
        selected_model_name = st.selectbox("Model", list(models), index=list(models).index(st.session_state.get("selected_model", model_default)), key="selected_model")
        use_country_thresholds = st.checkbox("Use per-country thresholds (experimental)", key="country_thresholds")
        for kind, message in load_models()[1]:
            st.info(message) if kind == "info" else st.warning(message)
    selected_model = models[selected_model_name]
    options = lu.category_options(selected_model)
    if random_mode:
        if "random_row" not in st.session_state:
            st.session_state["random_row"] = choose_random_row(sample_df)
        random_row = st.session_state["random_row"]
        st.markdown('<div class="section-label">Selected sample applicant</div>', unsafe_allow_html=True)
        components.outcome_card(random_row)
        draw_col, _ = st.columns([1, 3])
        with draw_col:
            st.button("Draw another", key="draw_another", on_click=draw_random, args=(sample_df,))
        row = random_row
    else:
        row = neutral_row(selected_model)
    st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
    namespace = f"{'random' if random_mode else 'custom'}_{st.session_state.get('form_revision', 0)}"
    submitted, input_data = input_form(row, options, random_mode, namespace)
    if submitted:
        if float(input_data.loc[0, "IncomeTotal"]) <= 0:
            st.error("Monthly income must be greater than zero before the score can be calculated.")
        else:
            if input_data.loc[0, "Amount"] > input_data.loc[0, "AppliedAmount"]:
                st.warning("The granted amount is higher than the applied amount. The score can still be calculated.")
            with st.spinner("Calculating the assessment"):
                engineered = lu.engineer_features(input_data)
                probabilities = {name: float(model.predict_proba(engineered)[0, 1]) for name, model in models.items()}
            country = input_data.loc[0, "Country"]
            threshold = COUNTRY_THRESHOLDS.get(country, THRESHOLD) if use_country_thresholds else THRESHOLD
            st.session_state["last_result"] = {"input": input_data, "engineered": engineered, "probabilities": probabilities, "selected_model": selected_model_name, "threshold": threshold, "random_mode": random_mode, "source_row": row}
    render_result(models)


def draw_random(sample_df):
    st.session_state["random_row"] = choose_random_row(sample_df)
    clear_form_values()
    st.session_state.pop("last_result", None)
    st.session_state["form_revision"] = st.session_state.get("form_revision", 0) + 1


def render_result(models):
    result = st.session_state.get("last_result")
    if not result:
        return
    probability = result["probabilities"][result["selected_model"]]
    threshold = result["threshold"]
    st.markdown('<div class="section-label">Step 3</div>', unsafe_allow_html=True)
    st.subheader("Assessment")
    left, right = st.columns([1, 1.4])
    with left:
        components.risk_result(probability, threshold)
        st.caption("Scores of 65% or above are flagged. The cutoff balances catching defaulters against wrongly flagging good borrowers.")
        if result["random_mode"]:
            source = result["source_row"]
            has_outcome = pd.notna(source.get("DefaultDate")) or str(source.get("Status", "")) in {"Repaid", "Defaulted"}
            if has_outcome:
                reveal = st.checkbox("Reveal what actually happened", key="reveal_outcome")
                if reveal:
                    actual = "Defaulted" if pd.notna(source.get("DefaultDate")) else str(source.get("Status"))
                    st.caption(f"Recorded outcome in the sample: {actual}.")
    with right:
        st.plotly_chart(charts.risk_meter(probability, threshold), use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(charts.model_bars(result["probabilities"], threshold), use_container_width=True, config={"displayModeBar": False})
    values = list(result["probabilities"].values())
    if len(values) > 1 and max(values) - min(values) > 0.15:
        st.info("The models disagree; treat this case with extra care.")
    st.caption("See the Explanation tab to understand why.")
    download = result["input"].copy()
    for name, value in result["probabilities"].items():
        download[f"{name}_probability"] = value
    download["threshold"] = threshold
    st.download_button("Download this assessment (CSV)", download.to_csv(index=False), "bondora_assessment.csv", "text/csv")


def render_explanation(models, sample_df):
    result = st.session_state.get("last_result")
    model_name = result["selected_model"] if result else ("XGBoost" if "XGBoost" in models else next(iter(models)))
    model = models[model_name]
    st.subheader("What drives a score")
    st.write("The model looks at things like interest rate, loan term, income and credit history. This page shows how much each factor pushed this applicant's risk up or down.")
    if result:
        st.subheader("Why this applicant got this score")
        try:
            local, base_value = explain.local_shap(model_name, model, result["engineered"])
            local["label"] = local["label"]
            local["display"] = local.apply(lambda row: f"{row['label']}: {row['display']}", axis=1)
            up = local.loc[local["shap"] > 0, "label"].head(2).tolist()
            down = local.loc[local["shap"] < 0, "label"].head(2).tolist()
            st.write(f"The biggest factors raising this applicant's risk are {', '.join(up) or 'none'}; the biggest factors lowering it are {', '.join(down) or 'none'}.")
            st.plotly_chart(charts.signed_shap_bar(local), use_container_width=True, config={"displayModeBar": False})
            with st.expander("Technical detail"):
                st.write(f"SHAP values are in log-odds on transformed features. Base value: {base_value:.3f}. Model: {model_name}.")
        except Exception as error:
            st.info(f"This applicant's explanation is unavailable: {error}")
    else:
        st.info("Run an assessment first and the explanation will appear here.")

    engineered_for_rules = result["engineered"] if result else lu.engineer_features(pd.DataFrame([neutral_row(model)]))
    predicates = lu.symbolic_predicates(engineered_for_rules).iloc[0]
    st.subheader("Rules of thumb that apply")
    rule_rows = []
    risk_count = protective_count = 0
    for rule in lu.RULEBOOK:
        applies = bool(predicates[rule["name"]])
        if applies and rule["expected"] > 0:
            risk_count += 1
        if applies and rule["expected"] < 0:
            protective_count += 1
        direction = "Normally raises risk" if rule["expected"] > 0 else "Normally lowers risk"
        rule_rows.append({"Rule": rule["text"], "Applies": "Applies" if applies else "Does not apply", "Direction": direction})
    st.dataframe(pd.DataFrame(rule_rows), hide_index=True, use_container_width=True)
    st.caption(f"{risk_count} risk signals and {protective_count} protective signals apply. These are simple credit rules written by hand; they are cruder than the model, so they can disagree with it.")

    if result:
        st.subheader("What if?")
        base_input = result["input"].iloc[0].copy()
        what_one, what_two = st.columns(2)
        with what_one:
            what_interest = st.slider("Interest rate (%)", 0.0, 60.0, float(base_input["Interest"]), key="what_interest")
            what_amount = st.slider("Loan amount (EUR)", 0.0, 50000.0, float(base_input["Amount"]), step=100.0, key="what_amount")
        with what_two:
            what_duration = st.slider("Duration (months)", 1, 120, int(base_input["LoanDuration"]), key="what_duration")
            what_income = st.slider("Monthly income (EUR)", 1.0, 15000.0, float(base_input["IncomeTotal"]), step=50.0, key="what_income")
        scenario = base_input.copy()
        scenario["Interest"], scenario["Amount"], scenario["LoanDuration"], scenario["IncomeTotal"] = what_interest, what_amount, what_duration, what_income
        scenario_frame = pd.DataFrame([scenario])
        scenario_probability = float(model.predict_proba(lu.engineer_features(scenario_frame))[0, 1])
        st.metric("Scenario probability", f"{scenario_probability:.1%}")
        rates = np.linspace(max(0, what_interest - 10), min(60, what_interest + 10), 9)
        scenarios = []
        for rate in rates:
            item = scenario.copy()
            item["Interest"] = rate
            scenarios.append({"interest": rate, "probability": float(model.predict_proba(lu.engineer_features(pd.DataFrame([item])))[0, 1])})
        st.plotly_chart(charts.line_what_if(pd.DataFrame(scenarios), {"interest": what_interest, "probability": scenario_probability}), use_container_width=True, config={"displayModeBar": False})

    st.subheader("What matters most overall")
    try:
        global_frame = sample_df.sample(min(500, len(sample_df)), random_state=42)
        global_engineered = lu.engineer_features(global_frame)
        importance = explain.global_shap_importance(model_name, model, global_engineered)
        st.plotly_chart(charts.global_importance(importance), use_container_width=True, config={"displayModeBar": False})
    except Exception as error:
        st.info(f"Overall explanation is unavailable: {error}")
    st.subheader("Does the model rank risk well?")
    st.plotly_chart(charts.decile_chart(), use_container_width=True, config={"displayModeBar": False})
    st.caption("From the notebook's held-out test set of 22,069 loans. The riskiest tenth defaults far more often than the safest tenth.")


def render_how_it_works(models):
    st.subheader("How it works")
    steps = [
        (1, "Data", "Bondora peer-to-peer loans.", "179,235 loans were reduced to 110,342 with a clear outcome and at least 12 months of history. 64% defaulted."),
        (2, "Cleaning and safety checks", "Inputs are reviewed before modelling.", "112 columns were reviewed. 37 post-start columns were removed to avoid leakage, along with 6 discontinued and 10 almost-empty fields. Protected fields are not model inputs."),
        (3, "Features", "The app turns application details into 26 model inputs.", "21 application fields plus five ratios. Numbers are clipped at the 0.5% and 99.5% quantiles and scaled; categories are one-hot encoded."),
        (4, "Models", "Several model families were compared before deployment.", "Seven families were compared with five-fold cross-validation. Tuned XGBoost was the notebook champion. This app uses XGBoost and LightGBM retrained on all loans; the shipped versions are untuned."),
        (5, "Decision", "A probability becomes a practical flag.", "The app uses a 65% cutoff. Per-country cutoffs are available as an experimental option."),
        (6, "Explanation", "The result includes model drivers and written rules.", "SHAP groups transformed columns back to readable features. Nine hand-written credit rules are evaluated for the applicant."),
        (7, "Fairness check", "The notebook audited country, gender code and age.", "Country and Age remain model inputs. The fairness figures below are from the notebook and are not recomputed live."),
    ]
    components.step_flow(steps)
    st.subheader("How good is it?")
    card_one, card_two = st.columns(2)
    with card_one:
        components.surface("<div class='summary-label'>Held-out test ROC-AUC</div><div class='risk-number'>0.781</div><div class='muted'>95% interval: 0.775 to 0.787</div>")
    with card_two:
        components.surface("<div class='summary-label'>Time-based ROC-AUC</div><div class='risk-number'>0.715</div><div class='muted'>Train through 2018, test from 2019</div>")
    st.write("The second figure is the more cautious estimate for real use because borrower behaviour changes over time. Precision is 80.4% and recall is 73.9% at the 65% cutoff. The deployed models were retrained on all data, so these figures come from the notebook.")
    st.subheader("Limits and fairness")
    st.caption("From the notebook's audit, not recomputed live")
    st.plotly_chart(charts.fairness_chart(), use_container_width=True, config={"displayModeBar": False})
    st.markdown("- Gender is not an input, but disparities appear, mostly through country.\n- Gender code 2 is almost entirely Spanish applicants.\n- Applicants aged 55 and over are wrongly flagged more often: 45.4% versus 26.8% for ages 25 to 34.\n- Removing Age does not close the gap. Country and Age remain model inputs.\n- Per-country cutoffs narrowed the country gap from 0.583 to 0.020, but lowered balanced accuracy from 0.709 to 0.665. This is a policy trade-off.")
    st.warning("Research prototype. Not suitable for real lending decisions.")
    with st.expander("Technical details"):
        st.write("Libraries: Streamlit, pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP and Plotly. Files: app.py, styles.py, components.py, charts.py, explain.py, loan_utils.py, train_quick_model.py, artifacts/, and the notebook. Retraining requires the gitignored LoanData_Bondora.csv and `python train_quick_model.py`.")


models, model_messages = load_models()
sample_df = load_sample_data()
if not models:
    st.error("No model artifacts are available. Add a trained pipeline under artifacts/.")
    st.stop()
st.session_state.setdefault("stage", "landing")
if st.session_state["stage"] == "landing":
    render_landing()
else:
    st.markdown('<div class="topbar"><div class="wordmark">Bondora Default Risk</div></div>', unsafe_allow_html=True)
    top_left, top_right = st.columns([5, 1])
    with top_right:
        st.button("Back to start", on_click=start_over, key="back_to_start")
    assess_tab, explanation_tab, how_tab = st.tabs(["Assess", "Explanation", "How it works"])
    with assess_tab:
        render_assess(models, sample_df)
    with explanation_tab:
        render_explanation(models, sample_df)
    with how_tab:
        render_how_it_works(models)
    st.button("Start over", on_click=start_over, key="start_over")
