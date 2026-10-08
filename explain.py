import numpy as np
import pandas as pd
import streamlit as st

FRIENDLY_NAMES = {
    "Age": "Age",
    "Interest": "Interest rate",
    "LoanDuration": "Loan duration",
    "IncomeTotal": "Monthly income",
    "PaymentToIncome": "Payment as share of income",
    "LiabToIncome": "Liabilities as share of income",
    "AmountToIncome": "Loan amount as share of income",
    "GrantedRatio": "Granted amount ratio",
    "PrevRepaymentRatio": "Previous repayment ratio",
    "AppliedAmount": "Amount applied for",
    "Amount": "Amount granted",
    "MonthlyPayment": "Monthly payment",
    "Country": "Country",
    "Education": "Education",
    "VerificationType": "Verification type",
    "HomeOwnershipType": "Home ownership",
    "EmploymentDurationCurrentEmployer": "Employment duration",
    "CreditScoreFiAsiakasTietoRiskGrade": "FI risk grade",
    "CreditScoreEeMini": "EE score",
    "NewCreditCustomer": "New customer",
}


def friendly_name(name):
    base = name.split("__", 1)[-1]
    for column in sorted(FRIENDLY_NAMES, key=len, reverse=True):
        if base == column or base.startswith(column + "_"):
            return FRIENDLY_NAMES[column]
    return base.replace("_", " ").replace("Ratio", " ratio").strip().title()


def _feature_base(name):
    base = name.split("__", 1)[-1]
    for column in sorted(FRIENDLY_NAMES, key=len, reverse=True):
        if base == column or base.startswith(column + "_"):
            return column
    return base


@st.cache_resource
def tree_explainer(model_name, _model):
    import shap
    return shap.TreeExplainer(_model.named_steps["classifier"])


def _class_one_values(values):
    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def local_shap(model_name, model, engineered):
    transformed = model.named_steps["preprocessor"].transform(engineered)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    raw_values = tree_explainer(model_name, model).shap_values(transformed)
    values = _class_one_values(raw_values)[0]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    grouped = {}
    for name, value in zip(names, values):
        base = _feature_base(name)
        grouped[base] = grouped.get(base, 0.0) + float(value)
    rows = []
    for base, value in grouped.items():
        raw_value = engineered.iloc[0].get(base, "")
        if isinstance(raw_value, (float, np.floating)):
            display = f"{float(raw_value):.2f}"
        else:
            display = str(raw_value)
        rows.append({"feature": base, "label": FRIENDLY_NAMES.get(base, friendly_name(base)), "shap": value, "display": display})
    frame = pd.DataFrame(rows)
    frame["absolute"] = frame["shap"].abs()
    return frame.nlargest(8, "absolute").drop(columns="absolute"), float(np.asarray(getattr(tree_explainer(model_name, model), "expected_value", 0)).reshape(-1)[-1])


@st.cache_data
def global_shap_importance(model_name, _model, sample):
    engineered = sample
    transformed = _model.named_steps["preprocessor"].transform(engineered)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    values = _class_one_values(tree_explainer(model_name, _model).shap_values(transformed))
    names = _model.named_steps["preprocessor"].get_feature_names_out()
    grouped = {}
    for name, value in zip(names, np.abs(values).mean(axis=0)):
        base = _feature_base(name)
        grouped[base] = grouped.get(base, 0.0) + float(value)
    frame = pd.DataFrame([{"feature": key, "label": FRIENDLY_NAMES.get(key, friendly_name(key)), "importance": value} for key, value in grouped.items()])
    return frame.nlargest(10, "importance")
