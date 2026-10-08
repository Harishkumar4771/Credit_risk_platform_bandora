import pandas as pd
import numpy as np
import loan_utils as lu
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
import json
import sklearn
import xgboost
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

THRESHOLD = lu.THRESHOLD

print("Loading data...")
raw = pd.read_csv("LoanData_Bondora.csv", low_memory=False)
raw["LoanDate"] = pd.to_datetime(raw["LoanDate"])
raw["DefaultFlag"] = raw["DefaultDate"].notna().astype(int)
REPORT_DATE = pd.to_datetime(raw["ReportAsOfEOD"].iloc[0])
SEASONING_CUTOFF = REPORT_DATE - pd.DateOffset(months=12)
is_resolved = (raw["DefaultFlag"] == 1) | (raw["Status"] == "Repaid")
is_seasoned = raw["LoanDate"] <= SEASONING_CUTOFF
model_df = raw.loc[is_resolved & is_seasoned].copy()

print("Engineering features...")
X = lu.engineer_features(model_df)
y = model_df["DefaultFlag"]
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_transformer = Pipeline(steps=[
    ('winsorizer', lu.Winsorizer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, lu.NUMERIC_ALL),
        ('cat', categorical_transformer, lu.CATEGORICAL)
    ])

models = {
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100, max_depth=6, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=42, n_estimators=150, max_depth=10),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15, n_jobs=-1)
}

os.makedirs("artifacts", exist_ok=True)

# These are quick untuned models for the shipped app; the notebook's tuned champion is separate.
metrics = {}
for name, model in models.items():
    evaluation_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    evaluation_pipe.fit(x_train, y_train)
    probabilities = evaluation_pipe.predict_proba(x_test)[:, 1]
    predictions = probabilities >= THRESHOLD
    metrics[name] = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "precision_at_threshold": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_test, predictions, zero_division=0)),
    }
with open("artifacts/metrics.json", "w", encoding="utf-8") as metrics_file:
    json.dump({"threshold": THRESHOLD, "metrics": metrics, "library_versions": {
        "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__
    }}, metrics_file, indent=2)

for name, model in models.items():
    print(f"Training {name}...")
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    clf.fit(X, y)
    filename = f"artifacts/{name.replace(' ', '_').lower()}_pipeline.joblib"
    joblib.dump(clf, filename)
    print(f"Saved {name} to {filename}")

print("All models trained and saved!")
