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
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=42, n_estimators=150, max_depth=10),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15, n_jobs=-1)
}

os.makedirs("artifacts", exist_ok=True)

for name, model in models.items():
    print(f"Training {name}...")
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    clf.fit(X, y)
    filename = f"artifacts/{name.replace(' ', '_').lower()}_pipeline.joblib"
    joblib.dump(clf, filename)
    print(f"Saved {name} to {filename}")

print("All models trained and saved!")
