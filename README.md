# Bondora Credit Risk Platform

A Streamlit research prototype for Bondora default-risk scoring. It combines sklearn pipelines, SHAP explanations, LIME/neurosymbolic analysis from the notebook, nine readable risk rules, and a fairness audit.

## Layout

- `app.py`: Streamlit landing screen, assessment flow, explanation view, and how-it-works page.
- `styles.py`, `components.py`, `charts.py`, `explain.py`: frontend styling, reusable UI, Plotly charts, and SHAP/rule explanation helpers.
- `loan_utils.py`: shared feature engineering, categorical normalization, threshold, and rulebook.
- `train_quick_model.py`: quick untuned XGBoost, LightGBM, and Random Forest training.
- `Bondora_Risk_NeSy_XAI.ipynb` / `.py`: analysis and tuned champion research; not used by the app at runtime.
- `artifacts/`: shipped sklearn pipelines and model metadata.
- `LoanData_Sample.csv`: 5,000-row sample used by the app.
- `figures/`: notebook figures.

## Install and run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The full `LoanData_Bondora.csv` dataset is gitignored and is needed only to retrain. Download it from Bondora's public dataset source, place it in the repository root, and run:

```powershell
python train_quick_model.py
```

The shipped artifacts are produced by `train_quick_model.py`. The notebook contains the tuned champion and its held-out evaluation.

## Results

The notebook reports test ROC-AUC 0.781 for tuned XGBoost. A time-based split gives the more cautious real-world figure of 0.715. The app uses a 0.65 decision threshold and labels outputs as decision-support scores, not automated decisions.

## Limitations and fairness

Country and Age remain model inputs, and Country may be legally restricted in credit decisions. The fairness tab reports the notebook's audit: disparities are mostly country-driven, while gender is not an input. Experimental country-specific thresholds reduce the measured false-positive gap at a cost to balanced accuracy and recall.

## Screenshots

Screenshots of the landing page, assessment result, and explanation view can be added here.
