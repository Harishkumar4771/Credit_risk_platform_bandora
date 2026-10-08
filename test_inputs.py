import joblib
import pandas as pd

import loan_utils as lu


def main():
    model = joblib.load("artifacts/xgboost_pipeline.joblib")
    options = lu.category_options(model)
    categories = dict(zip(lu.CATEGORICAL, model.named_steps["preprocessor"].named_transformers_["cat"].categories_))
    for column, values in options.items():
        assert set(values) == {str(value) for value in categories[column]}

    sample = pd.read_csv("LoanData_Sample.csv", nrows=1, low_memory=False)
    base = lu.engineer_features(sample)
    education_one = base.copy()
    education_five = base.copy()
    education_one.loc[:, "Education"] = "1"
    education_five.loc[:, "Education"] = "5"
    one_probability = model.predict_proba(education_one)[0, 1]
    five_probability = model.predict_proba(education_five)[0, 1]
    assert one_probability != five_probability
    assert lu.THRESHOLD == 0.65
    print("Input checks passed")


if __name__ == "__main__":
    main()
