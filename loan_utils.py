"""Shared helpers for the Bondora default-risk project.

Kept in a separate module so that the saved sklearn pipeline can be un-pickled
later (for example by the website in step 4) without re-running the notebook.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------------------------------------------------------
# Feature schema (application-time information only)
# ----------------------------------------------------------------------------
NUMERIC_RAW = [
    "Age", "AppliedAmount", "Amount", "Interest", "LoanDuration", "MonthlyPayment",
    "IncomeTotal", "ExistingLiabilities", "LiabilitiesTotal",
    "NoOfPreviousLoansBeforeLoan", "AmountOfPreviousLoansBeforeLoan",
    "PreviousRepaymentsBeforeLoan", "PreviousEarlyRepaymentsCountBeforeLoan",
]
BINARY = ["NewCreditCustomer"]
CATEGORICAL = [
    "Country", "VerificationType", "Education", "HomeOwnershipType",
    "EmploymentDurationCurrentEmployer", "CreditScoreFiAsiakasTietoRiskGrade",
    "CreditScoreEeMini",
]
ENGINEERED = ["PaymentToIncome", "LiabToIncome", "AmountToIncome", "GrantedRatio",
              "PrevRepaymentRatio"]
NUMERIC_ALL = NUMERIC_RAW + BINARY + ENGINEERED
RAW_INPUT_COLUMNS = NUMERIC_RAW + BINARY + CATEGORICAL


def _cat_to_str(s: pd.Series) -> pd.Series:
    """Turn a coded column into clean strings ('4.0' -> '4', NaN -> 'Missing')."""
    def conv(v):
        if pd.isna(v):
            return "Missing"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return str(int(v)) if float(v).is_integer() else str(v)
        return str(v)
    return s.map(conv)


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw Bondora columns -> model-ready feature frame (row-wise, no fitting)."""
    X = raw[RAW_INPUT_COLUMNS].copy()
    X["NewCreditCustomer"] = X["NewCreditCustomer"].astype(int)

    # Data-quality rule: a live loan cannot have a monthly instalment of zero. Those rows are
    # an artefact of old records (see section 1.6) and are treated as missing.
    X["MonthlyPayment"] = X["MonthlyPayment"].where(X["MonthlyPayment"] > 0)

    # Domain rule: no previous loans means nothing was repaid before this loan.
    no_prev = X["NoOfPreviousLoansBeforeLoan"].fillna(0).eq(0)
    X.loc[no_prev & X["PreviousRepaymentsBeforeLoan"].isna(), "PreviousRepaymentsBeforeLoan"] = 0.0

    income = X["IncomeTotal"].where(X["IncomeTotal"] > 0)
    prev_amt = X["AmountOfPreviousLoansBeforeLoan"].where(X["AmountOfPreviousLoansBeforeLoan"] > 0)
    X["PaymentToIncome"] = X["MonthlyPayment"] / income
    X["LiabToIncome"] = X["LiabilitiesTotal"] / income
    X["AmountToIncome"] = X["Amount"] / income
    X["GrantedRatio"] = X["Amount"] / X["AppliedAmount"].where(X["AppliedAmount"] > 0)
    X["PrevRepaymentRatio"] = X["PreviousRepaymentsBeforeLoan"] / prev_amt
    X[ENGINEERED] = X[ENGINEERED].replace([np.inf, -np.inf], np.nan)

    for c in CATEGORICAL:
        X[c] = _cat_to_str(X[c])
    return X[NUMERIC_ALL + CATEGORICAL]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training data only."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# ----------------------------------------------------------------------------
# Symbolic knowledge base (expert credit-risk rules, evaluated on engineer_features output)
# expected = direction an analyst would expect: +1 raises default risk, -1 lowers it
# ----------------------------------------------------------------------------
def _gt(col, thr):
    return lambda X: X[col].ge(thr).fillna(False)


RULEBOOK = [
    dict(name="HighPaymentBurden", expected=+1, features=["PaymentToIncome"],
         text="the new monthly instalment is at least 15% of monthly income",
         fn=_gt("PaymentToIncome", 0.15)),
    dict(name="HighLiabilityLoad", expected=+1, features=["LiabToIncome"],
         text="existing monthly liabilities are at least 40% of monthly income",
         fn=_gt("LiabToIncome", 0.40)),
    dict(name="LargeLoanVsIncome", expected=+1, features=["AmountToIncome"],
         text="the loan is at least 3 times the monthly income",
         fn=_gt("AmountToIncome", 3.0)),
    dict(name="HighInterestRate", expected=+1, features=["Interest"],
         text="the interest rate is at least 40% (lender priced the applicant as risky)",
         fn=_gt("Interest", 40.0)),
    dict(name="LongLoanTerm", expected=+1, features=["LoanDuration"],
         text="the loan term is 48 months or longer",
         fn=_gt("LoanDuration", 48)),
    dict(name="FirstTimeBorrower", expected=+1, features=["NewCreditCustomer"],
         text="the applicant is a new credit customer with no track record",
         fn=lambda X: X["NewCreditCustomer"].eq(1)),
    dict(name="ShortEmploymentTenure", expected=+1, features=["EmploymentDurationCurrentEmployer"],
         text="the applicant is on a trial period or has under one year with the current employer",
         fn=lambda X: X["EmploymentDurationCurrentEmployer"].isin(["TrialPeriod", "UpTo1Year"])),
    dict(name="ProvenRepayer", expected=-1, features=["PreviousRepaymentsBeforeLoan"],
         text="the applicant has already repaid 1000 or more on earlier loans",
         fn=_gt("PreviousRepaymentsBeforeLoan", 1000)),
    dict(name="LowIncome", expected=+1, features=["IncomeTotal"],
         text="monthly income is below 800",
         fn=lambda X: X["IncomeTotal"].lt(800).fillna(False)),
]


def symbolic_predicates(X: pd.DataFrame) -> pd.DataFrame:
    """Truth value (0/1) of every rule in RULEBOOK for each applicant."""
    return pd.DataFrame({r["name"]: r["fn"](X).astype(int) for r in RULEBOOK}, index=X.index)
