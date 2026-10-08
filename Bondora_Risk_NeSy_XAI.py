import importlib, subprocess, sys

for pkg, mod in [("shap", "shap"), ("lime", "lime"), ("xgboost", "xgboost"), ("lightgbm", "lightgbm")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("All libraries available.")

import os, json, time, warnings, textwrap
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 11})

FIG_DIR, ART_DIR = "figures", "artifacts"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=150, bbox_inches="tight")

def note(text):
    display(Markdown(text))

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

import importlib
import loan_utils as lu
importlib.reload(lu)
print("Helper module loaded. Rules in the symbolic knowledge base:", len(lu.RULEBOOK))

CANDIDATES = [
    "Bondora_Loan_Data.zip", "LoanData_Bondora.csv", "data/Bondora_Loan_Data.zip",
    "/content/Bondora_Loan_Data.zip", "/content/drive/MyDrive/Bondora_Loan_Data.zip",
    "/mnt/user-data/uploads/Bondora_Loan_Data.zip",
]
DATA_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
assert DATA_PATH is not None, "Dataset not found. Place Bondora_Loan_Data.zip next to this notebook."

raw = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded from: {DATA_PATH}")
print(f"Shape: {raw.shape[0]:,} loans x {raw.shape[1]} columns")
print(f"Report date of the snapshot: {raw['ReportAsOfEOD'].iloc[0]}")
print(f"Memory: {raw.memory_usage(deep=True).sum() / 1e6:,.0f} MB")

audit = pd.DataFrame({
    "dtype": raw.dtypes.astype(str),
    "missing_%": raw.isna().mean() * 100,
    "unique_values": raw.nunique(dropna=True),
}).sort_values("missing_%", ascending=False)

empty_cols = audit.index[audit["missing_%"] == 100].tolist()
constant_cols = audit.index[audit["unique_values"] <= 1].tolist()

print(f"Fully empty columns   ({len(empty_cols)}): {empty_cols}")
print(f"Constant columns      ({len(constant_cols)}): {constant_cols}")
print(f"Duplicate LoanId rows : {raw['LoanId'].duplicated().sum()}")
print(f"Fully duplicated rows : {raw.drop(columns=['LoanId', 'LoanNumber']).duplicated().sum()}")
print(f"Columns with any missing value: {(audit['missing_%'] > 0).sum()} of {len(audit)}")
display(audit.head(20))

raw["LoanDate"] = pd.to_datetime(raw["LoanDate"])
raw["LoanYear"] = raw["LoanDate"].dt.year
raw["DefaultFlag"] = raw["DefaultDate"].notna().astype(int)
REPORT_DATE = pd.to_datetime(raw["ReportAsOfEOD"].iloc[0])

print("Status versus default flag (all 179,235 loans):")
display(pd.crosstab(raw["Status"], raw["DefaultFlag"], margins=True))

days_to_default = (pd.to_datetime(raw["DefaultDate"]) - raw["LoanDate"]).dt.days.dropna()
print("Days from loan start to default:")
print(days_to_default.describe(percentiles=[.1, .25, .5, .75, .9, .95]).round(0).to_string())

SEASONING_CUTOFF = REPORT_DATE - pd.DateOffset(months=12)
print(f"\nSnapshot date      : {REPORT_DATE.date()}")
print(f"Seasoning cutoff   : loans issued on or before {SEASONING_CUTOFF.date()}")

is_resolved = (raw["DefaultFlag"] == 1) | (raw["Status"] == "Repaid")
is_seasoned = raw["LoanDate"] <= SEASONING_CUTOFF

steps = pd.DataFrame({
    "step": ["All loans", "Keep only loans with a final outcome", "Keep only loans with 12+ months of exposure"],
    "loans": [len(raw), int(is_resolved.sum()), int((is_resolved & is_seasoned).sum())],
})
steps["removed"] = (-steps["loans"].diff()).fillna(0).astype(int)
display(steps)

model_df = raw.loc[is_resolved & is_seasoned].copy()
print(f"\nModelling population: {len(model_df):,} loans, default rate {model_df['DefaultFlag'].mean():.1%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

by_year = raw.loc[is_resolved].groupby("LoanYear")["DefaultFlag"].agg(["mean", "size"])
colors = ["#4C78A8" if (pd.Timestamp(year=y, month=12, day=31) <= SEASONING_CUTOFF) else "#E45756" for y in by_year.index]
axes[0].bar(by_year.index.astype(str), by_year["mean"] * 100, color=colors)
axes[0].set_ylabel("Default rate among resolved loans (%)")
axes[0].set_title("Young cohorts look artificially safe (red = excluded or partly excluded)")
axes[0].tick_params(axis="x", rotation=60)

axes[1].hist(days_to_default.clip(upper=1500), bins=60, color="#72B7B2")
axes[1].axvline(days_to_default.median(), color="k", ls="--", label=f"median {days_to_default.median():.0f} days")
axes[1].axvline(365, color="#E45756", ls=":", label="12-month cutoff")
axes[1].set_xlabel("Days from loan start to default")
axes[1].set_title("Defaults take time to appear")
axes[1].legend()
plt.tight_layout(); save_fig(fig, "02_target_censoring"); plt.show()

FEATURE_COLS = lu.RAW_INPUT_COLUMNS

DROP_GROUPS = {
    "Identifiers": (
        ["ReportAsOfEOD", "LoanId", "LoanNumber", "UserName"],
        "Identify a loan or person but carry no risk signal."),
    "Dates and process timestamps": (
        ["ListedOnUTC", "BiddingStartedOn", "LoanApplicationStartedDate", "FirstPaymentDate",
         "MaturityDate_Original", "MaturityDate_Last", "ContractEndDate"],
        "Raw dates act as a proxy for vintage. ContractEndDate is only known after the loan ends."),
    "Kept only as helper metadata": (
        ["LoanDate", "LoanYear", "DefaultFlag"],
        "Used to build the target, apply the seasoning rule and run the time-based validation. Never a model input."),
    "Empty columns": (
        ["DateOfBirth", "County", "City", "EmploymentPosition"],
        "100% missing."),
    "Protected attributes (audit only)": (
        ["Gender", "LanguageCode"],
        "Not given to the model. Gender is used later to audit fairness; language is a proxy for ethnicity."),
    "Investor-side bidding": (
        ["BidsPortfolioManager", "BidsApi", "BidsManual"],
        "Investor behaviour after the loan is listed, not applicant information."),
    "Lender's own risk model outputs": (
        ["Rating", "ModelVersion", "ProbabilityOfDefault", "ExpectedLoss", "LossGivenDefault",
         "ExpectedReturn", "EL_V0", "Rating_V0", "EL_V1", "Rating_V1", "Rating_V2"],
        "These are another model's verdict. Using them would mean copying a model instead of building one."),
    "Fields Bondora stopped collecting": (
        ["UseOfLoan", "MaritalStatus", "NrOfDependants", "EmploymentStatus", "WorkExperience", "OccupationArea"],
        "Filled for loans up to about 2016, blank afterwards. A model that needs them cannot score today's applicants."),
    "Structurally empty in recent loans": (
        ["IncomeFromPrincipalEmployer", "IncomeFromPension", "IncomeFromFamilyAllowance", "IncomeFromSocialWelfare",
         "IncomeFromLeavePay", "IncomeFromChildSupport", "IncomeOther", "RefinanceLiabilities",
         "DebtToIncome", "FreeCash"],
        "Verified in 1.5: a single constant value for every loan issued 2018 onwards."),
    "Process artefacts and weak fields": (
        ["ApplicationSignedHour", "ApplicationSignedWeekday", "MonthlyPaymentDay",
         "CreditScoreEsMicroL", "CreditScoreEsEquifaxRisk", "PreviousEarlyRepaymentsBefoleLoan"],
        "Hour of day or payment day have no credit meaning. The two Spanish bureau fields are about 98% one value or missing, and the early-repayment amount is 67% missing."),
    "Post-origination outcome and servicing (leakage)": (
        ["ActiveScheduleFirstPaymentReached", "PlannedPrincipalTillDate", "PlannedInterestTillDate", "LastPaymentOn",
         "CurrentDebtDaysPrimary", "DebtOccuredOn", "CurrentDebtDaysSecondary", "DebtOccuredOnForSecondary",
         "DefaultDate", "PrincipalOverdueBySchedule", "PlannedPrincipalPostDefault", "PlannedInterestPostDefault",
         "EAD1", "EAD2", "PrincipalRecovery", "InterestRecovery", "RecoveryStage", "StageActiveSince", "Status",
         "Restructured", "ActiveLateCategory", "WorseLateCategory", "PrincipalPaymentsMade",
         "InterestAndPenaltyPaymentsMade", "PrincipalWriteOffs", "InterestAndPenaltyWriteOffs", "PrincipalBalance",
         "InterestAndPenaltyBalance", "GracePeriodStart", "GracePeriodEnd", "NextPaymentDate", "NextPaymentNr",
         "NrOfScheduledPayments", "ReScheduledOn", "PrincipalDebtServicingCost",
         "InterestAndPenaltyDebtServicingCost", "ActiveLateLastPaymentCategory"],
        "Only exist because the loan has already been repaid, paid late or defaulted. Using any of them would make the model look brilliant and be useless."),
}

original_cols = [c for c in raw.columns if c not in ("LoanYear", "DefaultFlag")]
dropped = [c for cols, _ in DROP_GROUPS.values() for c in cols if c not in ("LoanYear", "DefaultFlag")]
overlap = set(FEATURE_COLS) & set(dropped)
unassigned = set(original_cols) - set(FEATURE_COLS) - set(dropped)
unknown = set(dropped) - set(original_cols)

assert not overlap, f"Columns both kept and dropped: {overlap}"
assert not unassigned, f"Columns not assigned to any group: {unassigned}"
assert not unknown, f"Group lists a column that does not exist: {unknown}"
print(f"Check passed: {len(FEATURE_COLS)} model inputs + {len(dropped)} governed columns = {len(original_cols)} original columns.\n")

gov = pd.DataFrame(
    [(g, len([c for c in cols if c not in ("LoanYear", "DefaultFlag")]), why) for g, (cols, why) in DROP_GROUPS.items()],
    columns=["Group dropped from model inputs", "columns", "Reason"])
display(gov.style.set_properties(subset=["Reason"], **{"white-space": "pre-wrap", "text-align": "left"}).hide(axis="index"))

recent = raw[raw["LoanYear"].between(2018, 2020)]
candidates = FEATURE_COLS + ["DebtToIncome", "FreeCash", "RefinanceLiabilities", "IncomeFromPrincipalEmployer",
                              "IncomeFromPension", "IncomeFromFamilyAllowance", "IncomeFromSocialWelfare",
                              "IncomeFromLeavePay", "IncomeFromChildSupport", "IncomeOther",
                              "CreditScoreEsMicroL", "CreditScoreEsEquifaxRisk"]
rows = []
for c in candidates:
    s = recent[c]
    top_share = s.value_counts(dropna=False, normalize=True).iloc[0] * 100
    rows.append((c, s.isna().mean() * 100, top_share, s.nunique()))
avail = pd.DataFrame(rows, columns=["column", "missing_%", "most_common_value_%", "unique_values"]).set_index("column")
avail["verdict"] = np.where(
    (avail["unique_values"] <= 1) | (avail["most_common_value_%"] >= 97) | (avail["missing_%"] >= 95),
    "unusable today", "usable")

kept_flagged = avail.loc[FEATURE_COLS].query("verdict == 'unusable today'")
print("Kept features flagged unusable:", kept_flagged.index.tolist())
dropped_ok = avail.drop(index=FEATURE_COLS)
print("Dropped-for-availability columns that really are unusable:", (dropped_ok["verdict"] == "unusable today").sum(), "of", len(dropped_ok))
display(avail.sort_values("most_common_value_%", ascending=False).head(18))

zero_pay = model_df["MonthlyPayment"].eq(0)
print(f"Loans with a monthly payment of exactly 0: {zero_pay.sum():,} ({zero_pay.mean():.1%} of the modelling population)")
print(f"  share issued in 2013-2014 : {model_df.loc[zero_pay, 'LoanYear'].isin([2013, 2014]).mean():.1%}")
print(f"  default rate when payment is 0 : {model_df.loc[zero_pay, 'DefaultFlag'].mean():.1%}   (population: {model_df['DefaultFlag'].mean():.1%})")
print(f"  default rate when payment is missing : {model_df.loc[model_df['MonthlyPayment'].isna(), 'DefaultFlag'].mean():.1%}")

X_all = lu.engineer_features(model_df)
y_all = model_df["DefaultFlag"].astype(int)

assert X_all.index.equals(y_all.index)
assert not np.isinf(X_all[lu.NUMERIC_ALL].to_numpy(dtype=float)).any(), "infinite values found"

schema = pd.DataFrame({
    "type": ["numeric" if c in lu.NUMERIC_ALL else "categorical" for c in X_all.columns],
    "missing_%": X_all.isna().mean() * 100,
    "unique": X_all.nunique(),
})
print(f"Feature matrix: {X_all.shape[0]:,} rows x {X_all.shape[1]} features "
      f"({len(lu.NUMERIC_ALL)} numeric, {len(lu.CATEGORICAL)} categorical)")
display(schema)

from sklearn.model_selection import train_test_split

meta_all = model_df[["LoanYear", "Gender", "Age", "Country"]].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=SEED)
meta_train, meta_test = meta_all.loc[X_train.index], meta_all.loc[X_test.index]

print(f"Train: {len(X_train):,} loans, default rate {y_train.mean():.2%}")
print(f"Test : {len(X_test):,} loans, default rate {y_test.mean():.2%}")
assert set(X_train.index).isdisjoint(X_test.index)

eda = X_train.copy()
eda["Default"] = y_train
base_rate = y_train.mean()

cat_view = ["Country", "NewCreditCustomer", "Education", "EmploymentDurationCurrentEmployer", "VerificationType", "HomeOwnershipType"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), cat_view):
    g = eda.groupby(col)["Default"].agg(["mean", "size"])
    g = g[g["size"] >= 300].sort_values("mean")
    ax.barh(g.index.astype(str), g["mean"] * 100, color="#4C78A8")
    ax.axvline(base_rate * 100, color="#E45756", ls="--", label="overall")
    ax.set_title(f"Default rate by {col}")
    ax.set_xlabel("%")
axes[0, 0].legend()
plt.tight_layout(); save_fig(fig, "03_eda_categorical"); plt.show()

def rate_by_bucket(df, col, q=10):
    s = df[col]
    bins = pd.qcut(s, q=q, duplicates="drop")
    g = df.groupby(bins, observed=True)["Default"].mean() * 100
    mid = df.groupby(bins, observed=True)[col].median()
    return mid.values, g.values

num_view = ["Interest", "LoanDuration", "Age", "IncomeTotal", "PaymentToIncome", "AmountToIncome", "LiabToIncome", "PreviousRepaymentsBeforeLoan"]
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, col in zip(axes.ravel(), num_view):
    x, yv = rate_by_bucket(eda.dropna(subset=[col]), col, q=10)
    ax.plot(x, yv, marker="o", color="#4C78A8")
    ax.axhline(base_rate * 100, color="#E45756", ls="--", lw=1)
    ax.set_title(col)
    ax.set_ylabel("default rate %")
    if col in ("IncomeTotal", "PreviousRepaymentsBeforeLoan", "AmountToIncome", "LiabToIncome", "PaymentToIncome"):
        ax.set_xscale("symlog")
plt.suptitle("Default rate by decile of each numeric feature (bucket median on the x-axis)", y=1.01, fontweight="bold")
plt.tight_layout(); save_fig(fig, "04_eda_numeric_deciles"); plt.show()

num_cols = lu.NUMERIC_ALL
p995 = X_train[num_cols].quantile(0.995)
outlier = pd.DataFrame({
    "99.5th pct": p995,
    "max": X_train[num_cols].max(),
    "max / 99.5th pct": X_train[num_cols].max() / p995.replace(0, np.nan),
    "% above 99.5th pct": (X_train[num_cols].gt(p995)).mean() * 100,
}).sort_values("max / 99.5th pct", ascending=False)
note("**Features whose maximum is far beyond the 99.5th percentile** (these are clipped in the pipeline, see 1.9):")
display(outlier.head(8))

corr = X_train[num_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, linewidths=.3,
            cbar_kws={"shrink": .7}, ax=ax)
ax.set_title("Spearman correlation between numeric features (training data)")
plt.tight_layout(); save_fig(fig, "06_eda_correlation"); plt.show()

pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
high = pairs[pairs.abs() >= 0.80].sort_values(key=np.abs, ascending=False)
note("**Feature pairs with |Spearman| >= 0.80** (redundant information; harmless for tree models, a nuisance for linear ones):")
display(high.rename("spearman").to_frame())

assoc = X_train[num_cols].apply(lambda s: s.corr(y_train, method="spearman")).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(assoc.index, assoc.values, color=np.where(assoc.values > 0, "#E45756", "#4C78A8"))
ax.set_title("Spearman association of each numeric feature with default")
ax.axvline(0, color="k", lw=.8)
plt.tight_layout(); save_fig(fig, "07_eda_target_association"); plt.show()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

def build_preprocessor(drop=()):
    num = [c for c in lu.NUMERIC_ALL if c not in drop]
    cat = [c for c in lu.CATEGORICAL if c not in drop]
    return ColumnTransformer([
        ("num", Pipeline([("clip", lu.Winsorizer(0.005, 0.995)),
                          ("impute", SimpleImputer(strategy="median"))]), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False), cat),
    ], verbose_feature_names_out=False)

def build_pipe(estimator, scale=False, drop=()):
    steps = [("prep", build_preprocessor(drop))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)

demo_prep = build_preprocessor().fit(X_train)
A = demo_prep.transform(X_train)
feature_names = demo_prep.get_feature_names_out().tolist()
print(f"Transformed training matrix: {A.shape[0]:,} rows x {A.shape[1]} columns")
print("NaN left:", int(np.isnan(A).sum()), "| inf left:", int(np.isinf(A).sum()))
print("First 12 column names:", feature_names[:12])
print("Last 8 column names :", feature_names[-8:])

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import make_scorer, matthews_corrcoef
from scipy.stats import randint, uniform, loguniform

def make_models():
    return {
        "Logistic Regression": build_pipe(LogisticRegression(C=0.5, max_iter=1000), scale=True),
        "Decision Tree": build_pipe(DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=SEED)),
        "Random Forest": build_pipe(RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "Hist Gradient Boosting": build_pipe(HistGradientBoostingClassifier(random_state=SEED)),
        "XGBoost": build_pipe(XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                            colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)),
        "LightGBM": build_pipe(LGBMClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                              colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=SEED,
                                              deterministic=True, force_row_wise=True)),
        "Neural Network (MLP)": build_pipe(MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                         max_iter=200, random_state=SEED), scale=True),
    }

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "bal_acc": "balanced_accuracy",
    "mcc": make_scorer(matthews_corrcoef),
    "brier": "neg_brier_score",
}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def cv_row(pipe):
    t0 = time.time()
    r = cross_validate(pipe, X_train, y_train, cv=cv5, scoring=SCORING, return_train_score=False, n_jobs=1)
    out = {
        "ROC-AUC": r["test_roc_auc"].mean(), "ROC-AUC std": r["test_roc_auc"].std(),
        "PR-AUC": r["test_pr_auc"].mean(), "F1@0.5": r["test_f1"].mean(),
        "Bal.Acc@0.5": r["test_bal_acc"].mean(), "MCC@0.5": r["test_mcc"].mean(),
        "Brier": -r["test_brier"].mean(), "fit+score s/fold": (r["fit_time"] + r["score_time"]).mean(),
    }
    return out

baseline = {}
models = make_models()
for name, pipe in models.items():
    baseline[name] = cv_row(pipe)
    print(f"{name:<24} CV ROC-AUC = {baseline[name]['ROC-AUC']:.4f} +/- {baseline[name]['ROC-AUC std']:.4f}")

cv_table = pd.DataFrame(baseline).T.sort_values("ROC-AUC", ascending=False)
display(cv_table)

fig, ax = plt.subplots(figsize=(9, 4.5))
order = cv_table.sort_values("ROC-AUC")
ax.barh(order.index, order["ROC-AUC"], xerr=order["ROC-AUC std"], color="#4C78A8", capsize=3)
for i, v in enumerate(order["ROC-AUC"]):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center")
ax.set_xlim(0.6, 0.83)
ax.set_xlabel("Cross-validated ROC-AUC (mean, error bar = 1 std over 5 folds)")
ax.set_title("Model comparison on the training set")
plt.tight_layout(); save_fig(fig, "08_model_comparison"); plt.show()

SPACES = {
    "XGBoost": {
        "clf__n_estimators": randint(200, 700), "clf__learning_rate": uniform(0.02, 0.08),
        "clf__max_depth": randint(3, 9), "clf__min_child_weight": randint(1, 25),
        "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__reg_lambda": loguniform(0.5, 10)},
    "LightGBM": {
        "clf__n_estimators": randint(200, 700), "clf__learning_rate": uniform(0.02, 0.08),
        "clf__num_leaves": randint(15, 127), "clf__min_child_samples": randint(20, 200),
        "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__reg_lambda": loguniform(0.1, 10)},
    "Hist Gradient Boosting": {
        "clf__learning_rate": uniform(0.03, 0.1), "clf__max_leaf_nodes": randint(15, 80),
        "clf__min_samples_leaf": randint(20, 200), "clf__l2_regularization": loguniform(0.01, 10),
        "clf__max_iter": randint(150, 500)},
    "Random Forest": {
        "clf__n_estimators": randint(150, 300), "clf__max_depth": randint(8, 25),
        "clf__min_samples_leaf": randint(2, 30), "clf__max_features": uniform(0.2, 0.5)},
}
top2 = [n for n in cv_table.index if n in SPACES][:2]
print("Tuning:", top2)

tuned_pipes, tune_log = {}, []
for name in top2:
    t0 = time.time()
    search = RandomizedSearchCV(
        make_models()[name], SPACES[name], n_iter=8,
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="roc_auc",
        random_state=SEED, n_jobs=1, refit=False)
    search.fit(X_train, y_train)
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    tuned = make_models()[name].set_params(**search.best_params_)
    tuned_pipes[name + " (tuned)"] = tuned
    tune_log.append({"model": name, "best 3-fold AUC": search.best_score_, "minutes": (time.time() - t0) / 60, "best params": best})
    print(f"{name}: best 3-fold AUC {search.best_score_:.4f} in {(time.time() - t0) / 60:.1f} min")
pd.DataFrame(tune_log)

tuned_rows = {name: cv_row(pipe) for name, pipe in tuned_pipes.items()}
all_cv = pd.concat([cv_table, pd.DataFrame(tuned_rows).T]).sort_values("ROC-AUC", ascending=False)
display(all_cv[["ROC-AUC", "ROC-AUC std", "PR-AUC", "F1@0.5", "Bal.Acc@0.5", "MCC@0.5", "Brier"]])

all_pipes = {**make_models(), **tuned_pipes}

SHAP_TREE_OK = {"XGBoost", "LightGBM", "Random Forest"}
def base_name(n): return n.replace(" (tuned)", "")

champion_name = all_cv.index[0]
print(f"Highest cross-validated ROC-AUC: {champion_name}  ({all_cv.loc[champion_name, 'ROC-AUC']:.4f})")
if base_name(champion_name) not in SHAP_TREE_OK:
    champion_name = next(n for n in all_cv.index if base_name(n) in SHAP_TREE_OK)
    print(f"That model is not supported by the exact SHAP tree algorithm, so the explainable champion is: {champion_name}")

runner_up = next(n for n in all_cv.index if n != champion_name)
gap = all_cv.loc[champion_name, "ROC-AUC"] - all_cv.loc[runner_up, "ROC-AUC"]
print(f"Champion: {champion_name}")
print(f"Gap to runner-up ({runner_up}): {gap:.4f} AUC, against a fold-to-fold std of about {all_cv.loc[champion_name, 'ROC-AUC std']:.4f}")
final_pipe = clone(all_pipes[champion_name])

from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score,
                             precision_score, recall_score, brier_score_loss)

oof = cross_val_predict(final_pipe, X_train, y_train, cv=cv5, method="predict_proba", n_jobs=1)[:, 1]
grid = np.linspace(0.20, 0.90, 71)
curves = pd.DataFrame({
    "threshold": grid,
    "balanced_accuracy": [balanced_accuracy_score(y_train, oof >= t) for t in grid],
    "F1": [f1_score(y_train, oof >= t) for t in grid],
    "MCC": [matthews_corrcoef(y_train, oof >= t) for t in grid],
})
THRESHOLD = float(grid[curves["balanced_accuracy"].values.argmax()])
print(f"Out-of-fold ROC-AUC on training data: {roc_auc_score(y_train, oof):.4f}")
print(f"Chosen threshold (max balanced accuracy): {THRESHOLD:.2f}")

fig, ax = plt.subplots(figsize=(8, 4))
for c in ["balanced_accuracy", "F1", "MCC"]:
    ax.plot(curves["threshold"], curves[c], label=c)
ax.axvline(THRESHOLD, color="k", ls="--", label=f"chosen = {THRESHOLD:.2f}")
ax.set_xlabel("Decision threshold on P(default)"); ax.legend(); ax.set_title("Threshold selection on out-of-fold predictions")
plt.tight_layout(); save_fig(fig, "09_threshold"); plt.show()

t0 = time.time()
final_pipe.fit(X_train, y_train)
p_test = final_pipe.predict_proba(X_test)[:, 1]
p_train = final_pipe.predict_proba(X_train)[:, 1]
pred_test = (p_test >= THRESHOLD).astype(int)
print(f"Champion fitted in {time.time() - t0:.0f}s")

def bootstrap_ci(y_true, score, n=1000, seed=SEED):
    r = np.random.default_rng(seed); y_true = np.asarray(y_true); n_obs = len(y_true)
    vals = [roc_auc_score(y_true[i], score[i]) for i in (r.integers(0, n_obs, n_obs) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])

lo, hi = bootstrap_ci(y_test, p_test)
tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
test_metrics = pd.Series({
    "ROC-AUC": roc_auc_score(y_test, p_test), "ROC-AUC 95% CI low": lo, "ROC-AUC 95% CI high": hi,
    "PR-AUC": average_precision_score(y_test, p_test), "Accuracy": accuracy_score(y_test, pred_test),
    "Balanced accuracy": balanced_accuracy_score(y_test, pred_test),
    "Precision (default)": precision_score(y_test, pred_test), "Recall (default)": recall_score(y_test, pred_test),
    "Specificity (repaid)": tn / (tn + fp), "F1": f1_score(y_test, pred_test),
    "MCC": matthews_corrcoef(y_test, pred_test), "Brier score": brier_score_loss(y_test, p_test),
    "Train ROC-AUC (in-sample)": roc_auc_score(y_train, p_train),
})
display(test_metrics.to_frame(f"{champion_name}, threshold {THRESHOLD:.2f}"))

from sklearn.calibration import calibration_curve

baseline_test = {}
for name, pipe in make_models().items():
    m = pipe.fit(X_train, y_train)
    baseline_test[name] = m.predict_proba(X_test)[:, 1]
baseline_test[champion_name] = p_test

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
ax = axes[0, 0]
for name, sc in baseline_test.items():
    f, t, _ = roc_curve(y_test, sc)
    ax.plot(f, t, lw=2.5 if name == champion_name else 1, label=f"{name} ({roc_auc_score(y_test, sc):.3f})")
ax.plot([0, 1], [0, 1], "k:"); ax.set_title("ROC curves (test)"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=7)

ax = axes[0, 1]
pr, rc, _ = precision_recall_curve(y_test, p_test)
ax.plot(rc, pr, color="#4C78A8"); ax.axhline(y_test.mean(), color="k", ls=":", label="no-skill")
ax.set_title(f"Precision-recall, {champion_name}"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()

ax = axes[0, 2]
cm = confusion_matrix(y_test, pred_test, normalize="true")
sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Pred repaid", "Pred default"], yticklabels=["Actual repaid", "Actual default"])
ax.set_title(f"Confusion matrix (row-normalised), threshold {THRESHOLD:.2f}")

ax = axes[1, 0]
fr, mp = calibration_curve(y_test, p_test, n_bins=12, strategy="quantile")
ax.plot(mp, fr, marker="o", color="#4C78A8", label=champion_name); ax.plot([0, 1], [0, 1], "k:", label="perfect")
ax.set_title("Calibration (reliability) curve"); ax.set_xlabel("Mean predicted P(default)"); ax.set_ylabel("Observed default rate"); ax.legend()

ax = axes[1, 1]
for lab, colr, nm in [(0, "#54A24B", "Repaid"), (1, "#E45756", "Default")]:
    ax.hist(p_test[y_test.values == lab], bins=40, alpha=.6, color=colr, density=True, label=nm)
ax.axvline(THRESHOLD, color="k", ls="--"); ax.set_title("Score distribution by true outcome"); ax.set_xlabel("Predicted P(default)"); ax.legend()

ax = axes[1, 2]
dec = pd.qcut(p_test, 10, labels=False, duplicates="drop")
lift = pd.DataFrame({"decile": dec, "y": y_test.values}).groupby("decile")["y"].mean() * 100
ax.bar(lift.index + 1, lift.values, color="#72B7B2"); ax.axhline(y_test.mean() * 100, color="k", ls=":")
ax.set_title("Observed default rate by predicted-risk decile"); ax.set_xlabel("Risk decile (1 = safest)"); ax.set_ylabel("%")
plt.tight_layout(); save_fig(fig, "10_test_evaluation"); plt.show()

test_table = pd.Series({n: roc_auc_score(y_test, s) for n, s in baseline_test.items()}, name="Test ROC-AUC").sort_values(ascending=False)
display(test_table.to_frame())

LOANYEAR_CUT = 2018
tr_mask = meta_all["LoanYear"] <= LOANYEAR_CUT
Xt_tr, Xt_te = X_all[tr_mask], X_all[~tr_mask]
yt_tr, yt_te = y_all[tr_mask], y_all[~tr_mask]

def refit_score(drop=(), temporal=False):
    pipe = build_pipe(clone(final_pipe.named_steps["clf"]), drop=drop)
    if temporal:
        pipe.fit(Xt_tr, yt_tr); return roc_auc_score(yt_te, pipe.predict_proba(Xt_te)[:, 1])
    pipe.fit(X_train, y_train); return roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

robust = pd.DataFrame({
    "Setting": ["Random split (reference)", f"Time split: train <= {LOANYEAR_CUT}, test >= {LOANYEAR_CUT + 1}",
                "Ablation: without Interest", "Ablation: without the 5 engineered ratios",
                "Ablation: without Country"],
    "ROC-AUC": [refit_score(), refit_score(temporal=True), refit_score(("Interest",)),
                refit_score(tuple(lu.ENGINEERED)), refit_score(("Country",))],
})
robust["vs reference"] = robust["ROC-AUC"] - robust.loc[0, "ROC-AUC"]
print(f"Time-split sizes: train {len(Xt_tr):,} loans, test {len(Xt_te):,} loans")
display(robust)

import shap
from lime.lime_tabular import LimeTabularExplainer

prep = final_pipe.named_steps["prep"]
clf = final_pipe.named_steps["clf"]
assert "scale" not in final_pipe.named_steps, "explainers below assume an unscaled tree model"

feat_names = prep.get_feature_names_out().tolist()
def base_of(name):
    for c in lu.CATEGORICAL:
        if name.startswith(c + "_"):
            return c
    return name
BASE = pd.Series({n: base_of(n) for n in feat_names})

Xtr_t = pd.DataFrame(prep.transform(X_train), columns=feat_names, index=X_train.index)
Xte_t = pd.DataFrame(prep.transform(X_test), columns=feat_names, index=X_test.index)

N_SHAP = min(4000, len(Xte_t))
shap_pos = np.sort(rng.choice(len(Xte_t), size=N_SHAP, replace=False))
Xs = Xte_t.iloc[shap_pos]
Xs_raw = X_test.iloc[shap_pos]
p_s = p_test[shap_pos]
print(f"Champion: {champion_name} | explained sample: {N_SHAP:,} test applicants | {len(feat_names)} model columns "
      f"from {BASE.nunique()} original features")

explainer = shap.TreeExplainer(clf)
ex = explainer(Xs.values)
vals, base = ex.values, ex.base_values
if vals.ndim == 3:
    vals = vals[:, :, 1]
if np.ndim(base) == 2:
    base = base[:, 1]
base = np.broadcast_to(np.asarray(base, dtype=float), (len(Xs),))

is_boosted = type(clf).__name__ in ("XGBClassifier", "LGBMClassifier")
target = np.log(p_s / (1 - p_s)) if is_boosted else p_s
recon = base + vals.sum(axis=1)
add_err = float(np.abs(recon - target).max())
print(f"Scale of SHAP values: {'log-odds' if is_boosted else 'probability'}")
print(f"Additivity check, max |baseline + sum(SHAP) - model output| = {add_err:.2e}")
assert add_err < 0.05, "SHAP values do not add up to the model output"

expl = shap.Explanation(values=vals, base_values=base, data=Xs.values, feature_names=feat_names)

# one-hot columns folded back into their original feature (signed sum per applicant)
grouped = pd.DataFrame(vals, columns=feat_names).T.groupby(BASE).sum().T
grouped.index = Xs.index
importance = grouped.abs().mean().sort_values(ascending=False)
display(importance.head(12).rename("mean |SHAP| (log-odds)").to_frame())

shap.plots.beeswarm(expl, max_display=15, show=False)
fig = plt.gcf(); fig.set_size_inches(9.5, 6.5); plt.title("SHAP beeswarm: every dot is one test applicant")
plt.tight_layout(); save_fig(fig, "11_shap_beeswarm"); plt.show()

fig, ax = plt.subplots(figsize=(8, 5.5))
top = importance.head(15)[::-1]
ax.barh(top.index, top.values, color="#4C78A8")
ax.set_xlabel("Mean |SHAP value| (log-odds), one-hot columns merged into their feature")
ax.set_title("Global importance by original feature")
plt.tight_layout(); save_fig(fig, "12_shap_importance"); plt.show()

top_numeric = [f for f in importance.index if f in lu.NUMERIC_ALL][:3]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for ax, f in zip(axes, top_numeric):
    shap.dependence_plot(f, vals, Xs, feature_names=feat_names, ax=ax, show=False)
    ax.set_title(f"How {f} moves the score")
plt.tight_layout(); save_fig(fig, "13_shap_dependence"); plt.show()

cases = {
    "Highest predicted risk": int(np.argmax(p_s)),
    "Lowest predicted risk": int(np.argmin(p_s)),
    "Closest to the threshold": int(np.argmin(np.abs(p_s - THRESHOLD))),
}
for title, i in cases.items():
    shap.plots.waterfall(expl[i], max_display=11, show=False)
    fig = plt.gcf(); fig.set_size_inches(9, 5.2)
    plt.title(f"{title}: P(default) = {p_s[i]:.2f}, actual = {'default' if y_test.iloc[shap_pos[i]] else 'repaid'}")
    plt.tight_layout(); save_fig(fig, "14_shap_waterfall_" + title.split()[0].lower()); plt.show()

lime_num, lime_cat = lu.NUMERIC_ALL, lu.CATEGORICAL
lime_cols = lime_num + lime_cat
lime_med = X_train[lime_num].median()
cat_levels = {c: sorted(X_all[c].unique().tolist()) for c in lime_cat}
cat_index = {c: {v: i for i, v in enumerate(levels)} for c, levels in cat_levels.items()}

def to_lime_matrix(df):
    num = df[lime_num].fillna(lime_med).to_numpy(dtype=float)
    cat = np.column_stack([df[c].map(cat_index[c]).to_numpy(dtype=float) for c in lime_cat])
    return np.hstack([num, cat])

def from_lime_matrix(M):
    df = pd.DataFrame(M[:, :len(lime_num)], columns=lime_num)
    for j, c in enumerate(lime_cat):
        df[c] = [cat_levels[c][int(round(v))] for v in M[:, len(lime_num) + j]]
    return df[lime_cols]

def lime_predict(M):
    p1 = final_pipe.predict_proba(from_lime_matrix(M))[:, 1]
    return np.column_stack([1 - p1, p1])

lime_train = to_lime_matrix(X_train.iloc[rng.choice(len(X_train), size=20000, replace=False)])
cat_idx = list(range(len(lime_num), len(lime_cols)))

def make_lime(seed):
    return LimeTabularExplainer(
        lime_train, feature_names=lime_cols, class_names=["Repaid", "Default"],
        categorical_features=cat_idx, categorical_names={len(lime_num) + j: cat_levels[c] for j, c in enumerate(lime_cat)},
        discretize_continuous=True, mode="classification", random_state=seed)

lime_a = make_lime(SEED)
Xs_lime = to_lime_matrix(Xs_raw)

# sanity: the black box must reproduce the model's own score on the filled-in applicants
check_back = lime_predict(Xs_lime[:200])[:, 1]
print("Black-box wrapper vs model, max abs difference on 200 applicants:", float(np.abs(check_back - p_s[:200]).max()))
assert np.abs(check_back - p_s[:200]).max() < 1e-4

for title, i in cases.items():
    e = lime_a.explain_instance(Xs_lime[i], lime_predict, num_features=10, num_samples=4000)
    f = e.as_pyplot_figure(label=1); f.set_size_inches(9.5, 4.8)
    plt.title(f"LIME, {title}: model P(default) = {p_s[i]:.2f}, local surrogate R² = {e.score:.2f}")
    plt.tight_layout(); save_fig(f, "15_lime_" + title.split()[0].lower()); plt.show()

N_CMP = 40
lime_b = make_lime(SEED + 1)

def lime_weights(explainer_obj, x_row):
    e = explainer_obj.explain_instance(x_row, lime_predict, num_features=len(lime_cols), num_samples=4000)
    return pd.Series({lime_cols[i]: wt for i, wt in e.as_map()[1]}), e.score

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)

rows, r2s, stab = [], [], []
for j in range(N_CMP):
    s_vec = grouped.iloc[j]
    lw_a, r2 = lime_weights(lime_a, Xs_lime[j]); r2s.append(r2)
    top_s = s_vec.abs().sort_values(ascending=False).index[:5]
    top_l = lw_a.abs().sort_values(ascending=False).index[:5]
    shared = [f for f in top_s if f in top_l]
    sign_ok = np.mean([np.sign(s_vec[f]) == np.sign(lw_a[f]) for f in shared]) if shared else np.nan
    rows.append({"jaccard_top5": jaccard(top_s, top_l), "top1_in_other_top3": top_s[0] in list(lw_a.abs().sort_values(ascending=False).index[:3]),
                 "top1_same": top_s[0] == top_l[0], "sign_agree_on_shared": sign_ok})
    if j < 20:
        lw_b, _ = lime_weights(lime_b, Xs_lime[j])
        stab.append(jaccard(top_l, lw_b.abs().sort_values(ascending=False).index[:5]))

agree = pd.DataFrame(rows)
lime_summary = {
    "shap_vs_lime_top5_jaccard": agree["jaccard_top5"].mean(),
    "shap_vs_lime_top1_same": agree["top1_same"].mean(),
    "shap_top1_inside_lime_top3": agree["top1_in_other_top3"].mean(),
    "shap_vs_lime_sign_agreement": agree["sign_agree_on_shared"].mean(),
    "lime_local_r2_mean": float(np.mean(r2s)),
    "lime_seed_stability_top5_jaccard": float(np.mean(stab)),
}
display(pd.Series(lime_summary, name="value").to_frame())

kb_tr = lu.symbolic_predicates(X_train)
kb_te = lu.symbolic_predicates(X_test)
kb_s = lu.symbolic_predicates(Xs_raw)

kb_rows = []
for r in lu.RULEBOOK:
    n = r["name"]; m = kb_tr[n] == 1
    kb_rows.append({"rule": n, "IF ...": r["text"], "expert expects": "raises risk" if r["expected"] > 0 else "lowers risk",
                    "applicants covered": m.mean(), "default rate if true": y_train[m].mean(),
                    "default rate if false": y_train[~m].mean()})
kb_table = pd.DataFrame(kb_rows).set_index("rule")
display(kb_table.style.format({"applicants covered": "{:.1%}", "default rate if true": "{:.1%}", "default rate if false": "{:.1%}"})
        .set_properties(subset=["IF ..."], **{"white-space": "pre-wrap", "text-align": "left"}))

align_rows = []
for r in lu.RULEBOOK:
    n = r["name"]; m = kb_tr[n] == 1
    emp = y_train[m].mean() - y_train[~m].mean()
    ms = kb_s[n] == 1
    feat_cols = [f for f in r["features"] if f in grouped.columns]
    shap_vec = grouped[feat_cols].sum(axis=1)
    sh = shap_vec[ms.values].mean() - shap_vec[~ms.values].mean() if ms.any() and (~ms).any() else np.nan
    emp_ok, sh_ok = np.sign(emp) == r["expected"], np.sign(sh) == r["expected"]
    if emp_ok and sh_ok:            verdict = "aligned"
    elif (not emp_ok) and (not sh_ok): verdict = "CONFLICT: data and model both contradict the expert"
    elif emp_ok and (not sh_ok):    verdict = "CONFLICT: model contradicts expert and data"
    else:                           verdict = "CONFLICT: data contradicts expert, model follows expert"
    align_rows.append({"rule": n, "expert": "+" if r["expected"] > 0 else "-", "raw data effect (default-rate gap)": emp,
                       "model effect (mean SHAP gap)": sh, "verdict": verdict})
alignment = pd.DataFrame(align_rows).set_index("rule")
display(alignment.style.format({"raw data effect (default-rate gap)": "{:+.3f}", "model effect (mean SHAP gap)": "{:+.3f}"}))
print(f"Rules fully aligned: {(alignment['verdict'] == 'aligned').sum()} of {len(alignment)}")

ctr_train = meta_train["Country"]
strat = {}
for c in ["EE", "ES", "FI"]:
    mc = (ctr_train == c).values
    strat[c] = {r["name"]: (y_train[mc & (kb_tr[r["name"]] == 1).values].mean() - y_train[mc & (kb_tr[r["name"]] == 0).values].mean())
                for r in lu.RULEBOOK}
strat = pd.DataFrame(strat)
strat.insert(0, "pooled", alignment["raw data effect (default-rate gap)"])
strat.insert(0, "expert", ["+" if r["expected"] > 0 else "-" for r in lu.RULEBOOK])
strat["holds in all 3 countries"] = [
    all(np.sign(strat.loc[n, c]) == (1 if strat.loc[n, "expert"] == "+" else -1) for c in ["EE", "ES", "FI"]) for n in strat.index]
display(strat.style.format({"pooled": "{:+.3f}", "EE": "{:+.3f}", "ES": "{:+.3f}", "FI": "{:+.3f}"}))

scorecard = LogisticRegression(C=1.0, max_iter=1000).fit(kb_tr, y_train)
p_sym_tr = scorecard.predict_proba(kb_tr)[:, 1]
p_sym_te = scorecard.predict_proba(kb_te)[:, 1]
fpr_, tpr_, thr_ = roc_curve(y_train, p_sym_tr)
THR_SYM = float(thr_[np.argmax(tpr_ - fpr_)])

weights = pd.DataFrame({
    "weight (log-odds)": scorecard.coef_[0],
    "odds multiplier": np.exp(scorecard.coef_[0]),
    "expert expects": [("+" if r["expected"] > 0 else "-") for r in lu.RULEBOOK],
}, index=kb_tr.columns)
weights["sign matches expert"] = np.sign(weights["weight (log-odds)"]) == weights["expert expects"].map({"+": 1, "-": -1})
display(weights.sort_values("weight (log-odds)", ascending=False))
print(f"Bias (log-odds when no rule fires): {scorecard.intercept_[0]:+.3f}")
print(f"Scorecard test ROC-AUC = {roc_auc_score(y_test, p_sym_te):.4f}   vs   champion test ROC-AUC = {roc_auc_score(y_test, p_test):.4f}")
print(f"Scorecard threshold = {THR_SYM:.3f}")

top_num = [f for f in importance.index if f in lu.NUMERIC_ALL][:6]
top_cat = [f for f in importance.index if f in lu.CATEGORICAL][:3]

atom_specs = []
for f in top_num:
    for q in sorted(set(np.round(X_train[f].quantile([.25, .5, .75]).values, 4))):
        atom_specs.append(("num", f, float(q)))
for f in top_cat:
    freq = X_train[f].value_counts(normalize=True)
    for v in freq[freq >= 0.05].index[:4]:
        atom_specs.append(("cat", f, v))

def build_atoms(Xe):
    cols = {r["name"]: r["fn"](Xe).astype(int) for r in lu.RULEBOOK}
    for kind, f, v in atom_specs:
        if kind == "num":
            cols[f"{f} >= {v:g}"] = (Xe[f] >= v).astype(int)
        else:
            cols[f"{f} = {v}"] = (Xe[f] == v).astype(int)
    return pd.DataFrame(cols, index=Xe.index)

At_tr, At_te = build_atoms(X_train), build_atoms(X_test)
keep = [c for c in At_tr.columns if 0.02 <= At_tr[c].mean() <= 0.98]
At_tr, At_te = At_tr[keep], At_te[keep]
bb_tr = (p_train >= THRESHOLD).astype(int)
bb_te = (p_test >= THRESHOLD).astype(int)
print(f"Symbolic vocabulary: {At_tr.shape[1]} yes/no tests ({len(lu.RULEBOOK)} expert rules + {At_tr.shape[1] - len(lu.RULEBOOK)} mined tests)")

sweep = []
for d in range(1, 8):
    t_ = DecisionTreeClassifier(max_depth=d, min_samples_leaf=300, random_state=SEED).fit(At_tr, bb_tr)
    sweep.append({"max depth": d, "rules (leaves)": t_.get_n_leaves(),
                  "fidelity on train": (t_.predict(At_tr) == bb_tr).mean(), "fidelity on test": (t_.predict(At_te) == bb_te).mean()})
sweep = pd.DataFrame(sweep).set_index("max depth")
display(sweep.style.format({"fidelity on train": "{:.1%}", "fidelity on test": "{:.1%}"}))

RULE_DEPTH = 4
surrogate = DecisionTreeClassifier(max_depth=RULE_DEPTH, min_samples_leaf=300, random_state=SEED).fit(At_tr, bb_tr)
leaf_te = surrogate.apply(At_te.values)
atom_names = At_tr.columns.tolist()
t_ = surrogate.tree_
rule_rows = []

def walk(node, conds):
    if t_.children_left[node] == -1:
        cls = int(np.argmax(t_.value[node][0]))
        m = leaf_te == node
        rule_rows.append({
            "IF": " AND ".join(conds) if conds else "(always)", "THEN": "FLAG" if cls == 1 else "PASS",
            "test applicants %": m.mean() * 100, "champion mean P(default)": p_test[m].mean(),
            "observed default rate": y_test.values[m].mean(),
            "champion agrees %": (bb_te[m] == cls).mean() * 100})
        return
    name = atom_names[t_.feature[node]]
    walk(t_.children_left[node], conds + [f"NOT [{name}]"])
    walk(t_.children_right[node], conds + [f"[{name}]"])
walk(0, [])

rules_df = pd.DataFrame(rule_rows).sort_values("test applicants %", ascending=False).reset_index(drop=True)
rules_df.index = [f"R{i + 1}" for i in rules_df.index]
fidelity_test = (surrogate.predict(At_te) == bb_te).mean()
surrogate_auc_true = roc_auc_score(y_test, surrogate.predict_proba(At_te)[:, 1])
print(f"{len(rules_df)} rules | fidelity to champion on unseen test applicants: {fidelity_test:.1%} | "
      f"rule-list ROC-AUC against true outcomes: {surrogate_auc_true:.3f} (champion {roc_auc_score(y_test, p_test):.3f})\n")
pd.set_option("display.max_colwidth", 140)
display(rules_df.style.format({"test applicants %": "{:.1f}", "champion mean P(default)": "{:.2f}",
                               "observed default rate": "{:.2f}", "champion agrees %": "{:.1f}"})
        .set_properties(subset=["IF"], **{"white-space": "pre-wrap", "text-align": "left"}))

raw_test = model_df.loc[X_test.index, lu.RAW_INPUT_COLUMNS].copy()
EPS = 0.005

CONSTRAINTS = [
    ("Interest +5 points", +1, lambda r: r.assign(Interest=r["Interest"] + 5),
     "higher interest should not lower risk"),
    ("Loan term +6 months (max 60)", +1, lambda r: r.assign(LoanDuration=np.minimum(r["LoanDuration"] + 6, 60)),
     "a longer exposure should not lower risk"),
    ("Monthly income +20%", -1, lambda r: r.assign(IncomeTotal=r["IncomeTotal"] * 1.2),
     "more income should not raise risk"),
    ("Monthly liabilities +20%", +1, lambda r: r.assign(LiabilitiesTotal=r["LiabilitiesTotal"] * 1.2),
     "more existing debt should not lower risk"),
    ("Earlier repayments +500", -1, lambda r: r.assign(PreviousRepaymentsBeforeLoan=r["PreviousRepaymentsBeforeLoan"].fillna(0) + 500),
     "a better repayment record should not raise risk"),
]

def check_constraints(model):
    base_p = model.predict_proba(X_test)[:, 1]
    rows, viol = [], {}
    for name, sign, fn, why in CONSTRAINTS:
        d = model.predict_proba(lu.engineer_features(fn(raw_test)))[:, 1] - base_p
        v = (d * sign) < -EPS
        viol[name] = v
        rows.append({"constraint": name, "expected rule": why, "mean change in P(default)": d.mean(), "applicants violating": v.mean()})
    return pd.DataFrame(rows).set_index("constraint"), pd.DataFrame(viol, index=X_test.index)

con_df, viol_df = check_constraints(final_pipe)
n_viol = viol_df.sum(axis=1).values
display(con_df.style.format({"mean change in P(default)": "{:+.4f}", "applicants violating": "{:.1%}"}))

acc_by_viol = pd.DataFrame({"violations": np.minimum(n_viol, 3), "correct": (pred_test == y_test.values)}).groupby("violations")["correct"].agg(["size", "mean"])
acc_by_viol.index = [{0: "0", 1: "1", 2: "2", 3: "3 or more"}[i] for i in acc_by_viol.index]
acc_by_viol.columns = ["applicants", "champion accuracy"]
note("**Champion accuracy against how many expert constraints it violates for the same applicant:**")
display(acc_by_viol.style.format({"champion accuracy": "{:.1%}"}))

MONO = {"Interest": +1, "LoanDuration": +1, "PaymentToIncome": +1, "PreviousRepaymentsBeforeLoan": -1, "PrevRepaymentRatio": -1}
mono_vec = tuple(MONO.get(n, 0) for n in feat_names)
assert base_name(champion_name) in ("XGBoost", "LightGBM"), "monotone constraints are set up for XGBoost or LightGBM"
mono_pipe = clone(final_pipe).set_params(clf__monotone_constraints=mono_vec)
mono_pipe.fit(X_train, y_train)

p_mono = mono_pipe.predict_proba(X_test)[:, 1]
con_mono, viol_mono = check_constraints(mono_pipe)
cmp_mono = pd.DataFrame({
    "violations, champion": con_df["applicants violating"],
    "violations, constrained": con_mono["applicants violating"],
    "enforced?": ["yes" if n in ("Interest +5 points", "Loan term +6 months (max 60)", "Earlier repayments +500") else "no (data disagree)" for n in con_df.index],
})
display(cmp_mono.style.format({"violations, champion": "{:.1%}", "violations, constrained": "{:.1%}"}))

mono_summary = pd.DataFrame({
    "ROC-AUC": [roc_auc_score(y_test, p_test), roc_auc_score(y_test, p_mono)],
    "Balanced accuracy @ same threshold": [balanced_accuracy_score(y_test, p_test >= THRESHOLD), balanced_accuracy_score(y_test, p_mono >= THRESHOLD)],
    "Brier": [brier_score_loss(y_test, p_test), brier_score_loss(y_test, p_mono)],
}, index=["Champion (unconstrained)", "Knowledge-constrained variant"])
display(mono_summary)
print(f"Accuracy cost of the knowledge constraints: {roc_auc_score(y_test, p_test) - roc_auc_score(y_test, p_mono):+.4f} ROC-AUC")

d_nn = p_test >= THRESHOLD
d_sym = p_sym_te >= THR_SYM
verdict = np.select(
    [d_nn & d_sym, ~d_nn & ~d_sym, d_nn & ~d_sym],
    ["Consensus: default", "Consensus: repay", "Conflict: champion says default, rules say repay"],
    default="Conflict: champion says repay, rules say default")
y_arr = y_test.values
correct_nn = (d_nn.astype(int) == y_arr)
correct_sym = (d_sym.astype(int) == y_arr)

tri = pd.DataFrame({"verdict": verdict, "actual": y_arr, "nn_ok": correct_nn, "sym_ok": correct_sym}).groupby("verdict").agg(
    applicants=("actual", "size"), actual_default_rate=("actual", "mean"), champion_accuracy=("nn_ok", "mean"), rules_accuracy=("sym_ok", "mean"))
tri["share"] = tri["applicants"] / tri["applicants"].sum()
display(tri[["applicants", "share", "actual_default_rate", "champion_accuracy", "rules_accuracy"]]
        .style.format({"share": "{:.1%}", "actual_default_rate": "{:.1%}", "champion_accuracy": "{:.1%}", "rules_accuracy": "{:.1%}"}))

is_conflict = np.char.startswith(verdict.astype(str), "Conflict")
review_share = is_conflict.mean()
acc_all = correct_nn.mean()
acc_auto = correct_nn[~is_conflict].mean()
acc_review = correct_nn[is_conflict].mean()

# same review budget, but chosen by champion uncertainty alone
dist = np.abs(p_test - THRESHOLD)
cut = np.quantile(dist, review_share)
conf_review = dist <= cut
acc_conf_auto = correct_nn[~conf_review].mean()

triage = pd.DataFrame({
    "applicants sent to a human": [0, review_share, conf_review.mean()],
    "accuracy on automatic decisions": [acc_all, acc_auto, acc_conf_auto],
}, index=["No triage", "Neurosymbolic: review when champion and rules disagree", "Baseline: review the least-confident champion scores (same volume)"])
display(triage.style.format({"applicants sent to a human": "{:.1%}", "accuracy on automatic decisions": "{:.1%}"}))
print(f"Champion accuracy inside the conflict group: {acc_review:.1%}")

BASE_VALUE_TXT = {"Interest": "{:.1f}%", "LoanDuration": "{:.0f} months", "IncomeTotal": "{:,.0f}/month",
                  "PaymentToIncome": "{:.0%} of income", "LiabToIncome": "{:.0%} of income", "AmountToIncome": "{:.1f}x income",
                  "Age": "{:.0f} years", "Amount": "{:,.0f}", "AppliedAmount": "{:,.0f}", "MonthlyPayment": "{:,.0f}",
                  "LiabilitiesTotal": "{:,.0f}", "PreviousRepaymentsBeforeLoan": "{:,.0f}"}

def fmt_val(feat, row):
    v = row[feat]
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "not available"
    return BASE_VALUE_TXT.get(feat, "{:,.2f}").format(v)

def nesy_report(pos):
    row = X_test.iloc[[pos]]
    xt = prep.transform(row)
    e1 = explainer(xt)
    v1 = e1.values[0] if e1.values.ndim == 2 else e1.values[0, :, 1]
    g1 = pd.Series(v1, index=feat_names).groupby(BASE).sum()
    drivers = g1.reindex(g1.abs().sort_values(ascending=False).index)[:4]

    fired = [(n, weights.loc[n, "weight (log-odds)"]) for n in kb_te.columns if kb_te.iloc[pos][n] == 1]
    fired.sort(key=lambda t: -abs(t[1]))
    viol_here = [n for n in viol_df.columns if viol_df.iloc[pos][n]]
    actual = "DEFAULT" if y_arr[pos] else "REPAID"

    lines = [
        f"APPLICANT (test row {pos})   |   actual outcome: {actual}",
        f"  Learned model : P(default) = {p_test[pos]:.2f}  ->  {'DEFAULT' if d_nn[pos] else 'REPAY'} (threshold {THRESHOLD:.2f})",
        f"  Written rules : P(default) = {p_sym_te[pos]:.2f}  ->  {'DEFAULT' if d_sym[pos] else 'REPAY'} (threshold {THR_SYM:.2f})",
        f"  Reasoning outcome: {verdict[pos]}",
        f"  Action: {'route to a human credit officer' if is_conflict[pos] else 'automatic decision supported by both views'}",
        "  Rules that fire (weight in log-odds): " + (", ".join(f"{n} ({w:+.2f})" for n, w in fired) if fired else "none"),
        "  Strongest model drivers (SHAP):",
    ]
    for f, val in drivers.items():
        lines.append(f"     - {f} = {fmt_val(f, row.iloc[0])}  {'raises' if val > 0 else 'lowers'} risk ({val:+.2f} log-odds)")
    lines.append("  Expert constraints violated by the model: " + (", ".join(viol_here) if viol_here else "none"))
    print("\n".join(lines) + "\n")

show = {
    "Consensus default": np.where(verdict == "Consensus: default")[0],
    "Consensus repay": np.where(verdict == "Consensus: repay")[0],
    "Conflict": np.where(is_conflict)[0],
}
nesy_report(int(show["Consensus default"][np.argmax(p_test[show["Consensus default"]])]))
nesy_report(int(show["Consensus repay"][np.argmin(p_test[show["Consensus repay"]])]))
nesy_report(int(show["Conflict"][np.argmin(np.abs(p_test[show["Conflict"]] - THRESHOLD))]))

compare = pd.DataFrame({
    "SHAP": {
        "Explains": "Each feature's push on one score, and global feature importance",
        "Faithfulness evidence": f"Exact: values add up to the model output (max error {add_err:.1e})",
        "Stability": "Deterministic: same input, same explanation",
        "Speed / cost": "Fast for tree models",
        "Speaks to": "Data scientists, model validators",
    },
    "LIME": {
        "Explains": "A local linear approximation around one applicant",
        "Faithfulness evidence": f"Local surrogate R² averages {lime_summary['lime_local_r2_mean']:.2f}",
        "Stability": f"Top-5 features overlap {lime_summary['lime_seed_stability_top5_jaccard']:.2f} (Jaccard) when only the random seed changes",
        "Speed / cost": "Slower: thousands of model calls per applicant",
        "Speaks to": "Analysts who want a quick second opinion",
    },
    "Neurosymbolic": {
        "Explains": "Model behaviour as readable rules, checked against written credit knowledge",
        "Faithfulness evidence": f"Rule list reproduces {fidelity_test:.1%} of champion decisions on unseen applicants",
        "Stability": "Deterministic rules and fixed constraints",
        "Speed / cost": "Cheap at decision time, needs a curated rule base",
        "Speaks to": "Credit officers, auditors, regulators, applicants",
    },
})
display(compare.style.set_properties(**{"white-space": "pre-wrap", "text-align": "left"}))
print(f"SHAP vs LIME agreement on the top-5 features (Jaccard): {lime_summary['shap_vs_lime_top5_jaccard']:.2f} | "
      f"same top-1 feature: {lime_summary['shap_vs_lime_top1_same']:.0%} | SHAP's top feature inside LIME's top 3: {lime_summary['shap_top1_inside_lime_top3']:.0%} | same direction on shared features: {lime_summary['shap_vs_lime_sign_agreement']:.0%}")

def group_report(groups, name, mask=None):
    keep_rows = np.ones(len(y_arr), dtype=bool) if mask is None else np.asarray(mask)
    d = pd.DataFrame({"group": np.asarray(groups)[keep_rows] if mask is None else np.asarray(groups),
                      "y": y_arr[keep_rows], "p": p_test[keep_rows], "flag": pred_test[keep_rows], "review": is_conflict[keep_rows]})
    out = []
    for g, s in d.groupby("group", observed=True):
        if len(s) < 200 or s["y"].nunique() < 2:
            continue
        out.append({"attribute": name, "group": str(g), "applicants": len(s), "actual default rate": s["y"].mean(),
                    "flagged as default": s["flag"].mean(), "TPR (defaulters caught)": s.loc[s.y == 1, "flag"].mean(),
                    "FPR (good borrowers flagged)": s.loc[s.y == 0, "flag"].mean(), "ROC-AUC": roc_auc_score(s["y"], s["p"]),
                    "sent to human review": s["review"].mean()})
    return pd.DataFrame(out)

gender_lbl = meta_test["Gender"].map(lambda v: "missing" if pd.isna(v) else f"code {int(v)}")
age_lbl = pd.cut(meta_test["Age"], [0, 24, 34, 44, 54, 200], labels=["<=24", "25-34", "35-44", "45-54", "55+"])
fair = pd.concat([group_report(gender_lbl, "Gender (not a model input)"),
                  group_report(age_lbl, "Age band (model input)"),
                  group_report(meta_test["Country"], "Country (model input)")], ignore_index=True)
display(fair.set_index(["attribute", "group"]).style.format({
    "actual default rate": "{:.1%}", "flagged as default": "{:.1%}", "TPR (defaulters caught)": "{:.1%}",
    "FPR (good borrowers flagged)": "{:.1%}", "ROC-AUC": "{:.3f}", "sent to human review": "{:.1%}"}))

summ = []
for a, g in fair.groupby("attribute", sort=False):
    appr = 1 - g["flagged as default"]
    summ.append({"attribute": a, "approval-rate ratio (min / max)": appr.min() / appr.max(),
                 "TPR gap (max - min)": g["TPR (defaulters caught)"].max() - g["TPR (defaulters caught)"].min(),
                 "FPR gap (max - min)": g["FPR (good borrowers flagged)"].max() - g["FPR (good borrowers flagged)"].min(),
                 "AUC range": g["ROC-AUC"].max() - g["ROC-AUC"].min()})
fair_summary = pd.DataFrame(summ).set_index("attribute")
display(fair_summary.style.format("{:.3f}"))

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for ax, (a, g) in zip(axes, fair.groupby("attribute", sort=False)):
    x = np.arange(len(g)); w = 0.2
    ax.bar(x - 1.5 * w, g["actual default rate"], w, label="actual default rate", color="#9D9D9D")
    ax.bar(x - 0.5 * w, g["flagged as default"], w, label="flagged as default", color="#4C78A8")
    ax.bar(x + 0.5 * w, g["TPR (defaulters caught)"], w, label="TPR", color="#54A24B")
    ax.bar(x + 1.5 * w, g["FPR (good borrowers flagged)"], w, label="FPR", color="#E45756")
    ax.set_xticks(x); ax.set_xticklabels(g["group"]); ax.set_title(a); ax.set_ylim(0, 1)
axes[0].legend(fontsize=8, loc="upper left")
plt.tight_layout(); save_fig(fig, "16_fairness_audit"); plt.show()

ctab = pd.crosstab(meta_test["Gender"].map(lambda v: "missing" if pd.isna(v) else f"code {int(v)}"), meta_test["Country"])
note("**Test applicants by gender code and country:**")
display(ctab)

within = pd.concat([group_report(gender_lbl[(meta_test["Country"] == c).values], f"Gender within {c}", mask=(meta_test["Country"] == c).values)
                    for c in ["EE", "ES", "FI"]], ignore_index=True)
display(within.set_index(["attribute", "group"]).style.format({
    "actual default rate": "{:.1%}", "flagged as default": "{:.1%}", "TPR (defaulters caught)": "{:.1%}",
    "FPR (good borrowers flagged)": "{:.1%}", "ROC-AUC": "{:.3f}", "sent to human review": "{:.1%}"}))

cal = pd.DataFrame({"country": meta_test["Country"].values, "p": p_test, "y": y_arr}).groupby("country").agg(
    applicants=("y", "size"), mean_predicted=("p", "mean"), observed_default_rate=("y", "mean"))
note("**Calibration by country (test):**")
display(cal.style.format({"mean_predicted": "{:.3f}", "observed_default_rate": "{:.3f}"}))

tr_country = meta_train["Country"].values
good_tr = y_train.values == 0
global_fpr = (oof[good_tr] >= THRESHOLD).mean()
thr_by_country = {}
for c in pd.Series(tr_country).unique():
    sel = (tr_country == c)
    thr_by_country[c] = float(np.quantile(oof[sel & good_tr], 1 - global_fpr)) if sel.sum() >= 1000 else THRESHOLD
print(f"Target false-positive rate (global threshold {THRESHOLD:.2f} on training OOF): {global_fpr:.1%}")
print("Per-country thresholds:", {k: round(v, 3) for k, v in thr_by_country.items()})

thr_vec = meta_test["Country"].map(thr_by_country).to_numpy()
pred_eq = (p_test >= thr_vec).astype(int)

def rates(pred, groups):
    d = pd.DataFrame({"g": np.asarray(groups), "y": y_arr, "f": pred})
    return d.groupby("g").apply(lambda s: pd.Series({"flagged": s["f"].mean(), "TPR": s.loc[s.y == 1, "f"].mean(), "FPR": s.loc[s.y == 0, "f"].mean()}))

r_glob, r_eq = rates(pred_test, meta_test["Country"]), rates(pred_eq, meta_test["Country"])
mit = pd.concat({"global threshold": r_glob, "per-country threshold": r_eq}, axis=1)
display(mit.style.format("{:.1%}"))

mit_summary = pd.DataFrame({
    "balanced accuracy": [balanced_accuracy_score(y_test, pred_test), balanced_accuracy_score(y_test, pred_eq)],
    "FPR gap across EE/ES/FI": [r_glob.loc[["EE", "ES", "FI"], "FPR"].max() - r_glob.loc[["EE", "ES", "FI"], "FPR"].min(),
                                r_eq.loc[["EE", "ES", "FI"], "FPR"].max() - r_eq.loc[["EE", "ES", "FI"], "FPR"].min()],
    "TPR gap across EE/ES/FI": [r_glob.loc[["EE", "ES", "FI"], "TPR"].max() - r_glob.loc[["EE", "ES", "FI"], "TPR"].min(),
                                r_eq.loc[["EE", "ES", "FI"], "TPR"].max() - r_eq.loc[["EE", "ES", "FI"], "TPR"].min()],
    "overall share flagged": [pred_test.mean(), pred_eq.mean()],
}, index=["Global threshold", "Per-country thresholds (equal FPR)"])
display(mit_summary.style.format("{:.3f}"))

noage = build_pipe(clone(final_pipe.named_steps["clf"]), drop=("Age",)).fit(X_train, y_train)
p_noage = noage.predict_proba(X_test)[:, 1]
thr_noage = float(np.quantile(p_noage, 1 - pred_test.mean()))
flag_noage = (p_noage >= thr_noage).astype(int)

def age_table(flag, score):
    d = pd.DataFrame({"band": age_lbl.to_numpy(), "y": y_arr, "f": flag, "p": score})
    return d.groupby("band", observed=True).apply(lambda s: pd.Series({
        "FPR": s.loc[s.y == 0, "f"].mean(), "TPR": s.loc[s.y == 1, "f"].mean(), "AUC": roc_auc_score(s["y"], s["p"])}))

at = pd.concat({"with Age (champion)": age_table(pred_test, p_test), "without Age": age_table(flag_noage, p_noage)}, axis=1)
display(at.style.format("{:.3f}"))
age_cmp = pd.DataFrame({
    "overall ROC-AUC": [roc_auc_score(y_test, p_test), roc_auc_score(y_test, p_noage)],
    "FPR gap across age bands": [at["with Age (champion)"]["FPR"].max() - at["with Age (champion)"]["FPR"].min(),
                                 at["without Age"]["FPR"].max() - at["without Age"]["FPR"].min()],
}, index=["With Age (champion)", "Without Age"])
display(age_cmp.style.format("{:.3f}"))

import joblib

joblib.dump(final_pipe, f"{ART_DIR}/default_risk_pipeline.joblib")
joblib.dump(scorecard, f"{ART_DIR}/symbolic_scorecard.joblib")
joblib.dump(mono_pipe, f"{ART_DIR}/default_risk_pipeline_monotone.joblib")

schema_json = {
    "model": champion_name, "positive_class": "default (1)", "decision_threshold": THRESHOLD,
    "symbolic_scorecard_threshold": THR_SYM,
    "raw_input_columns": lu.RAW_INPUT_COLUMNS, "numeric": lu.NUMERIC_ALL, "categorical": lu.CATEGORICAL,
    "sdgs": ["SDG 8 Decent Work and Economic Growth", "SDG 10 Reduced Inequalities"],
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "symbolic_rules": [{"name": r["name"], "meaning": r["text"], "expected_direction": r["expected"],
                        "scorecard_weight": float(weights.loc[r["name"], "weight (log-odds)"])} for r in lu.RULEBOOK],
    "mined_rule_list_fidelity": float(fidelity_test),
    "knowledge_constrained_variant": {"file": "default_risk_pipeline_monotone.joblib", "roc_auc_test": float(roc_auc_score(y_test, p_mono)),
                                      "monotone_features": MONO},
    "per_country_thresholds_equal_fpr": thr_by_country,
}
with open(f"{ART_DIR}/model_card.json", "w") as fh:
    json.dump(schema_json, fh, indent=2)

all_cv.to_csv(f"{ART_DIR}/cv_results.csv")
test_metrics.to_csv(f"{ART_DIR}/test_metrics.csv", header=["value"])
rules_df.to_csv(f"{ART_DIR}/mined_rules.csv")
fair.to_csv(f"{ART_DIR}/fairness_audit.csv", index=False)
alignment.to_csv(f"{ART_DIR}/knowledge_alignment.csv")
X_test.head(200).to_csv(f"{ART_DIR}/sample_applicants.csv")

check = r'''
import joblib, pandas as pd, json
card = json.load(open("artifacts/model_card.json"))
pipe = joblib.load("artifacts/default_risk_pipeline.joblib")
X = pd.read_csv("artifacts/sample_applicants.csv", index_col=0, dtype={c: str for c in card["categorical"]})
print(json.dumps([float(v) for v in pipe.predict_proba(X)[:, 1]]))
'''
out = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
assert out.returncode == 0, out.stderr[-800:]
fresh = np.array(json.loads(out.stdout.strip().splitlines()[-1]))
ref = final_pipe.predict_proba(X_test.head(200))[:, 1]
print("Fresh-process reload OK:", len(fresh), "applicants scored")
print("Max difference vs the in-memory model:", float(np.abs(fresh - ref).max()))
assert np.allclose(fresh, ref, atol=1e-6)
print("Saved files:", sorted(os.listdir(ART_DIR)))
