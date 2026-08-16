"""
preprocess.py
-------------
Loads the loan approval dataset (4,269 records) and cleans it:
- Strips whitespace from string columns
- Encodes categorical columns
- Engineers asset/income/loan ratio features
Returns a clean, model-ready DataFrame.
"""

import pandas as pd
import numpy as np


def load_and_clean_data(path="data/loan_data.csv"):
    df = pd.read_csv(path)

    # Column names and string values in this dataset have leading spaces
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    df = df.drop(columns=["loan_id"])

    # ---- Encode categorical variables ----
    df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
    df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})
    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

    # ---- Feature engineering ----
    # Total assets the applicant holds - a proxy for collateral / financial cushion
    df["total_assets"] = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )

    # How large the loan is relative to the applicant's assets (higher = riskier)
    df["loan_to_assets_ratio"] = df["loan_amount"] / (df["total_assets"] + 1)

    # How large the loan is relative to annual income (a classic underwriting ratio)
    df["loan_to_income_ratio"] = df["loan_amount"] / df["income_annum"]

    # Log-transform skewed monetary columns to reduce the effect of outliers
    for col in ["income_annum", "loan_amount", "total_assets"]:
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    return df


if __name__ == "__main__":
    data = load_and_clean_data()
    print(data.head())
    print("\nShape:", data.shape)
    print("\nMissing values:", data.isnull().sum().sum())
