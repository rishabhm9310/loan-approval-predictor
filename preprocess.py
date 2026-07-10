"""
preprocess.py
-------------
Loads the raw loan dataset and cleans it:
- Fills missing values
- Encodes categorical columns
- Engineers a couple of useful features
Returns a clean, model-ready DataFrame.
"""

import pandas as pd
import numpy as np


def load_and_clean_data(path="data/loan_data.csv"):
    df = pd.read_csv(path)

    # Drop the ID column - it carries no predictive signal
    df = df.drop(columns=["Loan_ID"])

    # ---- Handle missing values ----
    # Categorical columns -> fill with mode (most frequent value)
    cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Numerical columns -> fill with median (robust to outliers)
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())

    # ---- Feature engineering ----
    # Total household income is often more predictive than applicant income alone
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Log-transform skewed numeric columns to reduce the effect of outliers
    df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
    df["TotalIncome_log"] = np.log1p(df["TotalIncome"])

    # EMI-like ratio: loan amount relative to income (higher = riskier)
    df["Loan_Income_Ratio"] = df["LoanAmount"] / (df["TotalIncome"] + 1)

    # ---- Encode categorical variables ----
    # Dependents has a "3+" category -> convert to numeric
    df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    binary_maps = {
        "Gender": {"Male": 1, "Female": 0},
        "Married": {"Yes": 1, "No": 0},
        "Education": {"Graduate": 1, "Not Graduate": 0},
        "Self_Employed": {"Yes": 1, "No": 0},
        "Loan_Status": {"Y": 1, "N": 0},
    }
    for col, mapping in binary_maps.items():
        df[col] = df[col].map(mapping)

    # Property_Area is not ordinal -> one-hot encode
    df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

    return df


if __name__ == "__main__":
    data = load_and_clean_data()
    print(data.head())
    print("\nShape:", data.shape)
    print("\nMissing values:\n", data.isnull().sum().sum(), "total")
