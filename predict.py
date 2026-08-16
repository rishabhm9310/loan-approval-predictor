"""
predict.py
----------
Loads the trained model and makes a prediction for a new loan applicant.
Edit the `new_applicant` dictionary below to test different profiles.
"""

import joblib
import numpy as np
import pandas as pd

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


def predict_loan_status(applicant: dict):
    row = {
        "no_of_dependents": applicant["no_of_dependents"],
        "education": 1 if applicant["education"] == "Graduate" else 0,
        "self_employed": 1 if applicant["self_employed"] == "Yes" else 0,
        "income_annum": applicant["income_annum"],
        "loan_amount": applicant["loan_amount"],
        "loan_term": applicant["loan_term"],
        "cibil_score": applicant["cibil_score"],
        "residential_assets_value": applicant["residential_assets_value"],
        "commercial_assets_value": applicant["commercial_assets_value"],
        "luxury_assets_value": applicant["luxury_assets_value"],
        "bank_asset_value": applicant["bank_asset_value"],
    }

    total_assets = (
        row["residential_assets_value"] + row["commercial_assets_value"]
        + row["luxury_assets_value"] + row["bank_asset_value"]
    )
    row["total_assets"] = total_assets
    row["loan_to_assets_ratio"] = row["loan_amount"] / (total_assets + 1)
    row["loan_to_income_ratio"] = row["loan_amount"] / row["income_annum"]
    row["income_annum_log"] = np.log1p(max(row["income_annum"], 0))
    row["loan_amount_log"] = np.log1p(max(row["loan_amount"], 0))
    row["total_assets_log"] = np.log1p(max(total_assets, 0))

    df_row = pd.DataFrame([row])[feature_columns]
    scaled = scaler.transform(df_row)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    status = "APPROVED" if prediction == 1 else "REJECTED"
    return status, probability


if __name__ == "__main__":
    new_applicant = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 5000000,
        "loan_amount": 10000000,
        "loan_term": 10,
        "cibil_score": 750,
        "residential_assets_value": 5000000,
        "commercial_assets_value": 3000000,
        "luxury_assets_value": 12000000,
        "bank_asset_value": 4000000,
    }

    status, prob = predict_loan_status(new_applicant)
    print(f"Prediction: {status}")
    print(f"Approval probability: {prob:.2%}")

    print("\n--- Testing a low CIBIL score profile ---")
    risky = new_applicant.copy()
    risky["cibil_score"] = 400
    status2, prob2 = predict_loan_status(risky)
    print(f"Prediction: {status2}")
    print(f"Approval probability: {prob2:.2%}")
