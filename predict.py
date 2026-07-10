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
    """
    applicant keys required (raw, human-readable values):
    Gender: 'Male'/'Female'
    Married: 'Yes'/'No'
    Dependents: 0, 1, 2, or 3 (use 3 for '3+')
    Education: 'Graduate'/'Not Graduate'
    Self_Employed: 'Yes'/'No'
    ApplicantIncome: number
    CoapplicantIncome: number
    LoanAmount: number (in thousands)
    Loan_Amount_Term: number (in days, e.g. 360)
    Credit_History: 1 or 0
    Property_Area: 'Urban'/'Semiurban'/'Rural'
    """
    row = {
        "Gender": 1 if applicant["Gender"] == "Male" else 0,
        "Married": 1 if applicant["Married"] == "Yes" else 0,
        "Dependents": applicant["Dependents"],
        "Education": 1 if applicant["Education"] == "Graduate" else 0,
        "Self_Employed": 1 if applicant["Self_Employed"] == "Yes" else 0,
        "ApplicantIncome": applicant["ApplicantIncome"],
        "CoapplicantIncome": applicant["CoapplicantIncome"],
        "LoanAmount": applicant["LoanAmount"],
        "Loan_Amount_Term": applicant["Loan_Amount_Term"],
        "Credit_History": applicant["Credit_History"],
        "Property_Area_Semiurban": 1 if applicant["Property_Area"] == "Semiurban" else 0,
        "Property_Area_Urban": 1 if applicant["Property_Area"] == "Urban" else 0,
    }

    total_income = row["ApplicantIncome"] + row["CoapplicantIncome"]
    row["TotalIncome"] = total_income
    row["LoanAmount_log"] = np.log1p(row["LoanAmount"])
    row["TotalIncome_log"] = np.log1p(total_income)
    row["Loan_Income_Ratio"] = row["LoanAmount"] / (total_income + 1)

    df_row = pd.DataFrame([row])[feature_columns]  # ensure correct column order
    scaled = scaler.transform(df_row)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]  # probability of approval

    status = "APPROVED" if prediction == 1 else "REJECTED"
    return status, probability


if __name__ == "__main__":
    # Example applicant - edit these values to test different scenarios
    new_applicant = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": 1,
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    }

    status, prob = predict_loan_status(new_applicant)
    print(f"Prediction: {status}")
    print(f"Approval probability: {prob:.2%}")

    print("\n--- Testing a riskier profile (no credit history) ---")
    risky_applicant = new_applicant.copy()
    risky_applicant["Credit_History"] = 0
    status2, prob2 = predict_loan_status(risky_applicant)
    print(f"Prediction: {status2}")
    print(f"Approval probability: {prob2:.2%}")
