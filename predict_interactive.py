"""
predict_interactive.py
-----------------------
Interactive version of predict.py - asks the user questions in the terminal,
then predicts whether the loan would be approved or rejected.

Run with: python predict_interactive.py
"""

import joblib
import numpy as np
import pandas as pd

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


def ask_choice(prompt, choices):
    choices_display = "/".join(choices)
    while True:
        answer = input(f"{prompt} ({choices_display}): ").strip()
        for c in choices:
            if answer.lower() == c.lower():
                return c
        print(f"  Please enter one of: {choices_display}")


def ask_number(prompt, allow_zero=True, min_val=None, max_val=None):
    while True:
        answer = input(f"{prompt}: ").strip()
        try:
            value = float(answer)
            if value < 0:
                print("  Please enter a non-negative number.")
                continue
            if not allow_zero and value == 0:
                print("  Please enter a number greater than 0.")
                continue
            if min_val is not None and value < min_val:
                print(f"  Please enter a number >= {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"  Please enter a number <= {max_val}.")
                continue
            return value
        except ValueError:
            print("  Please enter a valid number.")


def collect_applicant_details():
    print("=" * 55)
    print("LOAN APPROVAL PREDICTOR - Enter Applicant Details")
    print("=" * 55)

    applicant = {}
    applicant["no_of_dependents"] = int(ask_number("Number of dependents"))
    applicant["education"] = ask_choice("Education", ["Graduate", "Not Graduate"])
    applicant["self_employed"] = ask_choice("Self-Employed", ["Yes", "No"])
    applicant["income_annum"] = ask_number("Annual income (Rs.)", allow_zero=False)
    applicant["loan_amount"] = ask_number("Loan amount requested (Rs.)", allow_zero=False)
    applicant["loan_term"] = ask_number("Loan term (years, e.g. 10)", allow_zero=False)
    applicant["cibil_score"] = ask_number("CIBIL score", min_val=300, max_val=900)
    applicant["residential_assets_value"] = ask_number("Residential assets value (Rs., 0 if none)")
    applicant["commercial_assets_value"] = ask_number("Commercial assets value (Rs., 0 if none)")
    applicant["luxury_assets_value"] = ask_number("Luxury assets value (Rs., 0 if none)")
    applicant["bank_asset_value"] = ask_number("Bank asset value (Rs., 0 if none)")

    return applicant


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


def main():
    applicant = collect_applicant_details()
    status, prob = predict_loan_status(applicant)

    print("\n" + "=" * 55)
    print("PREDICTION RESULT")
    print("=" * 55)
    print(f"Loan Status:            {status}")
    print(f"Approval Probability:   {prob:.1%}")
    print("=" * 55)

    if status == "REJECTED" and applicant["cibil_score"] < 550:
        print("Note: A low CIBIL score is the dominant factor lowering approval odds.")

    again = input("\nPredict for another applicant? (y/n): ").strip().lower()
    if again == "y":
        print()
        main()


if __name__ == "__main__":
    main()
