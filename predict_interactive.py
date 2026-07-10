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
    """Ask the user to pick from a fixed set of choices (case-insensitive)."""
    choices_display = "/".join(choices)
    while True:
        answer = input(f"{prompt} ({choices_display}): ").strip()
        for c in choices:
            if answer.lower() == c.lower():
                return c
        print(f"  Please enter one of: {choices_display}")


def ask_number(prompt, allow_zero=True):
    """Ask the user for a numeric value, re-prompting on invalid input."""
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
            return value
        except ValueError:
            print("  Please enter a valid number.")


def ask_dependents():
    while True:
        answer = input("Number of dependents (0/1/2/3+): ").strip()
        if answer in ("0", "1", "2"):
            return int(answer)
        if answer in ("3", "3+"):
            return 3
        print("  Please enter 0, 1, 2, or 3+")


def collect_applicant_details():
    print("=" * 55)
    print("LOAN APPROVAL PREDICTOR - Enter Applicant Details")
    print("=" * 55)

    applicant = {}
    applicant["Gender"] = ask_choice("Gender", ["Male", "Female"])
    applicant["Married"] = ask_choice("Married", ["Yes", "No"])
    applicant["Dependents"] = ask_dependents()
    applicant["Education"] = ask_choice("Education", ["Graduate", "Not Graduate"])
    applicant["Self_Employed"] = ask_choice("Self-Employed", ["Yes", "No"])
    applicant["ApplicantIncome"] = ask_number("Applicant's monthly income")
    applicant["CoapplicantIncome"] = ask_number("Co-applicant's monthly income (0 if none)")
    applicant["LoanAmount"] = ask_number("Loan amount requested (in thousands, e.g. 150 for 1,50,000)", allow_zero=False)
    applicant["Loan_Amount_Term"] = ask_number("Loan term in days (e.g. 360 for 30 years)", allow_zero=False)
    applicant["Credit_History"] = int(ask_choice("Credit history - good repayment record", ["1", "0"]))
    applicant["Property_Area"] = ask_choice("Property area", ["Urban", "Semiurban", "Rural"])

    return applicant


def predict_loan_status(applicant: dict):
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

    if status == "REJECTED" and applicant["Credit_History"] == 0:
        print("Note: Lack of credit history is the biggest factor lowering approval odds.")

    again = input("\nPredict for another applicant? (y/n): ").strip().lower()
    if again == "y":
        print()
        main()


if __name__ == "__main__":
    main()
