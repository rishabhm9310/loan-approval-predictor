"""
app.py
------
Flask backend for the Loan Approval Predictor web app.
Serves the frontend and exposes a /api/predict endpoint that loads the
trained model and returns a prediction for a submitted applicant.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


def predict_loan_status(applicant: dict):
    row = {
        "no_of_dependents": float(applicant["dependents"]),
        "education": 1 if applicant["education"] == "Graduate" else 0,
        "self_employed": 1 if applicant["selfEmployed"] == "Yes" else 0,
        "income_annum": float(applicant["income"]),
        "loan_amount": float(applicant["loanAmount"]),
        "loan_term": float(applicant["loanTerm"]),
        "cibil_score": float(applicant["cibilScore"]),
        "residential_assets_value": float(applicant["residentialAssets"]),
        "commercial_assets_value": float(applicant["commercialAssets"]),
        "luxury_assets_value": float(applicant["luxuryAssets"]),
        "bank_asset_value": float(applicant["bankAssets"]),
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
    probability = float(model.predict_proba(scaled)[0][1])

    return {
        "status": "APPROVED" if prediction == 1 else "REJECTED",
        "probability": round(probability * 100, 1),
        "cibilScore": row["cibil_score"],
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        applicant = request.get_json()

        required_fields = [
            "dependents", "education", "selfEmployed", "income", "loanAmount",
            "loanTerm", "cibilScore", "residentialAssets", "commercialAssets",
            "luxuryAssets", "bankAssets"
        ]
        missing = [f for f in required_fields if f not in applicant or applicant[f] == ""]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        cibil = float(applicant["cibilScore"])
        if cibil < 300 or cibil > 900:
            return jsonify({"error": "CIBIL score must be between 300 and 900."}), 400

        result = predict_loan_status(applicant)
        return jsonify(result)

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception:
        return jsonify({"error": "Something went wrong processing the prediction."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
