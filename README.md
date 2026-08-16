# Loan Approval Predictor

A machine learning project that predicts whether a bank loan application will be
**approved** or **rejected**, based on applicant details like income, CIBIL
score, and asset holdings. Includes a trained model, CLI tools, and a full
web app (Flask backend + HTML/CSS/JS frontend).

## Problem Statement
Banks receive many loan applications and need a fast, consistent way to screen
them. This project builds a binary classification model that automates the
initial eligibility decision using historical applicant data, and wraps it in
a usable web interface.

## Dataset
**4,269 records**, 0 missing values. Columns:

| Column | Description |
|---|---|
| no_of_dependents | Number of people financially dependent on the applicant |
| education | Graduate / Not Graduate |
| self_employed | Yes / No |
| income_annum | Annual income (Rs.) |
| loan_amount | Loan amount requested (Rs.) |
| loan_term | Loan term in years |
| cibil_score | Applicant's CIBIL credit score (300-900) |
| residential/commercial/luxury/bank_asset_value | Value of assets held (Rs.) |
| loan_status | Target: Approved / Rejected |

## Project Structure
```
loan_approval_predictor/
├── data/
│   └── loan_data.csv           # raw dataset
├── preprocess.py                # cleaning + feature engineering
├── train_model.py               # trains & compares 3 models, saves the best
├── predict.py                   # edit-and-run script for a single prediction
├── predict_interactive.py       # terminal Q&A prediction tool
├── eda_visuals.py                # generates exploratory charts
├── eda_charts.png                # EDA output
├── app.py                        # Flask backend + API for the web app
├── templates/
│   └── index.html                # web app frontend markup
├── static/
│   ├── style.css                 # web app styling
│   └── script.js                 # web app frontend logic
├── requirements.txt
├── loan_model.pkl                # saved best model
├── scaler.pkl                    # saved StandardScaler
├── feature_columns.pkl           # saved column order
├── model_comparison_results.csv  # metrics for all 3 models
└── README.md
```

## Approach

1. **Data Cleaning** (`preprocess.py`) — strips whitespace from string
   columns/values (a quirk of the raw CSV), encodes `education`,
   `self_employed`, and `loan_status`.
2. **Feature Engineering** — `total_assets` (sum of all four asset types),
   `loan_to_assets_ratio`, `loan_to_income_ratio`, and log-transforms of the
   large monetary columns.
3. **Modeling** (`train_model.py`) — Logistic Regression, Decision Tree, and
   Random Forest compared with 5-fold cross-validation; best model chosen by
   F1-score.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Decision Tree** | 1.000 | 1.000 | 1.000 | **1.000** |
| Random Forest | 0.999 | 0.998 | 1.000 | 0.999 |
| Logistic Regression | 0.909 | 0.917 | 0.938 | 0.927 |

**Decision Tree** was selected. **Important note on the near-perfect score:**
this isn't overfitting — 5-fold cross-validation confirms it (99.88% mean
accuracy). It reflects a real characteristic of this dataset: **CIBIL score
alone accounts for ~81% of the model's decision weight**, with
`loan_to_income_ratio` and `loan_term` contributing most of the rest. This is
a known property of this popular synthetic dataset — the approval rule is
close to deterministic, which makes it very learnable but also less "noisy"
than real-world lending data. Worth mentioning explicitly in any writeup or
presentation, since a 100% accuracy claim invites scrutiny otherwise.

## How to Run — Command Line

```bash
pip install -r requirements.txt

python eda_visuals.py            # generates eda_charts.png
python train_model.py            # trains models, saves loan_model.pkl
python predict.py                # single prediction (edit values in the file)
python predict_interactive.py    # asks you questions in the terminal
```

## How to Run — Web App

```bash
pip install -r requirements.txt
python app.py
```
Then open **http://localhost:5000** in your browser. Fill in the application
form and submit — it calls the live trained model through a `/api/predict`
endpoint and displays an approved/rejected decision with a confidence score.

## Limitations & Future Work
- The near-deterministic CIBIL-driven approval rule makes this dataset easier
  to model than real-world credit data, which is noisier and involves many
  more soft factors.
- Could add hyperparameter tuning (GridSearchCV) for further gains, though
  there's little headroom left given the near-perfect scores.
- Web app currently runs locally; could be deployed (Render/Railway/PythonAnywhere)
  for a public link.
