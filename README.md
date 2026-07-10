# Loan Approval Predictor
Intern id: CTTS143
A machine learning project that predicts whether a bank loan application will be
**approved** or **rejected**, based on applicant details like income, credit
history, education, and property location.

## Problem Statement
Banks receive many loan applications and need a fast, consistent way to screen
them. This project builds a binary classification model that automates the
initial eligibility decision using historical applicant data.

## Dataset
The standard **Loan Prediction** dataset (614 records, 12 features), commonly
used for this exact problem. Columns include:

| Column | Description |
|---|---|
| Gender, Married, Dependents | Applicant demographics |
| Education, Self_Employed | Applicant background |
| ApplicantIncome, CoapplicantIncome | Income details |
| LoanAmount, Loan_Amount_Term | Loan details |
| Credit_History | 1 = good repayment history, 0 = none/bad |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Target: Y (Approved) / N (Rejected) |

## Project Structure
```
loan_approval_predictor/
├── data/
│   └── loan_data.csv          # raw dataset
├── preprocess.py               # cleaning + feature engineering
├── train_model.py              # trains & compares 3 models, saves the best
├── predict.py                  # load saved model, predict on new applicant
├── eda_visuals.py               # generates exploratory charts
├── eda_charts.png               # output charts
├── loan_model.pkl               # saved best model
├── scaler.pkl                   # saved StandardScaler
├── feature_columns.pkl          # saved column order (needed for predict.py)
├── model_comparison_results.csv # metrics for all 3 models
└── README.md
```

## Approach

1. **Data Cleaning** (`preprocess.py`)
   - Missing categorical values filled with mode; numeric values filled with median.
   - `Dependents` "3+" converted to numeric 3.
   - Categorical variables encoded (binary map + one-hot for Property_Area).

2. **Feature Engineering**
   - `TotalIncome` = Applicant + Coapplicant income.
   - Log-transforms of income and loan amount (reduces skew from outliers).
   - `Loan_Income_Ratio` = loan amount relative to income (risk indicator).

3. **Modeling** (`train_model.py`)
   Three models trained and compared with 5-fold cross-validation:
   - Logistic Regression
   - Decision Tree
   - Random Forest

   Metrics used: Accuracy, Precision, Recall, F1-Score. F1-score was used to
   pick the winner since it balances the cost of wrongly rejecting a good
   applicant against wrongly approving a risky one.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | 0.870 | 0.849 | 0.988 | **0.913** |
| Random Forest | 0.870 | 0.856 | 0.977 | 0.912 |
| Decision Tree | 0.846 | 0.844 | 0.953 | 0.895 |

**Logistic Regression** was selected as the final model. Credit history turned
out to be by far the strongest predictor — applicants with a good credit
history are approved roughly **80%** of the time, versus about **8%** for
those without one (see `eda_charts.png`).

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# 2. Generate EDA charts
python eda_visuals.py

# 3. Train models (saves loan_model.pkl, scaler.pkl)
python train_model.py

# 4. Predict on a new applicant (edit the dict inside predict.py)
python predict.py
```

## Limitations & Future Work
- Dataset is small (614 rows) and somewhat imbalanced toward approvals (~69%).
- Could add hyperparameter tuning (GridSearchCV) for further gains.
- A simple Streamlit front-end could turn `predict.py` into an interactive
  web form for demo purposes.
