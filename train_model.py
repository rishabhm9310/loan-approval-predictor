"""
train_model.py
---------------
Trains and compares multiple classification models on the loan dataset,
picks the best one, and saves it to disk along with the scaler.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from preprocess import load_and_clean_data


def main():
    df = load_and_clean_data()

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
    }

    results = []
    trained_models = {}

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "CV Mean Accuracy": round(cv_scores.mean(), 4),
        })
        trained_models[name] = model

        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by F1-Score)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    joblib.dump(best_model, "loan_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")
    results_df.to_csv("model_comparison_results.csv", index=False)

    print("\nSaved: loan_model.pkl, scaler.pkl, feature_columns.pkl, model_comparison_results.csv")

    if hasattr(best_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        print("\nTop feature importances:")
        print(importance_df.to_string(index=False))
        importance_df.to_csv("feature_importance.csv", index=False)


if __name__ == "__main__":
    main()
