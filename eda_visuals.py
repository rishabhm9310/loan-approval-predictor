"""
eda_visuals.py
--------------
Generates exploratory data analysis charts for the loan approval dataset
and saves them as PNG files for use in the project report / presentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("data/loan_data.csv")
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].str.strip()

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 1. CIBIL score distribution by loan status - the dominant predictor
for status, color in [("Approved", "#27ae60"), ("Rejected", "#e74c3c")]:
    subset = df[df["loan_status"] == status]["cibil_score"]
    axes[0, 0].hist(subset, bins=30, alpha=0.6, label=status, color=color)
axes[0, 0].set_title("CIBIL Score Distribution by Loan Status")
axes[0, 0].set_xlabel("CIBIL Score")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend()

# 2. Overall approval distribution
status_counts = df["loan_status"].value_counts()
axes[0, 1].pie(status_counts.values, labels=status_counts.index,
               autopct="%1.1f%%", colors=["#27ae60", "#e74c3c"], startangle=90)
axes[0, 1].set_title("Overall Loan Approval Distribution")

# 3. Loan amount vs income, colored by status
colors = df["loan_status"].map({"Approved": "#27ae60", "Rejected": "#e74c3c"})
axes[1, 0].scatter(df["income_annum"] / 1e6, df["loan_amount"] / 1e6,
                    c=colors, alpha=0.3, s=10)
axes[1, 0].set_title("Loan Amount vs Annual Income")
axes[1, 0].set_xlabel("Annual Income (Rs. millions)")
axes[1, 0].set_ylabel("Loan Amount (Rs. millions)")

# 4. Approval rate by CIBIL score bracket
df["cibil_bracket"] = pd.cut(
    df["cibil_score"], bins=[300, 500, 600, 700, 800, 900],
    labels=["300-500", "500-600", "600-700", "700-800", "800-900"]
)
approval_by_bracket = df.groupby("cibil_bracket", observed=True)["loan_status"].apply(
    lambda x: (x == "Approved").mean() * 100
)
axes[1, 1].bar(approval_by_bracket.index.astype(str), approval_by_bracket.values, color="#3498db")
axes[1, 1].set_title("Approval Rate by CIBIL Score Bracket")
axes[1, 1].set_ylabel("Approval Rate (%)")
axes[1, 1].set_xlabel("CIBIL Score Range")
for i, v in enumerate(approval_by_bracket.values):
    axes[1, 1].text(i, v + 1, f"{v:.0f}%", ha="center")

plt.tight_layout()
plt.savefig("eda_charts.png", dpi=150, bbox_inches="tight")
print("Saved eda_charts.png")
