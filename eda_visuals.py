"""
eda_visuals.py
--------------
Generates exploratory data analysis charts and saves them as PNG files
for use in the project report / presentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("data/loan_data.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 1. Loan approval rate by credit history
approval_by_credit = df.groupby("Credit_History")["Loan_Status"].apply(
    lambda x: (x == "Y").mean() * 100
)
axes[0, 0].bar(["No Credit History (0)", "Has Credit History (1)"],
               approval_by_credit.values, color=["#e74c3c", "#27ae60"])
axes[0, 0].set_title("Approval Rate by Credit History")
axes[0, 0].set_ylabel("Approval Rate (%)")
for i, v in enumerate(approval_by_credit.values):
    axes[0, 0].text(i, v + 1, f"{v:.1f}%", ha="center")

# 2. Overall approval distribution
status_counts = df["Loan_Status"].value_counts()
axes[0, 1].pie(status_counts.values, labels=["Approved", "Rejected"],
               autopct="%1.1f%%", colors=["#27ae60", "#e74c3c"], startangle=90)
axes[0, 1].set_title("Overall Loan Approval Distribution")

# 3. Applicant income distribution by loan status
df.boxplot(column="ApplicantIncome", by="Loan_Status", ax=axes[1, 0])
axes[1, 0].set_title("Applicant Income vs Loan Status")
axes[1, 0].set_xlabel("Loan Status")
axes[1, 0].set_ylabel("Applicant Income")
axes[1, 0].set_ylim(0, 20000)  # clip extreme outliers for readability
plt.suptitle("")  # remove default pandas subtitle

# 4. Approval rate by property area
approval_by_area = df.groupby("Property_Area")["Loan_Status"].apply(
    lambda x: (x == "Y").mean() * 100
)
axes[1, 1].bar(approval_by_area.index, approval_by_area.values, color="#3498db")
axes[1, 1].set_title("Approval Rate by Property Area")
axes[1, 1].set_ylabel("Approval Rate (%)")
for i, v in enumerate(approval_by_area.values):
    axes[1, 1].text(i, v + 1, f"{v:.1f}%", ha="center")

plt.tight_layout()
plt.savefig("eda_charts.png", dpi=150, bbox_inches="tight")
print("Saved eda_charts.png")
