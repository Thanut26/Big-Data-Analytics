# --- Phase 2: Data Collection ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df  = pd.read_csv('Lab1/employee_data.csv')
print(f"Shape: {df.shape}")
print(df.isnull().sum())
print(df.duplicated().sum())

# --- Phase 3: Data Preparation & Storage ---
df["HireDate"] = pd.to_datetime(df["HireDate"])
df["TenureYears"] = (pd.Timestamp("2024-01-01") - df["HireDate"]).dt.days / 365.25

Q1, Q3 = df["Salary"].quantile([0.25, 0.75])
IQR = Q3 - Q1
df["Salary"] = df["Salary"].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

edu_order = {"High School": 1, "Associate": 2, "Bachelor": 3, "Master": 4, "PhD": 5}
df["EducationLevel"] = df["Education"].map(edu_order)

print("\n--- Phase 3: New Columns Sample ---")
print(df[["HireDate", "TenureYears", "Education", "EducationLevel"]].head())

# --- Phase 4: Data Processing & Analysis ---
dept_salary = df.groupby("Department")["Salary"].mean().sort_values(ascending=False)
edu_perf = df.groupby("Education")["PerformanceScore"].mean()
corr = df[["Age", "Salary", "PerformanceScore", "YearsExperience", "TrainingHours"]].corr()
status_dist = df["Status"].value_counts()

print("\n--- Phase 4: Average Salary by Department ---")
print(dept_salary)
print("\n--- Phase 4: Correlation Matrix ---")
print(corr)

# --- Phase 5: Visualisation ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Lab 1 - Employee Data Analytics Dashboard", fontsize=16, fontweight="bold")

# 1. Average Salary by Department
axes[0, 0].barh(dept_salary.index, dept_salary.values, color="steelblue")
axes[0, 0].set_title("Average Salary by Department")
axes[0, 0].set_xlabel("Salary (USD)")

# 2. Distribution of Performance Scores
axes[0, 1].hist(df["PerformanceScore"], bins=15, color="coral", edgecolor="black")
axes[0, 1].set_title("Distribution of Performance Scores")
axes[0, 1].set_xlabel("Performance Score")
axes[0, 1].set_ylabel("Frequency")

# 3. Salary vs Years of Experience
axes[1, 0].scatter(df["YearsExperience"], df["Salary"], alpha=0.5, color="seagreen")
axes[1, 0].set_title("Salary vs Years of Experience")
axes[1, 0].set_xlabel("Years of Experience")
axes[1, 0].set_ylabel("Salary (USD)")

# 4. Employee Status Distribution
axes[1, 1].pie(status_dist.values, labels=status_dist.index, autopct="%1.1f%%", startangle=90)
axes[1, 1].set_title("Employee Status Distribution")

plt.tight_layout()
plt.savefig("Lab1/lab1_dashboard.png", dpi=150)
print("\n--- Phase 5: Saved lab1_dashboard.png in Lab1 folder! ---")
plt.show()