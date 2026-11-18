import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {"Name": ["John", "Alice", "Bob", "David", "Emma", "John"],
        "Age": [25, np.nan, 30, 22, 29, 25],
        "Salary": [50000, 60000, np.nan, 45000, 70000, 50000],
        "Department": ["HR", "IT", "Finance", "HR", np.nan, "HR"]}
df = pd.DataFrame(data)
print(df)

# Data cleaning
print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df["Department"] = df["Department"].fillna("Unknown")
df = df.drop_duplicates()
print(df)
print(df.dtypes)

# Visualization
fig, ax = plt.subplots(2, 2, figsize=(9, 6))
df.groupby("Department")["Salary"].mean().plot(kind="bar", color="skyblue", ax=ax[0, 0],
    title="Average Salary by Department", xlabel="Department", ylabel="Average Salary")
ax[0, 1].hist(df["Age"], bins=5, color="green", edgecolor="black")
ax[0, 1].set_title("Age Distribution")
dc = df["Department"].value_counts()
ax[1, 0].pie(dc, labels=dc.index, autopct="%1.1f%%", startangle=90)
ax[1, 0].set_title("Department Distribution")
ax[1, 1].scatter(df["Age"], df["Salary"], color="red")
ax[1, 1].set_title("Age vs Salary"); ax[1, 1].grid(True)
plt.tight_layout()
import os; os.makedirs("images", exist_ok=True)
plt.savefig("images/practical2_plots.png", dpi=110)
plt.show()
