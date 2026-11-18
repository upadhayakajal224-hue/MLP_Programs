import numpy as np
import pandas as pd

# NumPy array and properties
arr = np.array([10, 20, 30, 40, 50])
print(arr)
print("Shape:", arr.shape, "| Dimensions:", arr.ndim, "| Size:", arr.size)
print("Sum =", np.sum(arr), "| Mean =", np.mean(arr), "| Max =", np.max(arr),
      "| Min =", np.min(arr), "| Std =", np.std(arr))

# 2D array
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Transpose:\n", matrix.T)
print("Sum:", matrix.sum(), "| Mean:", matrix.mean())

# Pandas Series and DataFrame
marks = pd.Series([85, 90, 78, 92, 88])
print(marks)

data = {"Name": ["Rahul", "Priya", "Amit", "Sneha", "Karan"],
        "Age": [20, 21, 20, 22, 21],
        "Marks": [85, 92, 78, 95, 88],
        "City": ["Pune", "Mumbai", "Nashik", "Nagpur", "Aurangabad"]}
df = pd.DataFrame(data)
print(df)

# Exploration
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.describe())
print("Missing values:\n", df.isnull().sum())
print("Unique values:\n", df.nunique())

# Selection, filtering, sorting
print(df[["Name", "Marks"]])
print(df[df["Marks"] > 80])
print(df[(df["Marks"] > 80) & (df["Age"] == 21)])
print(df.sort_values(by="Marks", ascending=False))

# Add / update columns
df["Result"] = "Pass"
df["Grade"] = ["B", "A", "C", "A+", "A"]
df.loc[df["Marks"] < 80, "Result"] = "Needs Improvement"
print(df)

print("Average Marks =", df["Marks"].mean())
print("Max =", df["Marks"].max(), "| Min =", df["Marks"].min())
print(df["City"].value_counts())

# Save and reload CSV
df.to_csv("student_dataset.csv", index=False)
print(pd.read_csv("student_dataset.csv"))
