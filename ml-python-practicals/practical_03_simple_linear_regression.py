import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {'Study_Hours': [1,2,3,4,5,6,7,8],
        'Marks': [30,35,45,50,60,65,75,80]}
df = pd.DataFrame(data)
X = df[['Study_Hours']]
Y = df['Marks']

model = LinearRegression()
model.fit(X, Y)
predicted_marks = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted Marks:", predicted_marks.round(2))
print("Predicted Marks for 9 Study Hours:", model.predict(pd.DataFrame({'Study_Hours': [9]}))[0])

plt.figure(figsize=(5, 3.2))
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, predicted_marks, color='red', label='Regression Line')
plt.xlabel("Study Hours"); plt.ylabel("Marks")
plt.title("Simple Linear Regression"); plt.legend()
plt.tight_layout()
import os; os.makedirs("images", exist_ok=True)
plt.savefig("images/practical3_plot.png", dpi=110)
plt.show()
