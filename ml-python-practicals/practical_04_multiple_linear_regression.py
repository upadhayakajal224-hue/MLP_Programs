import pandas as pd
from sklearn.linear_model import LinearRegression

data = {'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9],
        'Attendance': [60, 65, 70, 75, 80, 85, 90, 95],
        'Marks': [35, 42, 48, 55, 63, 70, 78, 85]}
df = pd.DataFrame(data)
print(df)

X = df[['Study_Hours', 'Attendance']]
Y = df['Marks']

model = LinearRegression()
model.fit(X, Y)
predicted_marks = model.predict(X)

print("Coefficients:")
print("Study Hours =", model.coef_[0])
print("Attendance =", model.coef_[1])
print("Intercept =", model.intercept_)
print("Predicted Marks:", predicted_marks.round(2))

new_data = pd.DataFrame([[10, 96]], columns=['Study_Hours', 'Attendance'])
print("Predicted Marks for Study Hours = 10 and Attendance = 96:", model.predict(new_data)[0])
