import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = {'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
        'Attendance': [55, 60, 65, 70, 75, 80, 85, 90],
        'Result': ['Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass']}
df = pd.DataFrame(data)
print(df)

X = df[['Study_Hours', 'Attendance']]
Y = df['Result']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, Y)

new_data = pd.DataFrame([[5, 78]], columns=['Study_Hours', 'Attendance'])
print("Prediction for Study Hours = 5 and Attendance = 78:", model.predict(new_data)[0])
