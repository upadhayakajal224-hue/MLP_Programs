import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {'Study_Hours': [2,3,4,5,6,7,8,9,5,6],
        'Attendance': [60,65,70,75,80,85,90,95,72,88],
        'Assignments': [1,2,2,3,3,4,4,5,3,4],
        'Result': ['Fail','Fail','Fail','Pass','Pass','Pass','Pass','Pass','Fail','Pass']}
df = pd.DataFrame(data)
print(df)

X = df[['Study_Hours', 'Attendance', 'Assignments']]
Y = df['Result']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, Y_train)
predicted = pipeline.predict(X_test)

print("Actual Result:", list(Y_test))
print("Predicted Result:", list(predicted))
print("Accuracy:", accuracy_score(Y_test, predicted))
print(classification_report(Y_test, predicted))

new_student = pd.DataFrame([[6, 82, 3]], columns=['Study_Hours', 'Attendance', 'Assignments'])
print("Prediction for New Student:", pipeline.predict(new_student)[0])
