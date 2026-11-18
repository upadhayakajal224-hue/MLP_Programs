import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {'Age': [25, 30, 35, 40, 45, 50, 55, 60],
        'BP': [120, 130, 125, 140, 145, 150, 155, 160],
        'Sugar': [90, 95, 100, 110, 120, 130, 140, 150],
        'Disease': ['No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']}
df = pd.DataFrame(data)
print(df)

X = df[['Age', 'BP', 'Sugar']]
Y = df['Disease']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, Y)
predicted = model.predict(X)

print("Predicted Disease Status:", predicted)
print("Accuracy:", accuracy_score(Y, predicted))
print(classification_report(Y, predicted))

new_patient = pd.DataFrame([[42, 138, 115]], columns=['Age', 'BP', 'Sugar'])
print("Prediction for New Patient:", model.predict(new_patient)[0])
