import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

data = {'Study_Hours': [1,2,3,4,5,6,7,8],
        'Attendance': [55,60,65,70,75,80,85,90],
        'Result': ['Fail','Fail','Fail','Pass','Pass','Pass','Pass','Pass']}
df = pd.DataFrame(data)
print(df)

X = df[['Study_Hours', 'Attendance']]
Y = df['Result']

model = DecisionTreeClassifier(random_state=42)
model.fit(X, Y)
predicted = model.predict(X)

print("Predicted Result:", predicted)
print("Confusion Matrix:")
print(confusion_matrix(Y, predicted))
print("Classification Report:")
print(classification_report(Y, predicted))
