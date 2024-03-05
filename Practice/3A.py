import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

data = pd.read_csv(r'C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\football.csv')
x = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

ordinal_encoder = OrdinalEncoder()
label_encoder = LabelEncoder()
x = ordinal_encoder.fit_transform(x)
y = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_test, y_pred)
print("Confusion matrix : {}".format(confusion_matrix(y_test, y_pred)))
print("Accuracy Score   : {}".format(accuracy_score(y_test, y_pred)))
print(export_text(model))