import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv(r"C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\football.csv")
data = data.values

x = data[:, :-1]
y = data[:, -1]

oridnal_encoder = OrdinalEncoder()
label_encoder = LabelEncoder()
x = oridnal_encoder.fit_transform(x)
y = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train = oridnal_encoder.fit_transform(x_train)
y_train = label_encoder.fit_transform(y_train)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy Score: {}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix: {}".format(confusion_matrix(y_test, y_pred)))
print("Text Representation: \n{}".format(export_text(model)))