import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_text

data = pd.read_csv(r"C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\football.csv")
data = data.values
x = data[:, :-1]
y = data[:, -1]

oridinal = OrdinalEncoder()
x = oridinal.fit_transform(x)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y)
oridinal_encoder = OrdinalEncoder()
x_train = oridinal_encoder.fit_transform(x_train)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)


model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

text_representation = export_text(model)
print(text_representation)
