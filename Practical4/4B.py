import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

data = pd.read_csv(r"C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\football.csv");
data = data.values

x = data[:, :-1]
y = data[:, -1]

ordinal_encoder = OrdinalEncoder()
label_encoder = LabelEncoder()

x = ordinal_encoder.fit_transform(x)
y = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y)
model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy Score:\n {}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix:\n {}".format(confusion_matrix(y_test, y_pred)))
print("Recall Score:\n {}".format(recall_score(y_test, y_pred)))
print("Precision Score:\n {}".format(precision_score(y_test, y_pred)))