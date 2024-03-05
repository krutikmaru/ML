import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv(r'C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\comments.csv')
x = data[["comment"]].values
y = data[["category"]].values


label_encoder = LabelEncoder()
ordinal_encoder = OrdinalEncoder()
x = ordinal_encoder.fit_transform(x)
y = label_encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy Score   : {}".format(accuracy_score(y_test, y_pred)))
print("Recall Score     : {}".format(recall_score(y_test, y_pred)))
print("Precision Score  : {}".format(precision_score(y_test, y_pred)))
print("Confusion Matrix : \n{}".format(confusion_matrix(y_test, y_pred)))