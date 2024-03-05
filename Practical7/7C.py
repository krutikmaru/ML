from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

iris = datasets.load_iris()
x = iris.data[:, 0:4]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("R Squared Error    : {}".format(r2_score(y_test, y_pred)))
print("Accuracy Score     : {}".format(accuracy_score(y_test, y_pred)))
print("Mean Squared Error : {}".format(mean_squared_error(y_test, y_pred)))
