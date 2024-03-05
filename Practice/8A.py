import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

data = pd.read_csv(r"C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\purchase.csv")
x = data[["age", "estimated_salary"]].values
y = data[["purchase"]].values

x_train, x_test, y_train, y_test = train_test_split(x, y)
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.fit_transform(x_test)

model = SVC(kernel='linear')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy_score = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy Score   : {}".format(accuracy_score))
print("Confusion Matrix : \n{}".format(confusion_matrix))

X, Y = np.meshgrid(
    np.arange(start = x_train[:, 0].min() - 1, stop = x_train[:, 0].max() + 1, step=0.01),
    np.arange(start = x_train[:, 1].min() - 1, stop = x_train[:, 1].max() + 1, step=0.01)
)
plt.contourf(X, Y, model.predict(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape), alpha=0.75, cmap=ListedColormap(("green", "blue")))
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(x_train[y_train.ravel() == j, 0], x_train[y_train.ravel() == j, 1], c=ListedColormap(("green", "blue"))(i), label=j)
plt.title("SVM")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()