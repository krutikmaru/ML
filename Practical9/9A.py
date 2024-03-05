import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
x = iris.data[:, 0:4]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(y_test)
print(y_pred)

confusion_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix: \n{}".format(confusion_matrix))

plt.figure(figsize=(8,6))
sns.heatmap(
    confusion_matrix, annot=True, cmap='Blues', fmt='g',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()