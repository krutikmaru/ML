import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load data
data = pd.read_csv(r"C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\purchase.csv")
x = data[["age", 'estimated_salary']].values
y = data[["purchase"]].values

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Train SVM model
model = SVC(kernel="linear", random_state=0)
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Model evaluation
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", confusion_matrix)
print("Accuracy Score:", accuracy_score)

# Visualize decision boundary and separation of classes
x1, x2 = np.meshgrid(np.arange(start=x_train[:, 0].min() - 1, stop=x_train[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_train[:, 1].min() - 1, stop=x_train[:, 1].max() + 1, step=0.01))

plt.contourf(x1, x2, model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha=0.75,
             cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# Plot data points
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(x_train[y_train.ravel() == j, 0], x_train[y_train.ravel() == j, 1], c=ListedColormap(('red', 'green'))(i),
                label=j)

plt.title("SVM Classifier (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
