import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


x = np.arange(10).reshape(-1, 1)
y = np.array([0,0,0,0,1,1,1,1,1,1])

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x, y)
print(model.classes_)
print(model.intercept_)
print(model.coef_)


y_pred = model.predict(x)
y_prob = model.predict_proba(x)
print("Predicted Y:\n", y_pred)
print("Predicted Probability Y:\n", y_prob)

cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)

print(classification_report(y, y_pred))

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(cm)
ax.xaxis.set(ticks=(0,1), ticklabels=("Predicted 0s", "Predicted 1s"))
ax.yaxis.set(ticks=(0,1), ticklabels=("Actual 0s", "Actual 1s"))
for i in range(2):
    for j in range(2):
        ax.text(i, j, cm[i,j], ha='center', va='center', color='black')
plt.show()

