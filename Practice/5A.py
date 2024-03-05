import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(solver='liblinear')
model.fit(x, y)

y_pred = model.predict(x)
y_prob = model.predict_proba(x)
cm = confusion_matrix(y, y_pred)

print("Confusion Matrix      : \n{}".format(cm))
print("Classification Report : \n{}".format(classification_report(y, y_pred)))

figure, ax = plt.subplots(figsize=(8,8))
ax.imshow(cm)
ax.yaxis.set(ticks=(0,1), ticklabels=("Actual 0's", "Actual 1's"))
ax.xaxis.set(ticks=(0,1), ticklabels=("Predicted 0's", "Predicted 1's"))
for i in range(0,2):
    for j in range(0,2):
        ax.text(i, j, cm[i,j], va='center', ha='center', color='black')
plt.show()