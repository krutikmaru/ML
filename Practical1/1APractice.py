import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([10, 12, 15, 18, 25])

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

plt.plot(x, y, ".")
plt.plot(x, y_pred, color="green")
plt.show()