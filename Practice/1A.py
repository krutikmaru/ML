import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1);
y = np.array([10, 12, 15, 18, 25])

model = LinearRegression()
model.fit(x, y)

print("Score       : {}".format(model.score(x, y)))
print("Intercept   : {}".format(model.intercept_))
print("Coefficient : {}".format(model.coef_))

y_pred = model.predict(x)

plt.plot(x, y, '.')
plt.plot(x, y_pred, color='red')
plt.show()