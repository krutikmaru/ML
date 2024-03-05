import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
y = np.array([10, 12, 15, 18, 25])

model = LinearRegression()
model.fit(x, y)

print("Score: {}".format(model.score(x, y)))
print("Intercept: {}".format(model.intercept_))
print("Coefficient: {}".format(model.coef_))

new_x = np.array([6]).reshape(-1, 1)
predicted_y = model.predict(new_x)
print("When x is 60, predicted y will be: {}".format(predicted_y))

predicted_y = model.predict(x)
plt.plot(x, y, ".")
plt.plot(x, predicted_y, color='red')
plt.show()