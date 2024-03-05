import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\income.csv")

x = data[["age", "experience"]]
y = data["income"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

model = LinearRegression()
model.fit(x_train, y_train)

print("Score: {}".format(model.score(x_train, y_train)))
print("Intercept: {}".format(model.intercept_))
print("Coefficient: {}".format(model.coef_))

y_pred = model.predict(x_test)

difference = pd.DataFrame({
    "actual_value": y_test,
    "predicted_value": y_pred,
    "difference": y_test - y_pred
})
print(difference)

print("Mean Squared Error: {}".format(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_pred)))
print("Root Mean Squared Error: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

figure = plt.figure()
subplot = figure.add_subplot(projection='3d')
subplot.scatter(x[['age']], x[['experience']], y, label='y', s=5)
subplot.view_init(45, 0)
plt.show()