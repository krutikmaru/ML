import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv(r"C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\petrol_consumption.csv")

x = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

oob = model.oob_score_
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Out-of-bag score: {}".format(oob))
print("RMSE: {}".format(rmse))
print("R2: {}".format(r2))