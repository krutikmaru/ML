import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r'C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\salary.csv')
x_categorical = data[["Rank"]].values
x_numerical = data[["Levels"]].values
y = data[["Salary"]].values

label_encoder = LabelEncoder()
x_categorical = label_encoder.fit_transform(x_categorical)
x = np.column_stack((x_categorical, x_numerical))
x_train, x_test, y_train, y_test = train_test_split(x, y)

model = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("OOB score: {}".format(model.oob_score_))
print("MSE      : {}".format(mean_squared_error(y_test, y_pred)))
print("R2       : {}".format(r2_score(y_test, y_pred)))