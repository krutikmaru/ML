import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\salary.csv')

x_categorical = data[['Rank']].values 
x_numerical = data[['Levels']].values 
y = data[['Salary']].values # Target variable

label_encoder = LabelEncoder()
x_categorical = label_encoder.fit_transform(x_categorical)
x = np.column_stack((x_categorical, x_numerical))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

oob_score = model.oob_score_
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'Out-of-Bag Score: {oob_score}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')
