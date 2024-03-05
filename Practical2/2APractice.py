import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_dataset(n):
    x = []
    y = []
    random_x1 = np.random.rand()
    random_x2 = np.random.rand()
    for i in range(n):
        x1 = i
        x2 = i/2 + np.random.rand() * i
        x.append([x1, x2])
        y.append(x1 * random_x1 + x2 * random_x2 + 1)
    return np.array(x), np.array(y)

x, y = generate_dataset(200)
model = LinearRegression()
model.fit(x, y)


figure = plt.figure()
subplot = figure.add_subplot(projection='3d')
subplot.scatter(x[:, 0], x[:, 1], y, label='y', s=5)
subplot.view_init(45, 0)
plt.show()