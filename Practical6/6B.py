import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv(r"C:\Users\Krutik\Documents\IT College\Semester6\ML\Practicals\CSV\country.csv")
x = list(data["Latitude"])
y = list(data["Longitude"])

plt.scatter(x, y)
plt.show()

data = list(zip(x, y))
wcss = []

for i in range(1, 5):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 5), wcss, marker='o')
plt.show()
# From the Elbow Chart, we determine number of clusters as '3' which was already so obvious lol

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
plt.scatter(x,y, c=kmeans.labels_)
plt.show()