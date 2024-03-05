import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4,5,10,4,3,11,14,6,10,12]
y = [21,19,24,17,16,25,24,22,21,21]

# plt.scatter(x, y)
# plt.show()

data = list(zip(x, y))
intertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)    
    kmeans.fit(data)
    intertias.append(kmeans.inertia_)

plt.plot(range(1,11), intertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.show()

# From the elbow chart we determine number of clusters as '2'
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(x, y, c=kmeans.labels_)
plt.show()