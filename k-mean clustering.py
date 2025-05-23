import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate synthetic data (e.g., 2D data with 3 clusters)
np.random.seed(42)

# Cluster 1
data1 = np.random.normal(loc=0, scale=1, size=(300, 2))  # mean=0, std=1
# Cluster 2
data2 = np.random.normal(loc=5, scale=1, size=(300, 2))  # mean=5, std=1
# Cluster 3
data3 = np.random.normal(loc=10, scale=1, size=(300, 2))  # mean=10, std=1

# Concatenate the data into one dataset
data = np.vstack([data1, data2, data3])

# Fit K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

# Get the predicted cluster labels
labels = kmeans.labels_

# Plot the data points and the cluster centroids
plt.figure(figsize=(8, 6))

# Plot the data points, colored by cluster
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)

# Plot the centroids (cluster centers)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200, label='Centroids')

# Add title and labels
plt.title('K-Means Clustering with 3 Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

