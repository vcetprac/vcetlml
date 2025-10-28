import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Step 1: Load the dataset
data = pd.read_csv('customers_custom.csv')

# Step 2: Data Preprocessing
data['Region'] = data['Region'].map({1: 'Lisbon', 2: 'Oporto', 3: 'Other'})
data['Channel'] = data['Channel'].map({1: 'Hotel', 2: 'Retailer'})

# Step 3: Feature Selection
X = data.drop(columns=['Channel', 'Region'])

# Step 4: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Elbow Method to determine optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 6: Applying K-Means with optimal clusters (e.g. 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 7: Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Step 8: Visualize cluster centers
centers = kmeans.cluster_centers_
centers_pca = pca.transform(centers)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=300, c='red', marker='X', label='Centers')
plt.title('Clusters and Centers')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Step 9: Cluster Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', data=data, palette='viridis')
plt.title('Number of Customers per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# Step 10: Pairplot of original features by cluster
sns.pairplot(data.drop(columns=['Channel', 'Region']), hue='Cluster', palette='viridis')
plt.show()

# Optional: Save clustered data
data.to_csv('clustered_customers.csv', index=False)
