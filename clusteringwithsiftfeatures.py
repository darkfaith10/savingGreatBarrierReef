
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data1 = pd.read_csv('/Users/vijaykumarsingh/Desktop/onemore/SIFT/SIFT_cot.csv', dtype='uint8')
data2 = pd.read_csv('/Users/vijaykumarsingh/Desktop/onemore/SIFT/SIFT_not_cot.csv', dtype='uint8')


data = pd.concat([data1, data2], axis=0)


scaler = StandardScaler()
data_std = scaler.fit_transform(data)


k = 2

kmeans = KMeans(n_clusters=k, random_state=0)

kmeans.fit(data_std)


labels = kmeans.labels_
centers = kmeans.cluster_centers_


data['cluster'] = labels


data.to_csv('/Users/vijaykumarsingh/Desktop/onemore/SIFT/SIFT_clusters.csv', index=False)


data = pd.read_csv('/Users/vijaykumarsingh/Desktop/onemore/SIFT/SIFT_clusters.csv')


X = data.iloc[:,:-1].values
labels = data['cluster'].values


scaler = StandardScaler()
X_std = scaler.fit_transform(X)


k = 2


kmeans = KMeans(n_clusters=k, random_state=0)


kmeans.fit(X_std)


centers = kmeans.cluster_centers_


plt.scatter(X_std[labels==0, 0], X_std[labels==0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X_std[labels==1, 0], X_std[labels==1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='*', c='black', label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()