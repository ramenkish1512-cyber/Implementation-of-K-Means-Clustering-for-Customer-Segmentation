# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Choose the number of clusters (K).

2. Randomly initialize K centroids.

3. Assign each data point to the nearest centroid.

4. Recalculate the centroids.

5. Repeat steps 3 and 4 until centroids do not change
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Ramen kishore . N
RegisterNumber: 212225240116 
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mail_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
print(data.head())
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)


data['Cluster'] = y_kmeans

print("\nClustered Data:")
print(data.head())


plt.figure()
plt.scatter(X[y_kmeans == 0]['Annual Income (k$)'], 
            X[y_kmeans == 0]['Spending Score (1-100)'], label='Cluster 0')

plt.scatter(X[y_kmeans == 1]['Annual Income (k$)'], 
            X[y_kmeans == 1]['Spending Score (1-100)'], label='Cluster 1')

plt.scatter(X[y_kmeans == 2]['Annual Income (k$)'], 
            X[y_kmeans == 2]['Spending Score (1-100)'], label='Cluster 2')

plt.scatter(X[y_kmeans == 3]['Annual Income (k$)'], 
            X[y_kmeans == 3]['Spending Score (1-100)'], label='Cluster 3')

plt.scatter(X[y_kmeans == 4]['Annual Income (k$)'], 
            X[y_kmeans == 4]['Spending Score (1-100)'], label='Cluster 4')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            s=200, label='Centroids')

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
```

## Output:
<img width="1333" height="583" alt="555848392-acda5ae8-82f6-48a0-b8ba-3aab152f61b1" src="https://github.com/user-attachments/assets/03ef9b50-77bd-4396-b241-17f6fbb32b90" />
<img width="901" height="335" alt="555848275-60f6eecc-02a2-4180-b053-a86ec42b4c6b" src="https://github.com/user-attachments/assets/e058e482-86c0-4b48-9721-288eb7a0623c" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
