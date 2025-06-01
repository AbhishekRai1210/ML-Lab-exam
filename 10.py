/*Develop a program to implement k-means clustering using Wisconsin Breast Cancer data set and visualize
the clustering result. */

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Visualize clusters (using first two features)
sns.scatterplot(data=df, x=iris.feature_names[0], y=iris.feature_names[1], hue='Cluster', palette='Set1')
plt.title('K-Means Clustering on Iris Dataset')
plt.show()
