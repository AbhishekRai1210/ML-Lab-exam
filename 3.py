/*Develop a program to implement Principal Component Analysis (PCA) for reducing the dimensionality of
the Iris dataset from 4 features to 2. */
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2']); df['Target'] = y

sns.scatterplot(data=df, x='PC1', y='PC2', hue='Target', palette='Set1')
plt.title("PCA - Iris Dataset"); plt.show()
