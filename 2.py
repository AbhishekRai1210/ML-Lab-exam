/*Develop a program to Compute the correlation matrix to understand the relationships between pairs of
features. Visualize the correlation matrix using a heatmap to know which variables have strong
positive/negative correlations. Create a pair plot to visualize pairwise relationships between features. Use
California Housing dataset. */
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing as fch

# Load dataset
df = pd.DataFrame(fch().data, columns=fch().feature_names)
df['MedHouseVal'] = fch().target

# Correlation matrix and heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap"); plt.show()

# Pairplot for selected features (to keep it readable)
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal']], diag_kind='kde')
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.show()
