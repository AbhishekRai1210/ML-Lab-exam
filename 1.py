/*Develop a program to create histograms for all numerical features and analyze the distribution of each
feature. Generate box plots for all numerical features and identify any outliers. Use California Housing
dataset. */


import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import fetch_california_housing as fch

df = pd.DataFrame(fch().data, columns=fch().feature_names); df['MedHouseVal'] = fch().target

df.hist(figsize=(10, 6)); plt.tight_layout(); plt.show()
sns.boxplot(data=df, orient='h'); plt.tight_layout(); plt.show()

Q1, Q3 = df.quantile(.25), df.quantile(.75); IQR = Q3 - Q1
print("Outlier counts:\n", ((df < Q1 - 1.5*IQR) | (df > Q3 + 1.5*IQR)).sum())
