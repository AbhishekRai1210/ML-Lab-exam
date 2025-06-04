/*Develop a program to create histograms for all numerical features and analyze the distribution of each
feature. Generate box plots for all numerical features and identify any outliers. Use California Housing
dataset. */


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Plot histograms for all numerical features
df.hist(figsize=(10, 6))
plt.tight_layout()
plt.show()

# Plot box plots for all numerical features
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, orient='h')
plt.tight_layout()
plt.show()

# Find outliers using the IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("Number of outliers in each column:")
print(outliers)
