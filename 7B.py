/*Develop a program to demonstrate the working of Linear Regression and Polynomial Regression. Use
Boston Housing Dataset for Linear Regression and Auto MPG Dataset (for vehicle fuel efficiency prediction)
for Polynomial Regression. 


POLYNOMIAL REGRESSION  */

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and clean the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
data = pd.read_csv(url, names=columns, na_values="?", delim_whitespace=True).fillna(method='ffill')

# Feature and target
X = data[['weight']]
y = data['mpg']

# Split, transform, train, predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

model = LinearRegression().fit(X_poly_train, y_train)
y_pred = model.predict(X_poly_test)

# Output and plot
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
plt.scatter(X_test, y_test, color='blue')
plt.scatter(X_test, y_pred, color='red')
plt.show()
