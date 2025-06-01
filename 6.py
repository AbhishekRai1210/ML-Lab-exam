/*Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select
appropriate data set for your experiment and draw graphs*/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

# Generate synthetic data
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + 0.1 * np.random.randn(100)

# Apply Kernel Ridge Regression (Equivalent to Locally Weighted Regression)
model = KernelRidge(alpha=1, kernel='rbf', gamma=0.1)  # RBF kernel simulates LWR
model.fit(X, y)
y_pred = model.predict(X)

# Plot results
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='LWR Fit')
plt.title('Locally Weighted Regression (Using Kernel Ridge Regression)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
