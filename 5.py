/*Develop a program to implement k-Nearest Neighbour algorithm to classify the randomly generated 100 values of x in the range of [0,1]. Perform the following based on dataset generated.

a) Label the first 50 points {x1,……,x50} as follows: if (xi ≤ 0.5), then xi ∊ Class1, else xi ∊ Class1
b) Classify the remaining points, x51,……,x100 using KNN. Perform this for k=1,2,3,4,5,20,30*/

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Generate random data and labels
X = np.random.rand(0, 1)
y = np.array(['Class1' if xi <= 0.5 else 'Class2' for xi in X])

# Specify desired k values
k_values = [1, 2, 3, 4, 5, 10, 20]

# Loop over the specified k values and print confusion matrix
for k in k_values:
    y_pred = KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(X)
    print(f"Confusion Matrix for k={k}:\n{confusion_matrix(y, y_pred)}\n")
