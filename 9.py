/*Develop a program to implement the Naive Bayesian classifier considering Olivetti Face Data set for training.
Compute the accuracy of the classifier, considering a few test data sets. */

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset, split into train and test
X, y = fetch_olivetti_faces(shuffle=True, random_state=42).data, fetch_olivetti_faces().target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict
nb = GaussianNB().fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Accuracy and sample predictions
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print(f'Sample predictions: {list(zip(y_test[:5], y_pred[:5]))}')
