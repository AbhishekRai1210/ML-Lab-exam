/*Develop a program to demonstrate the working of the decision tree algorithm. Use Breast Cancer Data set
for building the decision tree and apply this knowledge to classify a new sample. */

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classify a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]
print(f'Predicted class: {iris.target_names[clf.predict(new_sample)]}')
