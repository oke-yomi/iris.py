import mglearn
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt

iris_dataSet = load_iris()
print("Keys of iris_dataSet: \n{}".format(iris_dataSet.keys()))
print(iris_dataSet['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataSet['target_names']))
print("Feature names: \n{}".format(iris_dataSet['feature_names']))
print("Type of data: {}".format(type(iris_dataSet['data'])))
print("Shape of data:  {}".format(iris_dataSet['data'].shape))
print("First five columns of data:\n{}".format(iris_dataSet['data'][:5]))
print("Type of target: {}".format(type(iris_dataSet['target'])))
print("Shape of target: {}".format(iris_dataSet['target'].shape))
print("Target:\n{}".format(iris_dataSet['target']))

X_train, X_test, y_train, y_test = train_test_split(
	iris_dataSet['data'], iris_dataSet['target'], random_state=0
)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

iris_dataFrame = pd.DataFrame(X_train, columns=iris_dataSet.feature_names)
grr = scatter_matrix(iris_dataFrame, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8,
cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Prediction target name: {}".format(
	iris_dataSet['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

X_train, X_test, y_train, y_test = train_test_split(
	iris_dataSet['data'], iris_dataSet['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))