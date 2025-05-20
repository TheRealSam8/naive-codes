from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
skf = StratifiedKFold(n_splits=5)
accuracies = []

iris = datasets.load_iris()
X, y = iris.data, iris.target

for train_idx, test_idx in skf.split(X, y):
    model = SVC()
    model.fit(X[train_idx], y[train_idx])
    acc = model.score(X[test_idx], y[test_idx])
    accuracies.append(acc)

print("Average Accuracy:", np.mean(accuracies))
