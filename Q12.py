from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X, y = make_classification(n_classes=2, weights=[0.1, 0.9], n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = SVC(class_weight='balanced')
clf.fit(X_train, y_train)
print("Balanced Class Weight Accuracy:", clf.score(X_test, y_test))
