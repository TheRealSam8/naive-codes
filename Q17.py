from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

kernels = ['linear', 'poly', 'rbf']
for k in kernels:
    clf = SVC(kernel=k)
    clf.fit(X_train, y_train)
    print(f"{k} kernel Accuracy:", clf.score(X_test, y_test))
