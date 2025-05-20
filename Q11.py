from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), params, cv=5)
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
