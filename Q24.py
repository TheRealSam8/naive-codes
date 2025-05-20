from sklearn.metrics import mean_absolute_error
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svr = SVR()
svr.fit(X_train, y_train)
pred = svr.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, pred))
