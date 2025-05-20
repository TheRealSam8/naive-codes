from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_unscaled = SVC()
model_unscaled.fit(X_train, y_train)
print("Unscaled Accuracy:", model_unscaled.score(X_test, y_test))

model_scaled = make_pipeline(StandardScaler(), SVC())
model_scaled.fit(X_train, y_train)
print("Scaled Accuracy:", model_scaled.score(X_test, y_test))
