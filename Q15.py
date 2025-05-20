from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_new = SelectKBest(chi2, k=5).fit_transform(X_train, y_train)
nb = GaussianNB()
nb.fit(X_new, y_train)
X_test_new = SelectKBest(chi2, k=5).fit(X_train, y_train).transform(X_test)
print("Accuracy with feature selection:", nb.score(X_test_new, y_test))
