from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_features=10, n_informative=5, n_classes=2, random_state=0)
X = (X > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
print("Accuracy:", bnb.score(X_test, y_test))
