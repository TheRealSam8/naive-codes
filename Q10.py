# Laplace smoothing isn't directly applied in GaussianNB; it's more relevant in categorical NB (like Multinomial)
# Here's a sample setup just for comparative effect
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Accuracy without smoothing:", gnb.score(X_test, y_test))
# No Laplace smoothing in GaussianNB by design
