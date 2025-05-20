from sklearn.metrics import log_loss
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load a binary classification dataset
data = load_breast_cancer()
X, y = data.data, data.target  # y has 2 classes: 0 and 1

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
nb = GaussianNB()
nb.fit(X_train, y_train)
probs = nb.predict_proba(X_test)
print("Log Loss:", log_loss(y_test, probs))
