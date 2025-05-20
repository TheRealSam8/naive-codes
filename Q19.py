from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load a binary classification dataset
data = load_breast_cancer()
X, y = data.data, data.target  # y has 2 classes: 0 and 1

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Try different prior settings
priors_list = [None, [0.5, 0.5], [0.3, 0.7]]

for priors in priors_list:
    model = GaussianNB(priors=priors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Priors: {priors} -> Accuracy: {acc:.4f}")
