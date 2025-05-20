# Python script Q25 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target  # y has 2 classes: 0 and 1

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)
probs = nb.predict_proba(X_test)
print("ROC-AUC Score:", roc_auc_score(y_test, probs[:, 1]))
