from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score

# Load dataset with more than 5 features
data = load_wine()
X, y = data.data, data.target  # 13 features

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply RFE
svc = SVC(kernel='linear')
selector = RFE(estimator=svc, n_features_to_select=5)
selector.fit(X_train, y_train)

# Evaluate
y_pred = selector.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy after RFE (5 features):", accuracy)
