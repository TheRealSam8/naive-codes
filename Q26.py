import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve

# Load binary classification dataset
X, y = load_breast_cancer(return_X_y=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM (use decision_function for precision-recall)
clf = SVC(kernel='rbf', probability=False)
clf.fit(X_train, y_train)

# Get decision scores
scores = clf.decision_function(X_test)

# Compute precision-recall pairs
precision, recall, _ = precision_recall_curve(y_test, scores)

# Plot
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (SVM)")
plt.grid(True)
plt.show()
