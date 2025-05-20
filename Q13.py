# Use scikit-learn's email data or simulate with text data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = ["Free money!!!", "Hi, how are you?", "Claim your prize now!", "Let's schedule a meeting"]
labels = [1, 0, 1, 0]  # 1 = spam

vec = CountVectorizer()
X = vec.fit_transform(emails)

clf = MultinomialNB()
clf.fit(X, labels)
print("Prediction:", clf.predict(vec.transform(["Free vacation!"])))
