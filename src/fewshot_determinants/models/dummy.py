from collections import Counter


class MajorityClassifier:
    def __init__(self):
        self.majority_label = 0

    def fit(self, y):
        self.majority_label = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X):
        return [self.majority_label for _ in range(len(X))]
