import math


# ======================================================
# Utility functions
# ======================================================

def count_items(items):
    counts = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return counts


def class_probabilities(labels):
    total = float(len(labels))
    counts = count_items(labels)
    return [counts[label] / total for label in counts]


def entropy(probabilities):
    return sum(-p * math.log2(p) for p in probabilities if p > 0)


def data_entropy(labels):
    return entropy(class_probabilities(labels))


# ======================================================
# Generalized Decision Tree (ID3)
# ======================================================

class DecisionTree:

    # ---------------- Internal Nodes ----------------

    class _Leaf:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return self.value

    class _Split:
        def __init__(self, attribute, branches, default_value):
            self.attribute = attribute
            self.branches = branches
            self.default_value = default_value

        def predict(self, x):
            val = x.get(self.attribute)
            if val not in self.branches:
                return self.default_value
            return self.branches[val].predict(x)

    # ---------------- Public API ----------------

    def __init__(self):
        self.root = None

    def fit(self, X, y):
        """
        X: list of dicts (feature_name -> value)
        y: list of labels
        """
        attributes = list(X[0].keys())
        self.root = self._build_tree(X, y, attributes)

    def predict(self, x):
        if self.root is None:
            raise ValueError("Tree not trained")
        return self.root.predict(x)

    def predict_all(self, X):
        return [self.predict(x) for x in X]

    def accuracy(self, X, y):
        correct = 0
        for x, label in zip(X, y):
            if self.predict(x) == label:
                correct += 1
        return correct / len(y)

    # ---------------- Tree Construction ----------------

    def _partition_by(self, X, y, attribute):
        partitions = {}
        for xi, yi in zip(X, y):
            key = xi[attribute]
            if key not in partitions:
                partitions[key] = ([], [])
            partitions[key][0].append(xi)
            partitions[key][1].append(yi)
        return partitions

    def _partition_entropy_by(self, X, y, attribute):
        partitions = self._partition_by(X, y, attribute)

        total = len(y)
        entropy_sum = 0.0

        for subset_y in [p[1] for p in partitions.values()]:
            entropy_sum += (len(subset_y) / total) * data_entropy(subset_y)

        return entropy_sum

    def _build_tree(self, X, y, attributes):

        label_counts = count_items(y)
        majority_label = max(label_counts, key=lambda k: label_counts[k])

        # Stop conditions
        if len(label_counts) == 1:
            return self._Leaf(majority_label)

        if not attributes:
            return self._Leaf(majority_label)

        # Choose best attribute
        best_attr = min(
            attributes,
            key=lambda a: self._partition_entropy_by(X, y, a)
        )

        partitions = self._partition_by(X, y, best_attr)
        remaining_attrs = [a for a in attributes if a != best_attr]

        branches = {}
        for attr_val, (subX, suby) in partitions.items():
            branches[attr_val] = self._build_tree(subX, suby, remaining_attrs)

        return self._Split(best_attr, branches, majority_label)
