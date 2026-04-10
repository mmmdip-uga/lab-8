"""
CART (Classification and Regression Trees) Implementation
Task: Modify the provided ID3 logic to use Gini Impurity and strictly Binary Splits.
"""

def gini_impurity(labels):
    """
    TODO: Calculate the Gini impurity of a list of labels.
    Formula: 1 - sum((p_i)^2)
    """
    pass


def weighted_gini(X, y, attribute, split_value):
    """
    TODO: Calculate the weighted Gini impurity of a binary split.
    The split is defined by: x[attribute] == split_value (True branch)
                             x[attribute] != split_value (False branch)
    """
    pass


class CARTTree:

    class _Leaf:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return self.value

    class _BinarySplit:
        def __init__(self, attribute, split_value, true_branch, false_branch, default_value):
            self.attribute = attribute
            self.split_value = split_value
            self.true_branch = true_branch
            self.false_branch = false_branch
            self.default_value = default_value

        def predict(self, x):
            # TODO: Implement the prediction traversal for a binary split.
            # If x[self.attribute] == self.split_value, go to true_branch.
            # Otherwise, go to false_branch. Handle missing attributes safely.
            pass

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, y):
        attributes = list(X[0].keys())
        self.root = self._build_tree(X, y, attributes, depth=0)

    def predict(self, x):
        if self.root is None:
            raise ValueError("Tree not trained")
        return self.root.predict(x)

    def accuracy(self, X, y):
        correct = sum(1 for x_i, y_i in zip(X, y) if self.predict(x_i) == y_i)
        return correct / len(y)

    def _partition_by(self, X, y, attribute, split_value):
        """
        TODO: Partition the dataset into two subsets (True and False) 
        based on whether x[attribute] == split_value.
        Returns: (X_true, y_true), (X_false, y_false)
        """
        pass

    def _build_tree(self, X, y, attributes, depth):
        """
        TODO: Implement the recursive tree-building logic with stopping criteria.
        
        Stopping Criteria Checklist:
        1. Are all labels identical? (Pure node)
        2. Is depth >= self.max_depth?
        3. Is len(y) < self.min_samples_split?
        If any of the above are true, return a _Leaf.

        Split Logic:
        1. Calculate current node's Gini impurity.
        2. Find best attribute and split_value that minimizes weighted Gini.
        3. Calculate impurity decrease: (current_gini - weighted_gini).
        
        Post-Split Checks:
        1. Is impurity decrease < self.min_impurity_decrease?
        2. Does either the true or false branch have fewer than self.min_samples_leaf rows?
        If either is true, cancel the split and return a _Leaf.
        
        Otherwise, recursively build true and false branches and return a _BinarySplit.
        """
        pass
