"""
Random Forest Implementation
Task: Build an ensemble of CART trees using Bagging and Feature Subsampling.
"""

import random
import math
from cart import CARTTree

def bootstrap_sample(X, y):
    """
    TODO: Generate a bootstrap sample of the dataset.
    Randomly sample len(X) items WITH REPLACEMENT from X and y.
    Returns: X_sample, y_sample
    """
    pass

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        """
        TODO: Train the random forest.
        For each of the `n_trees`:
        1. Create a bootstrap sample.
        2. Determine the subset of features to use based on `max_features` 
           (e.g., math.isqrt(len(total_features))).
        3. Initialize a CART tree and modify it to only consider the feature subset during splits.
        4. Train the tree and append it to self.trees.
        """
        pass

    def predict(self, x):
        """
        TODO: Predict the label for a single instance using majority voting.
        1. Get predictions from all trees in self.trees.
        2. Return the most common predicted label.
        """
        pass

    def accuracy(self, X, y):
        correct = sum(1 for x_i, y_i in zip(X, y) if self.predict(x_i) == y_i)
        return correct / len(y)
