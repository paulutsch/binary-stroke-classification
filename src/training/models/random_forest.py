import numpy as np
from numpy import ndarray

class RandomForest:
    def __init__(self, n_trees: int = 5):
        pass

    def forward(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass


class DecisionTree:
    def __init__(self):
        pass

    def _build_tree(self):
        pass

    def _information_gain(self):
        pass

    def _entropy(self):
        pass

    def _information(self, y_i, y):
        return -np.log2(self._p(y_i, y))

    def _p(self, y_i, y: ndarray):
        n_y_i = np,sum(y == y_i)
        n_y = len(y)
        return n_y_i / n_y
