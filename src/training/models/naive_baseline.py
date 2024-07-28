import numpy as np
import numpy.typing as npt
from numpy import ndarray
from numpy.typing import ArrayLike


class NaiveBaseline:
    def __init__(self):
        super().__init__()
        self.model = None
        self.p_true = 0.0

    def fit(
        self,
        X_train: npt.ArrayLike,
        y_train: npt.ArrayLike,
        X_val: npt.ArrayLike,
        y_val: npt.ArrayLike,
    ):
        self.p_true = y_train.mean()

        val_preds = self.predict(X_val)
        val_accuracy = (val_preds == y_val).mean()
        print(f"Validation accuracy: {val_accuracy}")

    def predict(self, X: ArrayLike) -> ndarray:
        """returns a random prediction based on the base probability of the true class, without any training"""
        return np.random.choice([0, 1], size=len(X), p=[1 - self.p_true, self.p_true])

    def predict_proba(self, X: ArrayLike) -> ndarray:
        return np.full(len(X), self.p_true)
