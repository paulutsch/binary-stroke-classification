import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike

from src.data_preparation import StrokeDataset


class NaiveBaseline:
    def __init__(self):
        super().__init__()
        self.model = None
        self.p_true = 0.0

    def fit(
        self,
        train_dataset: StrokeDataset,
        val_dataset: StrokeDataset,
    ):
        self.p_true = train_dataset.y.mean()

        val_preds = self.predict(val_dataset.X)
        val_accuracy = (val_preds == val_dataset.y).mean()
        print(f"Validation accuracy: {val_accuracy}")

    def predict(self, X: ArrayLike) -> ndarray:
        """returns a random prediction based on the base probability of the true class, without any training"""
        return np.random.choice([0, 1], size=len(X), p=[1 - self.p_true, self.p_true])

    def predict_proba(self, X: ArrayLike) -> ndarray:
        return np.full(len(X), self.p_true)
