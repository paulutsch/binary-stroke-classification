from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score


class BaseModel(ABC):
    def __init__(self):
        self._model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> ndarray:
        pass

    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[float, float]:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return acc, f1
