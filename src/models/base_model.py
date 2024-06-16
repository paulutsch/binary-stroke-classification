from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class BaseModel(ABC):
    def __init__(self, use_dependencies: bool = False, model_name: str = "BaseModel"):
        self.use_dependencies = use_dependencies
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
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        logger.info(
            f"Accuracy: {acc}, F1 score: {f1}, Precision: {prec}, Recall: {rec}"
        )

        return acc, f1
