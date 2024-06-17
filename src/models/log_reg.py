from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogReg(BaseModel):
    def __init__(self, use_dependencies: bool = False):
        self.use_dependencies = use_dependencies
        if self.use_dependencies:
            self._model = LogisticRegression(
                class_weight="balanced",  # account for strong imbalance in data
                solver="liblinear",
            )
        else:
            self._model = None

    def fit(self, X: ArrayLike, y: ArrayLike):
        if self.use_dependencies:
            self._model.fit(X, y)
        else:
            pass

    def predict(self, X: ArrayLike) -> ndarray:
        if self.use_dependencies:
            return self._model.predict(X)
        else:
            pass

    def predict_proba(self, X: ArrayLike) -> ndarray:
        if self.use_dependencies:
            return self._model.predict_proba(X)[:, 1]
        else:
            pass
