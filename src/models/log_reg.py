from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogReg(BaseModel):
    def __init__(self):
        super(LogReg, self).__init__()
        self._model = LogisticRegression()

    def fit(self, X: ArrayLike, y: ArrayLike):
        self._model.fit(X, y)

    def predict(self, X: ArrayLike) -> ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: ArrayLike) -> ndarray:
        return self._model.predict_proba(X)
