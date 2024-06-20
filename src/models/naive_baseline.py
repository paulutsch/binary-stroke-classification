import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from ..data import StrokeDataset


class NaiveBaseline:
    def __init__(self):
        super().__init__()
        self.model = None

    def predict(self, X: ArrayLike) -> ndarray:
        return np.zeros(len(X))

    def predict_proba(self, X: ArrayLike) -> ndarray:
        return np.zeros(len(X))