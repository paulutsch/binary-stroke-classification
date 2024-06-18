from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(
    model_name: str, y_true: ArrayLike, y_pred: ArrayLike, y_pred_proba: ArrayLike
) -> Tuple[float, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    logger.info(
        f"{model_name} – Accuracy: {acc}, F1 score: {f1}, Precision: {prec}, Recall: {rec}, AUC: {auc}"
    )
    logger.info(f"number of pos preds: {sum(y_pred > 0.5)}")

    return acc, f1, auc
