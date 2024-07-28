from abc import ABC, abstractmethod
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold

from src.data_preparation import StrokeDataset


def scores(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    plot=False,
) -> Tuple[float, float, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    cm_df = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=["Actual Neg", "Actual Pos"],
        columns=["Pred Neg", "Pred Pos"],
    )

    if plot:
        logger.info(
            f"{model_name} – Accuracy: {acc}\nF1 score: {f1}\nPrecision: {prec}\nRecall: {rec}\nAUC: {auc}\n{cm_df}"
        )

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.figure()
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc:0.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic - {model_name}")
        plt.legend(loc="lower right")
        plt.show()

    return acc, f1, auc


def evaluate(model, X, y, k=10):
    kf = KFold(n_splits=k, shuffle=True)

    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        dataset_train_fold = StrokeDataset(X_train_fold, y_train_fold)
        dataset_val_fold = StrokeDataset(X_val_fold, y_val_fold)

        model.fit(
            dataset_train_fold,
            dataset_val_fold,
            plot=False,
        )
        y_pred_fold = model.predict(X_val_fold)
        y_pred_fold_proba = model.predict_proba(X_val_fold)
        acc, f1, auc = evaluate(
            "Logistic Regression",
            y_val_fold,
            y_pred_fold,
            y_pred_fold_proba,
            plot=False,
        )
        scores.append(auc)
    avg_score = np.mean(scores)
    return avg_score, std_dev
