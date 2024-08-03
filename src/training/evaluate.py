from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .utils import weighted_binary_cross_entropy_loss


def evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    plot=False,
) -> Tuple[float, float, float]:
    R_est = weighted_binary_cross_entropy_loss(y_pred_proba, y_true)
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
            f"{model_name} – Empirical Risk Estimate: {R_est}, Accuracy: {acc}, F1 score: {f1}, Precision: {prec}, Recall: {rec}, AUC: {auc}\n{cm_df}"
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

    return R_est, acc, f1, auc
