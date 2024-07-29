from collections import Counter
from typing import List, Set, Tuple

import numpy as np
import numpy.typing as npt

from ..utils import compute_class_weights


def sigmoid(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Sigmoid function with input clipping to avoid overflow.
    """
    z = np.clip(z, -500, 500)  # avoiding overflow
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of the sigmoid function. d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
    """
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    return np.maximum(0, z)


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    return np.where(z > 0, 1, 0)  # return z_i for z_i > 0, else 0


def delta_weighted_bce(Y_hat, Y, class_weights=None, epsilon=0.001) -> npt.ArrayLike:
    """
    Compute the derivative of the weighted binary cross entropy loss.

    Parameters:
    Y_hat (npt.ArrayLike): Predicted probabilities.
    Y (npt.ArrayLike): True labels.
    class_weights (npt.ArrayLike): Weights for the classes.
    epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-15.

    Returns:
    npt.ArrayLike: Derivative of the loss with respect to Y_hat.
    """
    if class_weights is None:
        class_weights = compute_class_weights(Y)

    # avoiding division by zero
    Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)

    dL_dy_hat = -(class_weights[1] * Y / Y_hat) + (
        class_weights[0] * (1 - Y) / (1 - Y_hat)
    )
    return dL_dy_hat
