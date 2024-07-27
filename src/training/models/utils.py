from collections import Counter
from typing import List, Set, Tuple

import numpy as np
import numpy.typing as npt


def sigmoid(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Sigmoid function.
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of the sigmoid function. d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
    """
    derivative = sigmoid(z) * (1 - sigmoid(z))
    print("Shape of sigmoid derivative:", derivative.shape)
    return derivative


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    return np.maximum(0, z)


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    derivative = np.where(z > 0, 1, 0)
    print("Shape of ReLU derivative:", derivative.shape)
    return np.where(z > 0, 1, 0)  # return z_i for z_i > 0, else 0


def compute_class_weights(y: npt.ArrayLike) -> npt.ArrayLike:
    """
    Compute class weights for imbalanced datasets.
    """
    y = np.array(y, dtype=np.int64)
    classes = np.unique(y)
    class_counts = np.bincount(y)
    total_samples = len(y)

    class_weights = {
        cls: total_samples / (len(classes) * count)
        for cls, count in zip(classes, class_counts)
    }

    weights_array = np.array([class_weights[cls] for cls in classes], dtype=np.float64)

    return weights_array


def weighted_binary_cross_entropy_loss(output, target, class_weights=None):
    """
    Compute the weighted binary cross-entropy loss.
    """
    if class_weights is None:
        class_weights = compute_class_weights(target)

    loss = class_weights[1] * (target * np.log(output)) + class_weights[0] * (
        (1 - target) * np.log(1 - output)
    )

    return -np.mean(loss)


def derivative_weighted_bce(Y_hat, Y, class_weights=None):
    if class_weights is None:
        class_weights = compute_class_weights(Y)

    weights_per_data_point = np.array([class_weights[int(y)] for y in Y])

    print("shape of weights_per_data_point: ", weights_per_data_point.shape)
    print("shape of Y_hat: ", Y_hat.shape, Y_hat[0])
    print("shape of Y: ", Y.shape, Y[0])

    d_L_d_z = (Y_hat - Y) * weights_per_data_point
    print("shape of d_L_d_z: ", d_L_d_z.shape)

    return d_L_d_z
