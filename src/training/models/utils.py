import numpy as np
import numpy.typing as npt

from ..utils import compute_class_weights


def sigmoid(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Sigmoid function with input clipping to avoid overflow.
    """

    z = np.asarray(z)
    result = np.zeros_like(z)
    negative_mask = z < 0
    result[negative_mask] = np.exp(z[negative_mask]) / (1 + np.exp(z[negative_mask]))
    result[~negative_mask] = 1 / (1 + np.exp(-z[~negative_mask]))
    return result


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
    Returns dL/dz = dL/dy_hat * dy_hat/dz = dL/dy_hat * sigmoid_prime(z) = weighted([Y / Y_hat] - [(1-Y) / (1-Y_hat)]) * sigmoid(z) * (1 - sigmoid(z))
    """
    if class_weights is None:
        class_weights = compute_class_weights(Y)

    weight_per_sample = np.where(Y == 1, class_weights[1], class_weights[0])
    d_L_d_z = weight_per_sample * (Y_hat - Y)

    return d_L_d_z
