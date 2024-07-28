import numpy as np
import numpy.typing as npt


def weighted_binary_cross_entropy_loss(
    output, target, class_weights=None, epsilon=0.001
):
    """
    Compute the weighted binary cross-entropy loss.
    """
    if class_weights is None:
        class_weights = compute_class_weights(target)

    # avoiding numerical instability
    output = np.clip(output, epsilon, 1 - epsilon)

    loss = class_weights[1] * (target * np.log(output)) + class_weights[0] * (
        (1 - target) * np.log(1 - output)
    )

    return -np.mean(loss)


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
