from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.utils.class_weight import compute_class_weight

from ..utils import weighted_binary_cross_entropy_loss
from .utils import compute_class_weights, delta_weighted_bce, sigmoid, sigmoid_prime


class BinaryLogisticRegression(object):
    def __init__(
        self,
        n_features: int,
        epochs: int = 20,
        learning_rate: float = 0.1,
        batch_size: int = 8,
        lambda_reg: float = 0.01,
    ):
        """
        X: (n_samples, n_features)
        Y: (n_samples,)

        self.W: (n_features, 1)
        self.B: (1,)
        """
        self.n_features = n_features
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg

        self.class_weights = np.zeros(2)

        # initialize weights and bias to zeros
        self.W = np.zeros(n_features)
        self.B = 0.0

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute forward pass of the logistic regression model.

        Y_hat: (n_samples,)
        """
        z = np.dot(X, self.W) + self.B
        Y_hat = sigmoid(z)
        return Y_hat

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`

        pred: (n_samples,)
        """
        y_hat = self.forward(X)
        pred = np.where(y_hat >= 0.5, 1, 0)

        return pred

    def predict_proba(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Predict probabilities for the input data.

        Y_hat: (n_samples,)
        """
        Y_hat = self.forward(X)
        return Y_hat

    def _backward(
        self,
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        Y_hat: npt.ArrayLike,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Compute back propagation for logistic regression.
        """
        batch_size = X.shape[0]

        # derivative of the loss with respect to the pre-activation of the output layer
        # equivalent to d_L_d_z = Y_hat - Y in non-weighted binary cross-entropy
        d_L_d_z = delta_weighted_bce(Y_hat, Y, self.class_weights)

        d_L_d_W = np.dot(X.T, d_L_d_z) / batch_size
        d_L_d_B = np.sum(d_L_d_z, axis=0) / batch_size

        # add L2 regularization derivative
        d_L_d_W += self.lambda_reg * self.W

        return d_L_d_W, d_L_d_B

    def fit(
        self,
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        X_val: npt.ArrayLike,
        Y_val: npt.ArrayLike,
        plot: bool = False,
    ):
        """
        Fit the logistic regression model to the training data.
        """
        n = X.shape[0]

        self.class_weights = compute_class_weight(
            class_weight="balanced", classes=np.array([0, 1]), y=Y
        )

        losses_train = []
        losses_val = []

        for epoch in range(self.epochs):
            epoch_loss_train = 0.0
            for i in range(0, n, self.batch_size):
                X_batch = X[i : i + self.batch_size]
                Y_batch = Y[i : i + self.batch_size]

                Y_hat = self.forward(X_batch)
                d_L_d_W, d_L_d_B = self._backward(X_batch, Y_batch, Y_hat)

                self.W -= self.learning_rate * d_L_d_W
                self.B -= self.learning_rate * d_L_d_B

                batch_loss = weighted_binary_cross_entropy_loss(
                    Y_hat, Y_batch, self.class_weights
                )
                epoch_loss_train += batch_loss * len(Y_batch)

            loss_train = epoch_loss_train / n

            Y_hat_val = self.forward(X_val)
            loss_val = weighted_binary_cross_entropy_loss(
                Y_hat_val, Y_val, self.class_weights
            )

            losses_train.append(loss_train)
            losses_val.append(loss_val)

            if plot:
                print(f"Epoch {epoch}: Train Loss: {loss_train}, Val Loss: {loss_val}")

        if plot:
            plt.plot(losses_train, label="Training Loss")
            plt.plot(losses_val, label="Validation Loss")
            plt.xlabel("Number of epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()
            print("Training complete.")
