from typing import Tuple

import numpy as np
import numpy.typing as npt

from .utils import (
    compute_class_weights,
    derivative_weighted_bce,
    sigmoid,
    sigmoid_prime,
    weighted_binary_cross_entropy_loss,
)


class BinaryLogisticRegression(object):
    def __init__(
        self,
        d_input: int,
        learning_rate: float = 0.005,
        batch_size: int = 32,
    ):
        """
        X: n_input x d_input
        Y: n_input

        self.W: d_input x 1
        self.B: 1
        """
        self.d_input = d_input
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.class_weights = np.zeros(2)

        # initialize weights and bias to zeros
        self.W = np.zeros(d_input)
        self.B = np.zeros(1)

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute forward pass of the logistic regression model.

        Y_hat: n_samples
        """
        z = np.dot(X, self.W) + self.B
        print("Shape of X:", X.shape)
        print("Shape of self.W:", self.W.shape)
        print("Shape of self.B:", self.B.shape)
        print("Shape of z:", z.shape)
        Y_hat = sigmoid(z)
        print("Shape of Y_hat:", Y_hat.shape)
        return Y_hat

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`

        pred: n_samples
        """
        y_hat = self.forward(X)
        pred = np.where(y_hat >= 0.5, 1, 0)

        return pred

    def predict_proba(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Predict probabilities for the input data.

        Y_hat: n_samples
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

        d_L_d_z = derivative_weighted_bce(Y_hat, Y, self.class_weights)
        print("Shape of d_L_d_z:", d_L_d_z.shape)
        print("Shape of X:", X.shape)

        d_L_d_W = np.dot(X.T, d_L_d_z) / batch_size
        d_L_d_B = np.sum(d_L_d_z, axis=0) / batch_size
        print("Shape of d_L_d_W:", d_L_d_W.shape)
        print("Shape of self.W:", self.W.shape)
        print("Shape of d_L_d_B:", d_L_d_B.shape, d_L_d_B)

        return d_L_d_W, d_L_d_B

    def fit(self, X: npt.ArrayLike, Y: npt.ArrayLike, epochs: int = 100):
        """
        Fit the logistic regression model to the training data.

        Parameters:
        X (npt.ArrayLike): Input data.
        Y (npt.ArrayLike): True labels.
        epochs (int): Number of training epochs.
        """
        n = X.shape[0]
        class_weights = compute_class_weights(Y)
        print(class_weights, type(class_weights))
        self.class_weights = compute_class_weights(Y)

        for epoch in range(epochs):
            for i in range(0, n, self.batch_size):
                X_batch = X[i : i + self.batch_size]
                Y_batch = Y[i : i + self.batch_size]

                Y_hat = self.forward(X_batch)
                d_L_d_W, d_L_d_B = self._backward(X_batch, Y_batch, Y_hat)

                self.W -= self.learning_rate * d_L_d_W
                self.B -= self.learning_rate * d_L_d_B

            loss = weighted_binary_cross_entropy_loss(
                Y_hat, Y_batch, self.class_weights
            )
            print(f"Epoch {epoch}: Loss: {loss}")

        print("Training complete.")
