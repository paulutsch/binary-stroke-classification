from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.utils.class_weight import compute_class_weight

from ..utils import weighted_binary_cross_entropy_loss
from .utils import (
    compute_class_weights,
    delta_weighted_bce,
    relu,
    relu_prime,
    sigmoid,
    sigmoid_prime,
)


class BinaryNeuralNetwork(object):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        n_hidden: int,
        epochs: int = 10,
        learning_rate: float = 0.05,
        batch_size: int = 32,
        lambda_reg: float = 0.01,
    ):
        """
        X: n_input x d_input
        Y: n_input

        self.W[0]: d_input x d_hidden
        self.B[0]: d_hidden

        self.W[i: 0 < i < (n_hidden-1)]: d_hidden x d_hidden
        self.B[i: 0 < i < (n_hidden-1)]: d_hidden

        self.W[-1]: d_hidden
        self.B[-1]: 1
        """
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg

        self.class_weights = np.zeros(2)

        self.W = []
        self.B = []

        self.initialize_params()

    def initialize_params(self):
        """Initialize weights and biases (Glorot initialization)"""
        for i in range(self.n_hidden + 1):
            if i == 0:  # input -> h_1
                limit = np.sqrt(6 / (self.d_input + self.d_hidden))
                w = np.random.uniform(-limit, limit, (self.d_input, self.d_hidden))
                b = np.random.uniform(-limit, limit, (self.d_hidden,))
            elif i < self.n_hidden:  # h_i -> h_{i+1}
                limit = np.sqrt(6 / (self.d_hidden + self.d_hidden))
                w = np.random.uniform(-limit, limit, (self.d_hidden, self.d_hidden))
                b = np.random.uniform(-limit, limit, (self.d_hidden,))
            else:  # h_{n} -> output
                limit = np.sqrt(6 / (self.d_hidden + 1))
                w = np.random.uniform(-limit, limit, (self.d_hidden, 1))
                b = np.random.uniform(-limit, limit, (1,))

            self.W.append(w)
            print(f"Shape of self.W[{i+1}]:", self.W[i].shape)
            self.B.append(b)
            print(f"Shape of self.B[{i+1}]:", self.B[i].shape)

    def forward(self, X: npt.ArrayLike, return_intermediates=False) -> npt.ArrayLike:
        """
        Compute forward pass of the neural network.

        Y_hat: n_samples
        """
        A = [X]
        Z = []

        # forward propagation
        a_i = X
        for i in range(self.n_hidden + 1):
            z_i = np.dot(a_i, self.W[i]) + self.B[i]  # add bias row-wise

            if i < self.n_hidden:
                a_i = relu(z_i)
            else:  # output layer
                z_i = z_i.squeeze()  # transform (n_samples, 1) to (n_samples,)
                a_i = sigmoid(z_i)  # (n_samples,)

            Z.append(z_i)
            A.append(a_i)

        print("Shape of A[-1]:", A[-1].shape)
        Y_hat = A[-1].squeeze()
        print("Shape of Y_hat:", Y_hat.shape)

        if return_intermediates:
            return Y_hat, A, Z

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
        Create a prediction matrix with `self.forward()`

        y_hat: n_samples
        """
        y_hat = self.forward(X)
        return y_hat

    def _backward(
        self,
        Y: npt.ArrayLike,
        Y_hat: npt.ArrayLike,
        A: npt.ArrayLike,
        Z: npt.ArrayLike,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Compute back propagation of the neural network.
        """
        batch_size = Y.shape[0]

        d_L_d_W = [np.zeros_like(w) for w in self.W]
        d_L_d_B = [np.zeros_like(b) for b in self.B]

        # derivative of the loss with respect to the pre-activation of the output layer
        delta_y = delta_weighted_bce(Y_hat, Y, self.class_weights) * sigmoid_prime(
            Z[-1]
        )  # multiply with the derivative of the activation function here, because dL/dy * dy/dz = dL/dz
        delta_y = delta_y.reshape(-1, 1)  # reshape from (n_samples,) to (n_samples, 1)

        # derivatives of the loss with respect to the weights and biases of the last layer
        d_L_d_W[-1] = np.dot(A[-2].T, delta_y) / batch_size
        d_L_d_B[-1] = np.sum(delta_y, axis=0) / batch_size

        delta_l = delta_y
        print("Shape of delta_l:", delta_l.shape)

        # back propagation using chain rule
        for l in range(self.n_hidden - 1, -1, -1):
            delta_l = np.dot(delta_l, self.W[l + 1].T) * relu_prime(
                Z[l]
            )  # d^l = (d^{l+1})^T * W^{l+1} x (dA^{l} / dz^{l})

            d_L_d_W[l] = (
                np.dot(A[l].T, delta_l) / batch_size
            )  # note: A[l] is actually correct here – A[0] is X!
            d_L_d_B[l] = np.sum(delta_l, axis=0) / batch_size

            # add L2 regularization derivative
            d_L_d_W[l] += self.lambda_reg * self.W[l]

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
        Fit the neural network to the training data.
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

                Y_hat, A, Z = self.forward(X_batch, return_intermediates=True)
                d_L_d_W, d_L_d_B = self._backward(Y_batch, Y_hat, A, Z)

                for j in range(self.n_hidden + 1):
                    self.W[j] -= self.learning_rate * d_L_d_W[j]
                    self.B[j] -= self.learning_rate * d_L_d_B[j]

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
