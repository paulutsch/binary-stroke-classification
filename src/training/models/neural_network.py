from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.utils.class_weight import compute_class_weight

from ..utils import weighted_binary_cross_entropy_loss
from .utils import (
    compute_class_weights,
    derivative_weighted_bce,
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
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.class_weights = np.zeros(2)

        self.W = []
        self.B = []

        self.m_W = []
        self.v_W = []
        self.m_B = []
        self.v_B = []

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

            self.m_W.append(np.zeros_like(w))
            self.v_W.append(np.zeros_like(w))
            self.m_B.append(np.zeros_like(b))
            self.v_B.append(np.zeros_like(b))

    def forward(self, X: npt.ArrayLike, return_intermediates=False) -> npt.ArrayLike:
        """
        Compute forward pass of the neural network.

        Y_hat: n_samples
        """
        H = [X]
        Z = []

        # forward propagation
        h_i = X
        for i in range(self.n_hidden + 1):
            z_i = np.dot(h_i, self.W[i]) + self.B[i]

            if i < self.n_hidden:
                h_i = relu(z_i)
            else:  # output layer
                z_i = z_i.squeeze()
                h_i = sigmoid(z_i)

            Z.append(z_i)
            H.append(h_i)

        Y_hat = H[-1].squeeze()

        if return_intermediates:
            return Y_hat, H, Z

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
        H: npt.ArrayLike,
        Z: npt.ArrayLike,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Compute back propagation of the neural network.
        """
        batch_size = Y.shape[0]

        d_L_d_W = [np.zeros_like(w) for w in self.W]
        d_L_d_B = [np.zeros_like(b) for b in self.B]

        # derivative of the loss with respect to the pre-activation of the output layer
        d_L_d_z = derivative_weighted_bce(Y_hat, Y, self.class_weights)
        print("Shape of d_L_d_z:", d_L_d_z.shape)

        # derivatives of the loss with respect to the weights and biases of the last layer
        d_L_d_W[-1] = np.dot(H[-2].T, d_L_d_z) / batch_size
        d_L_d_W[-1] = d_L_d_W[-1].reshape(-1, 1)
        d_L_d_B[-1] = np.sum(d_L_d_z, axis=0) / batch_size

        # back propagation using chain rule
        for i in range(self.n_hidden - 1, -1, -1):
            d_L_d_z = np.dot(
                d_L_d_z.reshape(batch_size, -1), self.W[i + 1].T
            ) * relu_prime(Z[i])
            d_L_d_W[i] = np.dot(H[i].T, d_L_d_z) / batch_size
            d_L_d_B[i] = np.sum(d_L_d_z, axis=0) / batch_size

            # add L2 regularization derivative
            d_L_d_W[i] += self.lambda_reg * self.W[i]

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

        # Adam optimizer hyperparameters
        t = 0

        for epoch in range(self.epochs):
            epoch_loss_train = 0.0
            for i in range(0, n, self.batch_size):
                X_batch = X[i : i + self.batch_size]
                Y_batch = Y[i : i + self.batch_size]

                Y_hat, A, Z = self.forward(X_batch, return_intermediates=True)
                d_L_d_W, d_L_d_B = self._backward(Y_batch, Y_hat, A, Z)

                t += 1  # Increment time step

                for j in range(self.n_hidden + 1):
                    # Update biased first moment estimate
                    self.m_W[j] = (
                        self.beta1 * self.m_W[j] + (1 - self.beta1) * d_L_d_W[j]
                    )
                    self.m_B[j] = (
                        self.beta1 * self.m_B[j] + (1 - self.beta1) * d_L_d_B[j]
                    )

                    # Update biased second raw moment estimate
                    self.v_W[j] = self.beta2 * self.v_W[j] + (1 - self.beta2) * (
                        d_L_d_W[j] ** 2
                    )
                    self.v_B[j] = self.beta2 * self.v_B[j] + (1 - self.beta2) * (
                        d_L_d_B[j] ** 2
                    )

                    # Compute bias-corrected first moment estimate
                    m_W_hat = self.m_W[j] / (1 - self.beta1**t)
                    m_B_hat = self.m_B[j] / (1 - self.beta1**t)

                    # Compute bias-corrected second raw moment estimate
                    v_W_hat = self.v_W[j] / (1 - self.beta2**t)
                    v_B_hat = self.v_B[j] / (1 - self.beta2**t)

                    # Update parameters
                    self.W[j] -= (
                        self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                    )
                    self.B[j] -= (
                        self.learning_rate * m_B_hat / (np.sqrt(v_B_hat) + self.epsilon)
                    )

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
