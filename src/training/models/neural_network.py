from typing import Tuple

import numpy as np
import numpy.typing as npt

from .utils import (
    compute_class_weights,
    derivative_weighted_bce,
    relu,
    relu_prime,
    sigmoid,
    sigmoid_prime,
    weighted_binary_cross_entropy_loss,
)


class BinaryNeuralNetwork(object):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        n_hidden: int,
        learning_rate: int = 0.005,
        batch_size: int = 32,
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
        self.learning_rate = learning_rate
        self.batch_size = batch_size

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
        H = [X]
        Z = []

        # forward propagation
        h_i = X
        for i in range(self.n_hidden + 1):
            z_i = np.dot(h_i, self.W[i]) + self.B[i]
            print("Shape of z_i:", z_i.shape)

            if i < self.n_hidden:
                h_i = relu(z_i)
                print(f"Shape of h_{i+1} relu:", h_i.shape)
            else:  # output layer
                print(f"Shape of z_{i+1} in output:", z_i.shape)
                z_i = z_i.squeeze()
                h_i = sigmoid(z_i)
                print(f"Shape of h_{i+1} in output:", h_i.shape)
                print(f"Shape of self.W[{i+1}] in output:", self.W[i].shape)
                print(f"Shape of self.B[{i+1}] in output:", self.B[i].shape)

                print(f"Shape of h_{i+1} sigmoid:", h_i.shape)

            Z.append(z_i)
            H.append(h_i)

        Y_hat = H[-1].squeeze()
        print("Shape of Y_hat:", Y_hat.shape)

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
        print("Shape of pred:", pred.shape)

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

        # derivative of the loss with respect to the pre-activation of output layer
        d_L_d_z = derivative_weighted_bce(Y_hat, Y, self.class_weights) * sigmoid_prime(
            Z[-1]
        )
        print("Shape of d_L_d_z:", d_L_d_z.shape)

        # derivatives of the loss with respect to the weights and biases of last layer
        d_L_d_W[-1] = np.dot(H[-2].T, d_L_d_z) / batch_size
        d_L_d_W[-1] = d_L_d_W[-1].reshape(-1, 1)
        print("Shape of d_L_d_W[-1]:", d_L_d_W[-1].shape)
        d_L_d_B[-1] = np.sum(d_L_d_z, axis=0) / batch_size
        print("Shape of d_L_d_B[-1]:", d_L_d_B[-1].shape)

        # back propagation using chain rule
        for i in range(self.n_hidden - 1, -1, -1):
            print(f"Shape of d_L_d_z:", d_L_d_z.reshape(batch_size, -1).shape)  # (32,1)
            print(f"Shape of self.W[{i+1}]:", self.W[i + 1].shape)  # (1,32)
            d_L_d_z = np.dot(
                d_L_d_z.reshape(batch_size, -1), self.W[i + 1].T
            ) * relu_prime(Z[i])
            print("Shape of d_L_d_z:", d_L_d_z.shape)
            d_L_d_W[i] = np.dot(H[i].T, d_L_d_z) / batch_size
            print(f"Shape of d_L_d_W[{i+1}]:", d_L_d_W[i].shape)
            d_L_d_B[i] = np.sum(d_L_d_z, axis=0) / batch_size
            print(f"Shape of d_L_d_B[{i+1}]:", d_L_d_B[i].shape)

        return d_L_d_W, d_L_d_B

    def fit(self, X: npt.ArrayLike, Y: npt.ArrayLike, epochs: int = 100):
        """
        Fit the neural network to the training data.
        """
        n = X.shape[0]
        self.class_weights = compute_class_weights(Y)

        for epoch in range(epochs):
            for i in range(0, n, self.batch_size):
                X_batch = X[i : i + self.batch_size]
                Y_batch = Y[i : i + self.batch_size]
                print("Shape of X_batch:", X_batch.shape)
                print("Shape of Y_batch:", Y_batch.shape)

                Y_hat, A, Z = self.forward(X_batch, return_intermediates=True)
                d_L_d_W, d_L_d_B = self._backward(Y_batch, Y_hat, A, Z)

                for j in range(self.n_hidden + 1):
                    print(f"Shape of self.W[{j+1}]:", self.W[j].shape)
                    print(f"Shape of d_L_d_W[{j+1}]:", d_L_d_W[j].shape)
                    print(f"Shape of self.B[{j+1}]:", self.B[j].shape)
                    print(f"Shape of d_L_d_B[{j+1}]:", d_L_d_B[j].shape)

                    self.W[j] -= self.learning_rate * d_L_d_W[j]
                    self.B[j] -= self.learning_rate * d_L_d_B[j]

            loss = weighted_binary_cross_entropy_loss(
                Y_hat, Y_batch, self.class_weights
            )

            print(f"Epoch {epoch}: Loss: {loss}")

        print("Training complete.")
