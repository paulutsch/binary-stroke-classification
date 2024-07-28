import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.data_preparation import StrokeDataset


class NeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        n_features,
        n_hidden: int = 16,
        epochs=10,
        lr=0.05,
        batch_size=32,
        lambda_reg=0.01,
    ):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.l1 = torch.nn.Linear(in_features=n_features, out_features=n_hidden)
        self.l2 = torch.nn.Linear(in_features=n_hidden, out_features=1)

        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, weight_decay=lambda_reg
        )

    def forward(self, x):
        a1 = torch.sigmoid(self.l1(x))
        y_pred = torch.sigmoid(self.l2(a1))
        return y_pred

    def fit(self, train_dataset: StrokeDataset, val_dataset: StrokeDataset, plot=False):
        class_weights = torch.tensor(
            compute_class_weight(
                class_weight="balanced", classes=np.array([0, 1]), y=train_dataset.y
            ),
            dtype=torch.float32,
        )

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=False
        )

        losses_train = []
        losses_val = []
        accuracies = []
        for epoch in range(self.epochs):
            self.train()
            epoch_loss_train = 0.0
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred_batch = self(x_batch)
                loss = self.weighted_binary_cross_entropy(
                    y_pred_batch, y_batch.view(-1, 1).float(), weights=class_weights
                )
                loss.backward()
                self.optimizer.step()
                epoch_loss_train += loss.item()
            losses_train.append(epoch_loss_train / len(train_loader))

            self.eval()
            epoch_loss_val = 0.0
            n_correct = 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    y_pred_batch = self(x_batch)
                    loss = self.weighted_binary_cross_entropy(
                        y_pred_batch, y_batch.view(-1, 1).float(), weights=class_weights
                    )
                    epoch_loss_val += loss.item()
                    preds = (y_pred_batch > 0.5).float()
                    n_correct += (preds.view(-1) == y_batch).sum().item()
            losses_val.append(epoch_loss_val / len(test_loader))
            accuracy = 100 * n_correct / len(val_dataset)
            accuracies.append(accuracy)

            if plot:
                print(
                    "Epoch: {}. Train Loss: {}. Val Loss: {}. Val Accuracy: {}".format(
                        epoch,
                        epoch_loss_train / len(train_loader),
                        epoch_loss_val / len(test_loader),
                        accuracy,
                    )
                )
        if plot:
            plt.plot(losses_train, label="Training Loss")
            plt.plot(losses_val, label="Validation Loss")
            plt.xlabel("Number of epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()

    def predict(self, X: ArrayLike) -> ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_pred = self(X_tensor)
            preds = (y_pred > 0.5).float().numpy()
        return preds

    def predict_proba(self, X: ArrayLike) -> ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_pred = self(X_tensor).numpy()[:, 0]
        return y_pred

    def weighted_binary_cross_entropy(self, output, target, weights=None):
        if weights is not None:
            assert len(weights) == 2

            loss = weights[1] * (target * torch.log(output)) + weights[0] * (
                (1 - target) * torch.log(1 - output)
            )
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))
