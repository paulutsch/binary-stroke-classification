import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from ..data import StrokeDataset


class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.batch_size = 32

        self.linear = torch.nn.Linear(in_features=n_features, out_features=1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.04, weight_decay=0.005)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

    def fit(self, train_dataset: StrokeDataset, val_dataset: StrokeDataset):
        epochs = 100

        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.array([0, 1]), y=train_dataset.y
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print("class_weights", class_weights)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=False
        )

        losses_train = []
        losses_val = []
        accuracies = []
        for epoch in range(epochs):
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

            print(
                "Epoch: {}. Train Loss: {}. Val Loss: {}. Val Accuracy: {}".format(
                    epoch,
                    epoch_loss_train / len(train_loader),
                    epoch_loss_val / len(test_loader),
                    accuracy,
                )
            )

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
