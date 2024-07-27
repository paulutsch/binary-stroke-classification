import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset


class StrokeDataset(Dataset):
    def __init__(self, X: NDArray, y: NDArray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_idx = self.X[idx]
        y_idx = self.y[idx]

        return torch.tensor(x_idx, dtype=torch.float32), torch.tensor(
            y_idx, dtype=torch.float32
        )
