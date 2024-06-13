import pandas as pd


class DataLoader:
    def __init__(self, path: str = "data/raw/train.csv"):
        self.path = path
        self.cols = None
        self.df = None
        self.numpy_data = None

    def load_data(self):
        self.df = pd.read_csv(self.path)
        self.cols = self.df.columns
        print(self.df.info())
        print(self.df.columns)

    def to_numpy(self):
        if self.df is None:
            print("No data loaded!")
            return

        self.numpy_data = self.df.to_numpy()
        print(self.numpy_data)
