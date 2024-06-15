import pandas as pd
from loguru import logger


class DataLoader:
    def __init__(self):
        pass

    def load_data(
        self,
        path: str = "data/raw/train.csv",
    ):
        df = pd.read_csv(path)
        logger.success("Successfully created Pandas Dataframe from raw data.\n")

        return df

    def separate_data(self, df: pd.DataFrame):
        y = df["stroke"]
        X = df.drop(columns=["stroke", "id"])
        return X, y

    # def to_numpy(self):
    #     if self.df is None:
    #         logger.warning("No data loaded!")
    #         return

    #     self.numpy_data = self.df.to_numpy()
