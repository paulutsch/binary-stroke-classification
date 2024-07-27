from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def load_data(
    path: str = "data/raw/train.csv",
):
    df = pd.read_csv(path)
    logger.success("Successfully created Pandas Dataframe from raw data.\n")

    return df


def split_X_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df.drop(columns=["stroke"]).to_numpy()
    y = df["stroke"].to_numpy()
    return X, y
