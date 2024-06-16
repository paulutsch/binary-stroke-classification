import pandas as pd
from loguru import logger


class DataPreprocessor:
    def __init__(self):
        pass

    def remove_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df.values)
        logger.info(f"Samples before removing NaN: {n_before}")
        df_new = df.dropna()
        n_after = len(df.values)
        logger.info(f"Samples after removing NaN: {n_after}")
        logger.success(f"Removed {n_before - n_after} samples with missing values.")
        return df_new

    def transform_non_numericals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                # if column == "smoking_status":
                #     val_to_int = {"never smoked": 0, "formerly smoked": 1, "smokes": 2}
                #     df_encoded[column] = df[column].map(val_to_int)
                # else:
                unique_values = df[column].unique()
                if len(unique_values) == 2:
                    # binary encoding
                    df_encoded[column] = df[column].astype("category").cat.codes
                    logger.info(f"Transformed {column} into binary encoding")
                else:
                    # one-hot encoding for non-binary categories
                    df_encoded = pd.get_dummies(df_encoded, columns=[column], dtype=int)
                    logger.info(f"Transformed {column} into one-hot encoding")

        return df_encoded

    def get_processed_data(self):
        return self._processed_data
