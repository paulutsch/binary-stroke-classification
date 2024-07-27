import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression


def replace_na(df: pd.DataFrame, replace_with: str = "mean"):
    new_df = df.copy()

    for column in new_df.columns:
        if new_df[column].isnull().any():
            if replace_with == "mean":
                replacement = df[column].mean()
                new_df[column] = new_df[column].fillna(replacement)
            elif replace_with == "median":
                replacement = df[column].median()
                new_df[column] = new_df[column].fillna(replacement)
            elif replace_with == "regression":
                # predict missing values based on the other values
                not_null_df = new_df[new_df.dropna()]
                null_df = new_df[new_df[column].isnull()]

                if not not_null_df.shape[0] > 1:
                    # not enough samples
                    continue

                X = not_null_df.drop(columns=[column])
                y = not_null_df[column]

                model = LinearRegression()
                model.fit(X, y)

                predicted_values = model.predict(null_df.drop(columns=[column]))
                new_df.loc[new_df[column].isnull(), column] = predicted_values

    logger.info(f"Replaced NaN values with {replace_with}")

    return new_df


def transform_non_numericals(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            unique_values = df[column].unique()
            if len(unique_values) == 2:
                # binary encoding for other binary categories
                df_encoded[column] = df[column].astype("category").cat.codes
                logger.info(f"Transformed {column} into binary encoding")
            else:
                # one-hot encoding for non-binary categories
                df_encoded = pd.get_dummies(df_encoded, columns=[column], dtype=int)
                logger.info(f"Transformed {column} into one-hot encoding")

    return df_encoded


def min_max_normalize(
    df: pd.DataFrame,
) -> pd.DataFrame:
    new_df = df.copy()

    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()

        new_df[col] = (df[col] - min_val) / (max_val - min_val)

    logger.info("Normalized data using min-max normalization")

    return new_df
