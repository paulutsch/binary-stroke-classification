from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    def __init__(self, df_train: pd.DataFrame, threshold: float = 0.3):
        self.corr_matrix = df_train.corr(method="pearson")

    def drop(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df_new = df.copy()
        df_new.drop(columns=cols, inplace=True)
        return df_new

    def dropna(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        df_new.dropna(inplace=True)
        return df_new

    def split_X_y(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df.drop(columns=["stroke"]).to_numpy()
        y = df["stroke"].to_numpy()
        return X, y

    def combine_correlating_features(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()

        while True:
            # Get the upper triangle of the correlation matrix
            upper_triangle = self.corr_matrix.where(
                np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool)
            )

            # Find the most correlated pair
            if upper_triangle.max().max() < self.threshold:
                break

            most_correlated_pair = upper_triangle.stack().idxmax()
            feature1, feature2 = most_correlated_pair
            new_feature_name = f"{feature1}_{feature2}_combined"

            # Store the transformation
            self.combined_features.append((feature1, feature2, new_feature_name))

            # Combine the features into a new feature (example: taking their average)
            new_df[new_feature_name] = new_df[feature1] + new_df[feature2]

            # Drop the original features
            new_df = new_df.drop(columns=[feature1, feature2])

            # Update the correlation matrix
            self.corr_matrix = new_df.corr(method="pearson")

        return new_df

    def normalize_features(
        self,
        df: pd.DataFrame,
        range_min: float = 0.0,
        range_max: float = 1.0,
    ) -> pd.DataFrame:

        new_df = df.copy()

        for col in df.columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in DataFrame.")

            min_val = df[col].min()
            max_val = df[col].max()

            if min_val == max_val:
                raise ValueError(f"Column {col} has constant values, cannot normalize.")

            # Min-max normalization
            new_df[col] = (df[col] - min_val) / (max_val - min_val) * (
                range_max - range_min
            ) + range_min

        return new_df

    def combine_features(
        self,
        df: pd.DataFrame,
        cols: List[List[str]],
        transformation_style: str = "multiplicative",
    ) -> pd.DataFrame:

        new_df = df.copy()

        for col_pair in cols:
            if len(col_pair) != 2:
                raise ValueError(
                    "Each element in 'cols' should be a list of exactly two column names."
                )

            col1, col2 = col_pair

            if col1 not in df.columns or col2 not in df.columns:
                raise ValueError(
                    f"One or both columns {col1}, {col2} not found in DataFrame."
                )

            new_col_name = f"{col1}_x_{col2}"

            if transformation_style == "multiplicative":
                new_df[new_col_name] = df[col1] * df[col2]
            else:
                raise ValueError(
                    f"Transformation style '{transformation_style}' is not supported."
                )

        return new_df

    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()

        for feature1, feature2, new_feature_name in self.combined_features:
            if feature1 in new_df.columns and feature2 in new_df.columns:
                new_df[new_feature_name] = new_df[feature1] + new_df[feature2]
                new_df = new_df.drop(columns=[feature1, feature2])

        return new_df
