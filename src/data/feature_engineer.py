from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    def __init__(self, df_train: pd.DataFrame, threshold: float = 0.3):
        self.corr_matrix = df_train.corr(method="pearson")
        self.threshold = threshold
        self.combined_features = []

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

    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()

        for feature1, feature2, new_feature_name in self.combined_features:
            if feature1 in new_df.columns and feature2 in new_df.columns:
                new_df[new_feature_name] = new_df[feature1] + new_df[feature2]
                new_df = new_df.drop(columns=[feature1, feature2])

        return new_df
