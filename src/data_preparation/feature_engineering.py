from itertools import combinations_with_replacement
from typing import List, Tuple

import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA


def create_polynomials(
    df: pd.DataFrame, columns: List[str], degree: int = 2
) -> Tuple[pd.DataFrame, List[str]]:
    new_columns = {}
    feature_names = []

    for deg in range(2, degree + 1):
        for combination in combinations_with_replacement(columns, deg):
            new_col_name = "^".join(combination)
            new_columns[new_col_name] = df[list(combination)].prod(axis=1)
            feature_names.append(new_col_name)

    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    return new_df, feature_names


# transform the test set using the stored feature names
def transform_polynomials(
    df: pd.DataFrame, columns: List[str], feature_names: List[str]
) -> pd.DataFrame:
    new_columns = {}

    for new_col_name in feature_names:
        combination = new_col_name.split("^")
        new_columns[new_col_name] = df[list(combination)].prod(axis=1)

    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    return new_df


def pca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    principal_df = pd.DataFrame(
        data=principal_components,
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
    )

    return principal_df
