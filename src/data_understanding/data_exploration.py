import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger


def plot_distributions(df: pd.DataFrame):
    cols_per_row = 5
    num_columns = len(df.columns)
    num_rows = (num_columns + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(
        num_rows, cols_per_row, figsize=(5 * cols_per_row, 5 * num_rows)
    )
    axes = axes.flatten()

    for i, column in enumerate(df.columns):
        if pd.api.types.is_numeric_dtype(df[column]):
            unique_values = df[column].unique()
            if len(unique_values) <= 2:
                sns.countplot(
                    x=df[column],
                    ax=axes[i],
                    hue=df["stroke"] if "stroke" in df.columns else None,
                    palette="Set2" if "stroke" in df.columns else None,
                )
            else:
                sns.histplot(
                    df,
                    x=column,
                    hue="stroke" if "stroke" in df.columns else None,
                    ax=axes[i],
                    kde=True,
                    palette="Set2" if "stroke" in df.columns else None,
                    multiple="stack",
                )
        else:
            sns.countplot(
                x=df[column],
                ax=axes[i],
                hue=df["stroke"] if "stroke" in df.columns else None,
                palette="Set2" if "stroke" in df.columns else None,
            )

        axes[i].set_title(f"Distribution of {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Frequency")

        if not pd.api.types.is_numeric_dtype(df[column]):
            for tick in axes[i].get_xticklabels():
                tick.set_rotation(45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(pad=3.0)
    plt.show()


def plot_relative_distributions(df: pd.DataFrame):
    cols_per_row = 5
    num_columns = len(df.columns)
    num_rows = (num_columns + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(
        num_rows, cols_per_row, figsize=(5 * cols_per_row, 5 * num_rows)
    )
    axes = axes.flatten()

    for i, column in enumerate(df.columns):
        if pd.api.types.is_numeric_dtype(df[column]):
            if len(df[column].unique()) > 2:
                data = df.copy()
                range_min, range_max = data[column].min(), data[column].max()
                bin_edges = pd.cut(data[column], bins=10, retbins=True)[1]
                data["bin"] = pd.cut(data[column], bins=bin_edges, include_lowest=True)
                bin_data = (
                    data.groupby("bin", observed=True)["stroke"]
                    .value_counts(normalize=True)
                    .unstack()
                    .fillna(0)
                )
                bin_data = bin_data[[1, 0]]
                bin_data.plot(
                    kind="bar", stacked=True, ax=axes[i], color=["orange", "lightgray"]
                )
            else:
                prop_data = (
                    df.groupby([column, "stroke"], observed=True)
                    .size()
                    .groupby(level=0)
                    .apply(lambda x: x / x.sum())
                    .unstack()
                    .fillna(0)
                )
                prop_data = prop_data[[1, 0]]
                prop_data.plot(
                    kind="bar", stacked=True, ax=axes[i], color=["orange", "lightgray"]
                )
        else:
            prop_data = (
                df.groupby(column, observed=True)["stroke"]
                .value_counts(normalize=True)
                .unstack()
                .fillna(0)
            )
            prop_data = prop_data[[1, 0]]
            prop_data.plot(
                kind="bar", stacked=True, ax=axes[i], color=["orange", "lightgray"]
            )

        axes[i].set_title(f"Relative Distribution of {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Proportion")

        if not pd.api.types.is_numeric_dtype(df[column]):
            for tick in axes[i].get_xticklabels():
                tick.set_rotation(45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(pad=3.0)
    plt.show()


def print_na(df: pd.DataFrame):
    n = len(df)
    nan_counts = []
    for col in df.columns:
        n_nan_in_col = df[col].isnull().sum() + sum(df[col] == "Unknown")
        nan_counts.append((col, n_nan_in_col))
        logger.info(f"Number of NaN values in column {col}: {n_nan_in_col} / {n}")

    nan_df = pd.DataFrame(nan_counts, columns=["Column", "NaN Count"])
    total_nans_per_row = df.isnull().any(axis=1) | (df == "Unknown").any(axis=1)
    total_nans_count = total_nans_per_row.sum()
    logger.info(f"Number of data points with NaN values: {total_nans_count} / {n}")


def print_non_numericals(df: pd.DataFrame):
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            unique_values = df[column].unique()
            logger.info(f"unique values in {column}: {unique_values}")
