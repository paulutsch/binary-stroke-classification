import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.decomposition import PCA


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


# def plot_correlation_matrix(df: pd.DataFrame):
#     corr_matrix = df.corr(
#         method="spearman"
#     )  # spearman correlation for non-linear relationships

#     # Adjust figure size based on the number of columns
#     num_columns = len(corr_matrix.columns)
#     fig_width = max(12, num_columns)  # Ensure a minimum width of 12
#     plt.figure(figsize=(fig_width, 10))

#     sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")

#     # Set rotation of x and y axis labels
#     plt.xticks(rotation=45, ha="right")
#     plt.yticks(rotation=0)

#     plt.title("Correlation Matrix")
#     plt.tight_layout()  # Adjusts the padding to make sure everything fits
#     plt.show()


def pca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    # Fit PCA on the data
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    principal_df = pd.DataFrame(
        data=principal_components,
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
    )

    return principal_df


def plot_scatter_and_pair(df: pd.DataFrame):
    target_column = df["stroke"]
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(18, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_columns):
        sns.scatterplot(x=df[col], y=target_column, ax=axes[i])
        axes[i].set_title(f"{col} vs 'stroke'")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("stroke")

    plt.tight_layout()
    plt.show()

    if len(numeric_columns) > 1:
        sns.pairplot(df, hue="stroke", vars=numeric_columns.drop("stroke"))
        plt.suptitle("Pair Plot", y=1.02)
        plt.show()
