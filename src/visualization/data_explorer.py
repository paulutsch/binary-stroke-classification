import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger


class DataExplorer:
    def __init__(self):
        pass

    def plot_distributions(self, df: pd.DataFrame, cols_per_row=5):
        df = df.drop(columns=["id"])
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
                        hue=df["stroke"],
                        palette="Set2",
                    )
                else:
                    sns.histplot(df[column], ax=axes[i], kde=True, color="skyblue")
            else:
                sns.countplot(
                    x=df[column],
                    ax=axes[i],
                    hue=df["stroke"],
                    palette="Set2",
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

    def plot_nans(self, df: pd.DataFrame):
        n = len(df)
        nan_counts = []
        for col in df.columns:
            n_nan_in_col = df[col].isnull().sum() + sum(df[col] == "Unknown")
            nan_counts.append((col, n_nan_in_col))
            logger.info(f"Number of NaN values in column {col}: {n_nan_in_col} / {n}")

        nan_df = pd.DataFrame(nan_counts, columns=["Column", "NaN Count"])

        plt.figure(figsize=(12, 8))
        sns.barplot(y="NaN Count", x="Column", data=nan_df, palette="viridis")
        plt.title("Number of NaN/Unknown Values Per Column", fontsize=16)
        plt.ylabel("NaN/Unknown Count", fontsize=14)
        plt.xlabel("Columns", fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim((0, len(df)))
        plt.grid(axis="y", linestyle="--", linewidth=0.7)
        plt.show()

    def print_non_numericals(self, df: pd.DataFrame):
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                unique_values = df[column].unique()
                logger.info(f"unique values in {column}: {unique_values}")

    def plot_correlation_matrix(self, df: pd.DataFrame):
        corr_matrix = df.corr(method="pearson")

        # Adjust figure size based on the number of columns
        num_columns = len(corr_matrix.columns)
        fig_width = max(12, num_columns)  # Ensure a minimum width of 12
        plt.figure(figsize=(fig_width, 10))

        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")

        # Set rotation of x and y axis labels
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.title("Correlation Matrix")
        plt.tight_layout()  # Adjusts the padding to make sure everything fits
        plt.show()
