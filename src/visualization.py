import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import get_logger


logger = get_logger(__name__)


def plot_numeric_distribution(df, column):
    """
    Plot the distribution of a numerical column.
    """

    logger.info(
        "Creating numeric distribution plot for '%s'.",
        column
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {column}"
    )

    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    return fig


def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap for numerical columns.
    """

    logger.info(
        "Creating correlation heatmap."
    )

    numerical_df = df.select_dtypes(
        include="number"
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    sns.heatmap(
        numerical_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    return fig


def plot_categorical_distribution(df, column):
    """
    Plot the distribution of a categorical column.
    """

    logger.info(
        "Creating categorical distribution plot for '%s'.",
        column
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    value_counts = (
        df[column]
        .value_counts()
        .head(15)
    )

    value_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        f"Distribution of {column}"
    )

    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    return fig


def plot_scatter(df, x_column, y_column):
    """
    Plot a scatter chart between two numerical columns.
    """

    logger.info(
        "Creating scatter plot: %s vs %s.",
        x_column,
        y_column
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column,
        ax=ax
    )

    ax.set_title(
        f"{x_column} vs {y_column}"
    )

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    plt.tight_layout()

    return fig