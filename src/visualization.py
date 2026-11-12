import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import get_logger


logger = get_logger(__name__)


def plot_numeric_distribution(df, column):
    """
    Plot a compact distribution chart for a numerical column.
    """

    logger.info(
        "Creating numeric distribution plot for '%s'.",
        column
    )

    fig, ax = plt.subplots(
        figsize=(4, 2.8)
    )

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {column}",
        fontsize=9
    )

    ax.set_xlabel(
        column,
        fontsize=8
    )
    ax.set_ylabel(
        "Frequency",
        fontsize=8
    )

    ax.tick_params(
        axis="both",
        labelsize=7
    )

    plt.tight_layout()

    return fig


def plot_correlation_heatmap(df):
    """
    Plot a compact correlation heatmap for numerical columns.
    """

    logger.info(
        "Creating correlation heatmap."
    )

    numerical_df = df.select_dtypes(
        include="number"
    )

    # Scale the heatmap to the number of numerical columns.
    number_of_columns = max(
        len(numerical_df.columns),
        2
    )

    figure_width = min(
        max(3.5, number_of_columns * 0.55),
        5.0
    )

    figure_height = min(
        max(2.8, number_of_columns * 0.45),
        4.0
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    sns.heatmap(
        numerical_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.75},
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap",
        fontsize=9
    )

    ax.tick_params(
        axis="both",
        labelsize=7
    )

    plt.tight_layout()

    return fig


def plot_categorical_distribution(df, column):
    """
    Plot a compact distribution chart for a categorical column.
    """

    logger.info(
        "Creating categorical distribution plot for '%s'.",
        column
    )

    fig, ax = plt.subplots(
        figsize=(4, 2.8)
    )

    value_counts = (
        df[column]
        .value_counts()
        .head(10)
    )

    value_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        f"Distribution of {column}",
        fontsize=9
    )

    ax.set_xlabel(
        column,
        fontsize=8
    )
    ax.set_ylabel(
        "Count",
        fontsize=8
    )

    ax.tick_params(
        axis="x",
        labelrotation=35,
        labelsize=7
    )
    ax.tick_params(
        axis="y",
        labelsize=7
    )

    plt.tight_layout()

    return fig


def plot_scatter(df, x_column, y_column):
    """
    Plot a compact scatter chart between two numerical columns.
    """

    logger.info(
        "Creating scatter plot: %s vs %s.",
        x_column,
        y_column
    )

    fig, ax = plt.subplots(
        figsize=(4, 2.8)
    )

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column,
        ax=ax
    )

    ax.set_title(
        f"{x_column} vs {y_column}",
        fontsize=9
    )

    ax.set_xlabel(
        x_column,
        fontsize=8
    )
    ax.set_ylabel(
        y_column,
        fontsize=8
    )

    ax.tick_params(
        axis="both",
        labelsize=7
    )

    plt.tight_layout()

    return fig
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import get_logger


logger = get_logger(__name__)


def plot_numeric_distribution(df, column):
    """
    Plot a compact distribution chart for a numerical column.
    """

    logger.info(
        "Creating numeric distribution plot for '%s'.",
        column
    )

    fig, ax = plt.subplots(
        figsize=(4, 2.8)
    )

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {column}",
        fontsize=9
    )

    ax.set_xlabel(
        column,
        fontsize=8
    )
    ax.set_ylabel(
        "Frequency",
        fontsize=8
    )

    ax.tick_params(
        axis="both",
        labelsize=7
    )

    plt.tight_layout()

    return fig


def plot_correlation_heatmap(df):
    """
    Plot a compact correlation heatmap for numerical columns.
    """

    logger.info(
        "Creating correlation heatmap."
    )

    numerical_df = df.select_dtypes(
        include="number"
    )

    # Scale the heatmap to the number of numerical columns.
    number_of_columns = max(
        len(numerical_df.columns),
        2
    )

    figure_width = min(
        max(3.5, number_of_columns * 0.55),
        5.0
    )

    figure_height = min(
        max(2.8, number_of_columns * 0.45),
        4.0
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    sns.heatmap(
        numerical_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.75},
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap",
        fontsize=9
    )

    ax.tick_params(
        axis="both",
        labelsize=7
    )

    plt.tight_layout()

    return fig


def plot_categorical_distribution(df, column):
    """
    Plot a compact distribution chart for a categorical column.
    """

    logger.info(
        "Creating categorical distribution plot for '%s'.",
        column
    )

    fig, ax = plt.subplots(
        figsize=(4, 2.8)
    )

    value_counts = (
        df[column]
        .value_counts()
        .head(10)
    )

    value_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        f"Distribution of {column}",
        fontsize=9
    )

    ax.set_xlabel(
        column,
        fontsize=8
    )
    ax.set_ylabel(
        "Count",
        fontsize=8
    )

    ax.tick_params(
        axis="x",
        labelrotation=35,
        labelsize=7
    )
    ax.tick_params(
        axis="y",
        labelsize=7
    )

    plt.tight_layout()

    return fig


def plot_scatter(df, x_column, y_column):
    """
    Plot a compact scatter chart between two numerical columns.
    """

    logger.info(
        "Creating scatter plot: %s vs %s.",
        x_column,
        y_column
    )

    fig, ax = plt.subplots(
        figsize=(4, 2.8)
    )

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column,
        ax=ax
    )

    ax.set_title(
        f"{x_column} vs {y_column}",
        fontsize=9
    )

    ax.set_xlabel(
        x_column,
        fontsize=8
    )
    ax.set_ylabel(
        y_column,
        fontsize=8
    )

    ax.tick_params(
        axis="both",
        labelsize=7
    )

    plt.tight_layout()

    return fig
