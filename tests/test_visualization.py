import pandas as pd
import matplotlib.pyplot as plt

from src.visualization import (
    plot_numeric_distribution,
    plot_correlation_heatmap,
    plot_categorical_distribution,
    plot_scatter
)


def create_visualization_dataset():

    return pd.DataFrame({
        "Age": [22, 30, 40, 50],
        "Income": [25000, 40000, 60000, 80000],
        "Category": ["A", "A", "B", "B"]
    })


def test_plot_numeric_distribution():

    df = create_visualization_dataset()

    fig = plot_numeric_distribution(
        df,
        "Age"
    )

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1

    plt.close(fig)


def test_plot_correlation_heatmap():

    df = create_visualization_dataset()

    fig = plot_correlation_heatmap(df)

    assert isinstance(fig, plt.Figure)

    # Heatmap contains:
    # 1 main heatmap axis
    # 1 colorbar axis
    assert len(fig.axes) == 2

    plt.close(fig)


def test_plot_categorical_distribution():

    df = create_visualization_dataset()

    fig = plot_categorical_distribution(
        df,
        "Category"
    )

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1

    plt.close(fig)


def test_plot_scatter():

    df = create_visualization_dataset()

    fig = plot_scatter(
        df,
        "Age",
        "Income"
    )

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1

    plt.close(fig)