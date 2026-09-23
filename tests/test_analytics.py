import pandas as pd

from src.analytics import (
    generate_summary_statistics,
    generate_correlation_matrix,
    generate_categorical_summary,
    generate_kpis
)


def create_analytics_dataset():

    return pd.DataFrame({
        "Age": [22, 30, 40, 50],
        "Income": [25000, 40000, 60000, 80000],
        "Category": ["A", "A", "B", "B"]
    })


def test_generate_summary_statistics():

    df = create_analytics_dataset()

    summary = generate_summary_statistics(df)

    assert isinstance(summary, pd.DataFrame)

    assert "Age" in summary.index
    assert "Income" in summary.index

    assert "mean" in summary.columns
    assert "std" in summary.columns


def test_generate_correlation_matrix():

    df = create_analytics_dataset()

    correlation = generate_correlation_matrix(df)

    assert isinstance(correlation, pd.DataFrame)

    assert "Age" in correlation.columns
    assert "Income" in correlation.columns

    assert correlation.shape == (2, 2)

    assert correlation.loc["Age", "Age"] == 1.0


def test_generate_categorical_summary():

    df = create_analytics_dataset()

    summary = generate_categorical_summary(df)

    assert isinstance(summary, dict)

    assert "Category" in summary

    assert summary["Category"]["A"] == 2
    assert summary["Category"]["B"] == 2


def test_generate_kpis():

    df = create_analytics_dataset()

    kpis = generate_kpis(df)

    assert isinstance(kpis, dict)

    assert kpis["Total Records"] == 4
    assert kpis["Total Columns"] == 3

    assert kpis["Numerical Columns"] == 2
    assert kpis["Categorical Columns"] == 1

    assert kpis["Missing Values"] == 0
    assert kpis["Duplicate Rows"] == 0

    assert "Average Numerical Value" in kpis