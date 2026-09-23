import pandas as pd

from src.ml_models import (
    train_regression_model,
    perform_clustering,
    detect_anomalies
)


def create_regression_dataset():

    return pd.DataFrame({
        "Age": [22, 25, 28, 30, 35, 38, 42, 45, 48, 52],
        "Income": [
            25000, 30000, 35000, 40000, 50000,
            55000, 65000, 70000, 80000, 90000
        ],
        "Tenure": [3, 5, 8, 10, 15, 20, 25, 30, 35, 40],
        "MonthlySpend": [
            599, 699, 799, 849, 999,
            1099, 1199, 1299, 1399, 1499
        ]
    })


def test_train_regression_model():

    df = create_regression_dataset()

    model, metrics, y_test, predictions = (
        train_regression_model(
            df,
            "MonthlySpend"
        )
    )

    assert model is not None

    assert isinstance(metrics, dict)

    assert "MAE" in metrics
    assert "MSE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics

    assert len(y_test) == len(predictions)


def test_perform_clustering():

    df = create_regression_dataset()

    model, result, scaled_data, feature_names = (
        perform_clustering(
            df,
            n_clusters=3
        )
    )

    assert model is not None

    assert "Cluster" in result.columns

    assert len(result) == len(df)

    assert scaled_data.shape[0] == len(df)

    assert len(feature_names) > 0

    assert result["Cluster"].nunique() == 3


def test_detect_anomalies():

    df = create_regression_dataset()

    model, result, scaled_data, feature_names = (
        detect_anomalies(
            df,
            contamination=0.1
        )
    )

    assert model is not None

    assert "Anomaly" in result.columns
    assert "Anomaly Label" in result.columns
    assert "Anomaly Score" in result.columns

    assert len(result) == len(df)

    assert scaled_data.shape[0] == len(df)

    assert len(feature_names) > 0

    assert set(result["Anomaly"].unique()).issubset({1, -1})