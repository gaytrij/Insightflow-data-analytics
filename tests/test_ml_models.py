import pandas as pd

from src.ml_models import (
    train_classification_model,
    compare_classification_models
)


def create_classification_dataset():
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
        ],
        "Churn": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    })


def test_train_classification_model():

    df = create_classification_dataset()

    model, scaler, metrics, y_test, predictions = (
        train_classification_model(
            df,
            "Churn"
        )
    )

    assert model is not None
    assert scaler is not None

    assert isinstance(metrics, dict)

    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1 Score" in metrics

    assert len(y_test) == len(predictions)


def test_compare_classification_models():

    df = create_classification_dataset()

    (
        comparison_df,
        trained_models,
        predictions_dict,
        feature_importance,
        y_test,
        feature_names
    ) = compare_classification_models(
        df,
        "Churn"
    )

    assert not comparison_df.empty

    assert "Model" in comparison_df.columns
    assert "Accuracy" in comparison_df.columns
    assert "Precision" in comparison_df.columns
    assert "Recall" in comparison_df.columns
    assert "F1 Score" in comparison_df.columns

    assert len(trained_models) == 3

    assert "Logistic Regression" in trained_models
    assert "Random Forest" in trained_models
    assert "Gradient Boosting" in trained_models

    assert len(predictions_dict) == 3

    assert len(feature_names) > 0