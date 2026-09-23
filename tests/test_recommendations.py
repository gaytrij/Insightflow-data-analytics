import pandas as pd

from src.recommendations import generate_recommendations


def test_recommendations_empty_dataset():

    df = pd.DataFrame(
        columns=["Age", "Income"]
    )

    recommendations = generate_recommendations(df)

    assert isinstance(recommendations, list)
    assert len(recommendations) == 1

    assert "no records" in recommendations[0].lower()


def test_recommendations_complete_dataset():

    df = pd.DataFrame({
        "Age": [22, 30, 40, 50],
        "Income": [25000, 40000, 60000, 80000],
        "Category": ["A", "A", "B", "B"]
    })

    recommendations = generate_recommendations(df)

    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

    recommendation_text = " ".join(
        recommendations
    ).lower()

    assert "no missing values" in recommendation_text
    assert "no duplicate records" in recommendation_text


def test_recommendations_missing_values():

    df = pd.DataFrame({
        "Age": [22, None, 40, 50],
        "Income": [25000, 40000, None, 80000]
    })

    recommendations = generate_recommendations(df)

    recommendation_text = " ".join(
        recommendations
    ).lower()

    assert "missing values" in recommendation_text
    assert "age" in recommendation_text
    assert "income" in recommendation_text


def test_recommendations_duplicates():

    df = pd.DataFrame({
        "Age": [22, 30, 30],
        "Income": [25000, 50000, 50000]
    })

    recommendations = generate_recommendations(df)

    recommendation_text = " ".join(
        recommendations
    ).lower()

    assert "duplicate" in recommendation_text


def test_recommendations_anomalies():

    df = pd.DataFrame({
        "Age": [22, 30, 40, 50, 60],
        "Income": [25000, 40000, 60000, 80000, 100000]
    })

    anomaly_df = df.copy()

    anomaly_df["Anomaly"] = [
        1,
        1,
        1,
        1,
        -1
    ]

    recommendations = generate_recommendations(
        df,
        anomaly_df=anomaly_df
    )

    recommendation_text = " ".join(
        recommendations
    ).lower()

    assert "anomal" in recommendation_text


def test_recommendations_model_results():

    df = pd.DataFrame({
        "Age": [22, 30, 40, 50],
        "Income": [25000, 40000, 60000, 80000]
    })

    model_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],
        "Accuracy": [
            0.80,
            0.90
        ],
        "F1 Score": [
            0.78,
            0.88
        ]
    })

    recommendations = generate_recommendations(
        df,
        model_results=model_results
    )

    recommendation_text = " ".join(
        recommendations
    ).lower()

    assert "f1 score" in recommendation_text
    assert "random forest" in recommendation_text


def test_recommendations_feature_importance():

    df = pd.DataFrame({
        "Age": [22, 30, 40, 50],
        "Income": [25000, 40000, 60000, 80000]
    })

    feature_importance = {
        "Random Forest": pd.DataFrame({
            "Feature": [
                "Income",
                "Age"
            ],
            "Importance": [
                0.75,
                0.25
            ]
        })
    }

    recommendations = generate_recommendations(
        df,
        feature_importance=feature_importance
    )

    recommendation_text = " ".join(
        recommendations
    ).lower()

    assert "random forest" in recommendation_text
    assert "income" in recommendation_text