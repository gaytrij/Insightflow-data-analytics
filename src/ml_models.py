import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest
)

from sklearn.cluster import KMeans

from src.logger import get_logger


logger = get_logger(__name__)


def prepare_classification_data(df, target_column):

    logger.info(
        "Preparing classification data. Target: %s",
        target_column
    )

    data = df.copy()

    X = data.select_dtypes(
        include="number"
    ).drop(
        columns=[target_column],
        errors="ignore"
    )

    y = data[target_column]

    return X, y


def calculate_classification_metrics(
    y_test,
    predictions,
    probabilities=None
):

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    if probabilities is not None:

        try:

            if probabilities.shape[1] == 2:

                metrics["ROC-AUC"] = roc_auc_score(
                    y_test,
                    probabilities[:, 1]
                )

            else:

                metrics["ROC-AUC"] = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted"
                )

        except Exception:

            logger.warning(
                "ROC-AUC could not be calculated."
            )

    return metrics


def train_classification_model(
    df,
    target_column
):

    logger.info(
        "Starting classification model training. "
        "Target: %s",
        target_column
    )

    X, y = prepare_classification_data(
        df,
        target_column
    )

    if y.nunique() < 2:

        logger.error(
            "Classification target contains fewer than two classes."
        )

        raise ValueError(
            "Classification target must contain at least two classes."
        )

    if y.nunique() > 10:

        logger.error(
            "Classification target contains more than 10 classes."
        )

        raise ValueError(
            "Classification target has more than 10 unique values."
        )

    if X.empty:

        logger.error(
            "No numerical features available for classification."
        )

        raise ValueError(
            "No numerical features available for classification."
        )

    class_counts = y.value_counts()

    if class_counts.min() < 2:

        logger.error(
            "One or more classification classes contain fewer than 2 records."
        )

        raise ValueError(
            "Each class must contain at least 2 records "
            "for train/test splitting."
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )

    metrics = calculate_classification_metrics(
        y_test,
        predictions,
        probabilities
    )

    logger.info(
        "Classification model training completed. "
        "Accuracy: %.4f, F1: %.4f.",
        metrics["Accuracy"],
        metrics["F1 Score"]
    )

    return (
        model,
        scaler,
        metrics,
        y_test,
        predictions
    )


def compare_classification_models(
    df,
    target_column
):

    logger.info(
        "Starting classification model comparison. "
        "Target: %s",
        target_column
    )

    X, y = prepare_classification_data(
        df,
        target_column
    )

    if y.nunique() < 2:

        raise ValueError(
            "Classification target must contain at least two classes."
        )

    if y.nunique() > 10:

        raise ValueError(
            "Classification target has more than 10 unique values."
        )

    if X.empty:

        raise ValueError(
            "No numerical features available for classification."
        )

    class_counts = y.value_counts()

    if class_counts.min() < 2:

        raise ValueError(
            "Each class must contain at least 2 records."
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000
            ),
            True
        ),

        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ),
            False
        ),

        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            False
        )
    }

    results = []

    trained_models = {}

    predictions_dict = {}

    feature_importance = {}

    for model_name, (
        model,
        needs_scaling
    ) in models.items():

        logger.info(
            "Training %s.",
            model_name
        )

        if needs_scaling:

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

            probabilities = model.predict_proba(
                X_test_scaled
            )

        else:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            probabilities = model.predict_proba(
                X_test
            )

        metrics = calculate_classification_metrics(
            y_test,
            predictions,
            probabilities
        )

        row = {
            "Model": model_name,
            **metrics
        }

        results.append(row)

        trained_models[
            model_name
        ] = model

        predictions_dict[
            model_name
        ] = predictions

        if hasattr(
            model,
            "feature_importances_"
        ):

            feature_importance[
                model_name
            ] = pd.DataFrame({

                "Feature": X.columns,

                "Importance":
                    model.feature_importances_

            }).sort_values(
                "Importance",
                ascending=False
            )

    comparison_df = pd.DataFrame(
        results
    )

    logger.info(
        "Classification model comparison completed. "
        "Evaluated %s models.",
        len(results)
    )

    return (
        comparison_df,
        trained_models,
        predictions_dict,
        feature_importance,
        y_test,
        X.columns.tolist()
    )


def train_regression_model(
    df,
    target_column
):

    logger.info(
        "Starting regression model training. "
        "Target: %s",
        target_column
    )

    data = df.copy()

    X = data.select_dtypes(
        include="number"
    ).drop(
        columns=[target_column],
        errors="ignore"
    )

    y = data[target_column]

    if X.empty:

        raise ValueError(
            "No numerical features available for regression."
        )

    if y.nunique() <= 1:

        raise ValueError(
            "Regression target must contain "
            "more than one unique value."
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    metrics = {
        "MAE": mean_absolute_error(
            y_test,
            predictions
        ),

        "MSE": mse,

        "RMSE": mse ** 0.5,

        "R2": r2_score(
            y_test,
            predictions
        )
    }

    logger.info(
        "Regression model training completed. "
        "R2: %.4f, RMSE: %.4f.",
        metrics["R2"],
        metrics["RMSE"]
    )

    return (
        model,
        metrics,
        y_test,
        predictions
    )


def perform_clustering(
    df,
    n_clusters=3
):

    logger.info(
        "Starting K-Means clustering with %s clusters.",
        n_clusters
    )

    numerical_df = df.select_dtypes(
        include="number"
    )

    if numerical_df.empty:

        raise ValueError(
            "No numerical features available for clustering."
        )

    if len(numerical_df) < n_clusters:

        raise ValueError(
            "Number of clusters cannot be greater "
            "than the number of records."
        )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numerical_df
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(
        scaled_data
    )

    result = df.copy()

    result["Cluster"] = clusters

    logger.info(
        "K-Means clustering completed successfully."
    )

    return (
        model,
        result,
        scaled_data,
        numerical_df.columns.tolist()
    )


def detect_anomalies(
    df,
    contamination=0.05
):

    logger.info(
        "Starting Isolation Forest anomaly detection."
    )

    data = df.copy()

    numerical_df = data.select_dtypes(
        include="number"
    ).copy()

    if numerical_df.empty:

        raise ValueError(
            "No numerical columns available "
            "for anomaly detection."
        )

    if len(numerical_df) < 5:

        raise ValueError(
            "At least 5 records are required "
            "for anomaly detection."
        )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numerical_df
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(
        scaled_data
    )

    anomaly_count = int(
        (predictions == -1).sum()
    )

    logger.info(
        "Isolation Forest completed. "
        "Detected %s anomalies out of %s records.",
        anomaly_count,
        len(predictions)
    )

    anomaly_scores = model.decision_function(
        scaled_data
    )

    result = data.copy()

    result["Anomaly"] = predictions

    result["Anomaly Label"] = (
        result["Anomaly"].map({
            1: "Normal",
            -1: "Anomaly"
        })
    )

    result["Anomaly Score"] = (
        anomaly_scores
    )

    return (
        model,
        result,
        scaled_data,
        numerical_df.columns.tolist()
    )