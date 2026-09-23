import pandas as pd

from src.logger import get_logger


logger = get_logger(__name__)


def generate_recommendations(
    df,
    anomaly_df=None,
    model_results=None,
    feature_importance=None
):

    logger.info(
        "Generating automated recommendations for %s records.",
        len(df)
    )

    recommendations = []

    total_rows = len(df)

    if total_rows == 0:

        logger.warning(
            "Recommendation engine received an empty dataset."
        )

        return [
            "The dataset contains no records. "
            "Upload a dataset containing valid records."
        ]

    missing = df.isnull().sum()

    missing_columns = (
        missing[
            missing > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if not missing_columns.empty:

        for column, count in (
            missing_columns.items()
        ):

            percentage = (
                count /
                total_rows *
                100
            )

            recommendations.append(
                f"Review missing values in '{column}'. "
                f"{count:,} records ({percentage:.2f}%) "
                "are missing. Apply an appropriate "
                "imputation or data-collection strategy."
            )

    else:

        recommendations.append(
            "No missing values were detected. "
            "The dataset is complete with respect "
            "to null-value checks."
        )

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        recommendations.append(
            f"Review {duplicate_count:,} duplicate records "
            "before downstream analysis to avoid "
            "potentially duplicated observations."
        )

    else:

        recommendations.append(
            "No duplicate records were detected."
        )

    numerical_columns = (
        df.select_dtypes(
            include="number"
        ).columns.tolist()
    )

    if len(numerical_columns) >= 2:

        correlation = (
            df[numerical_columns]
            .corr()
            .abs()
        )

        strong_relationships = []

        for i in range(
            len(correlation.columns)
        ):

            for j in range(
                i + 1,
                len(correlation.columns)
            ):

                value = correlation.iloc[
                    i,
                    j
                ]

                if value >= 0.80:

                    strong_relationships.append(
                        (
                            correlation.columns[i],
                            correlation.columns[j],
                            value
                        )
                    )

        if strong_relationships:

            for (
                column1,
                column2,
                value
            ) in strong_relationships:

                recommendations.append(
                    f"Investigate the strong relationship "
                    f"between '{column1}' and '{column2}' "
                    f"(absolute correlation: {value:.2f}). "
                    "These variables may provide useful "
                    "predictive or segmentation signals."
                )

        else:

            recommendations.append(
                "No strong pairwise correlations above "
                "0.80 were detected among numerical variables."
            )

    categorical_columns = (
        df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    )

    for column in categorical_columns:

        unique_count = df[column].nunique()

        if unique_count > 50:

            recommendations.append(
                f"'{column}' contains "
                f"{unique_count:,} distinct categories. "
                "Review whether this column should be "
                "treated as an identifier, grouped into "
                "categories, or excluded from modeling."
            )

    if (
        anomaly_df is not None
        and "Anomaly" in anomaly_df.columns
    ):

        total_anomaly_records = len(
            anomaly_df
        )

        anomaly_count = int(
            (
                anomaly_df["Anomaly"] == -1
            ).sum()
        )

        if total_anomaly_records > 0:

            anomaly_percentage = (
                anomaly_count /
                total_anomaly_records *
                100
            )

            if anomaly_percentage >= 10:

                recommendations.append(
                    f"{anomaly_percentage:.2f}% of records "
                    "were flagged as anomalies. Review "
                    "these records for data-entry errors, "
                    "unusual business events, or legitimate "
                    "rare observations before removing them."
                )

            elif anomaly_percentage > 0:

                recommendations.append(
                    f"{anomaly_count:,} records were flagged "
                    f"as anomalies ({anomaly_percentage:.2f}%). "
                    "Review the detected records before using "
                    "them for sensitive modeling tasks."
                )

            else:

                recommendations.append(
                    "No records were flagged as anomalies "
                    "under the selected Isolation Forest settings."
                )

    if (
        model_results is not None
        and not model_results.empty
    ):

        required_columns = [
            "Model",
            "F1 Score"
        ]

        if all(
            column in model_results.columns
            for column in required_columns
        ):

            highest_f1 = (
                model_results["F1 Score"].max()
            )

            strongest_models = (
                model_results[
                    model_results["F1 Score"]
                    == highest_f1
                ]["Model"].tolist()
            )

            model_names = ", ".join(
                str(model)
                for model in strongest_models
            )

            recommendations.append(
                f"Among the evaluated classification "
                f"models, {model_names} recorded the "
                f"highest observed F1 Score of "
                f"{highest_f1:.4f} on the current "
                "test split. Validate model performance "
                "on a larger or independent dataset "
                "before production use."
            )

            if (
                "Accuracy" in model_results.columns
                and
                "F1 Score" in model_results.columns
            ):

                accuracy_range = (
                    model_results["Accuracy"].max()
                    -
                    model_results["Accuracy"].min()
                )

                if accuracy_range < 0.05:

                    recommendations.append(
                        "The evaluated classification models "
                        "have relatively similar accuracy. "
                        "Consider cross-validation and "
                        "additional feature engineering to "
                        "assess whether model performance "
                        "can be improved."
                    )

    if feature_importance:

        for (
            model_name,
            importance_df
        ) in feature_importance.items():

            if (
                importance_df is not None
                and not importance_df.empty
            ):

                top_features = (
                    importance_df
                    .head(3)["Feature"]
                    .tolist()
                )

                if top_features:

                    feature_text = ", ".join(
                        top_features
                    )

                    recommendations.append(
                        f"For {model_name}, the leading "
                        f"features by model-derived importance "
                        f"are: {feature_text}. Investigate these "
                        "variables further for domain relevance "
                        "and feature quality."
                    )

    if total_rows < 100:

        recommendations.append(
            f"The dataset contains {total_rows:,} records. "
            "Treat model metrics cautiously because a small "
            "dataset can produce unstable test-set estimates. "
            "Consider collecting more representative data."
        )

    elif total_rows < 1000:

        recommendations.append(
            f"The dataset contains {total_rows:,} records. "
            "Consider cross-validation and additional "
            "validation data before relying on model "
            "performance for production decisions."
        )

    else:

        recommendations.append(
            f"The dataset contains {total_rows:,} records, "
            "providing a larger basis for exploratory "
            "analysis and model validation."
        )

    logger.info(
        "Generated %s recommendations.",
        len(recommendations)
    )

    return recommendations