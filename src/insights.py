import pandas as pd


def generate_insights(df):

    insights = []

    # Dataset size
    rows = df.shape[0]
    columns = df.shape[1]

    insights.append(
        f"The dataset contains {rows:,} records "
        f"across {columns} columns."
    )

    # Missing values
    missing = df.isnull().sum()
    missing_columns = missing[missing > 0]

    if not missing_columns.empty:

        for column, count in missing_columns.items():

            percentage = (count / len(df)) * 100

            insights.append(
                f"'{column}' contains {count:,} missing values "
                f"({percentage:.2f}% of records)."
            )

    else:

        insights.append(
            "No missing values were detected in the dataset."
        )

    # Duplicate records
    duplicates = int(df.duplicated().sum())

    if duplicates > 0:

        insights.append(
            f"The dataset contains {duplicates:,} duplicate records "
            "that should be reviewed."
        )

    else:

        insights.append(
            "No duplicate records were detected."
        )

    # Numerical analysis
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numerical_columns:

        insights.append(
            f"{len(numerical_columns)} numerical columns "
            "are available for quantitative analysis."
        )

        # Correlation analysis
        if len(numerical_columns) >= 2:

            correlation = (
                df[numerical_columns]
                .corr()
                .abs()
            )

            high_correlations = []

            for i in range(len(correlation.columns)):

                for j in range(i + 1, len(correlation.columns)):

                    value = correlation.iloc[i, j]

                    if value >= 0.80:

                        high_correlations.append(
                            (
                                correlation.columns[i],
                                correlation.columns[j],
                                value
                            )
                        )

            if high_correlations:

                for col1, col2, value in high_correlations:

                    insights.append(
                        f"Strong correlation detected between "
                        f"'{col1}' and '{col2}' "
                        f"(correlation: {value:.2f})."
                    )

            else:

                insights.append(
                    "No strong correlations above 0.80 "
                    "were detected among numerical variables."
                )

    # Categorical analysis
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_columns:

        insights.append(
            f"{len(categorical_columns)} categorical columns "
            "are available for segmentation analysis."
        )

        for column in categorical_columns:

            unique_values = df[column].nunique()

            if unique_values <= 10:

                insights.append(
                    f"'{column}' contains "
                    f"{unique_values} distinct categories."
                )

    return insights