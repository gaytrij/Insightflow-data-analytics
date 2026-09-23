import pandas as pd

from src.logger import get_logger


logger = get_logger(__name__)


def get_data_quality_report(df):
    """
    Generate a data quality report.
    """

    logger.info(
        "Generating data quality report."
    )

    report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Missing %": (
            df.isnull().mean().values * 100
        ).round(2),
        "Unique Values": df.nunique().values
    })

    logger.info(
        "Data quality report generated successfully."
    )

    return report


def remove_duplicates(df):
    """
    Remove exact duplicate rows.
    """

    original_rows = len(df)

    cleaned_df = (
        df.drop_duplicates()
        .reset_index(drop=True)
    )

    removed_rows = (
        original_rows - len(cleaned_df)
    )

    if removed_rows > 0:

        logger.warning(
            "Removed %s duplicate rows.",
            removed_rows
        )

    else:

        logger.info(
            "No duplicate rows detected."
        )

    return cleaned_df


def handle_missing_values(df):
    """
    Handle missing numerical values using median
    and categorical values using mode.
    """

    cleaned_df = df.copy()

    numerical_columns = (
        cleaned_df.select_dtypes(
            include="number"
        ).columns
    )

    categorical_columns = (
        cleaned_df.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    missing_before = int(
        cleaned_df.isnull().sum().sum()
    )

    if missing_before > 0:

        logger.warning(
            "Detected %s missing values. "
            "Starting imputation.",
            missing_before
        )

    for column in numerical_columns:

        if cleaned_df[column].isnull().any():

            median = cleaned_df[column].median()

            cleaned_df[column] = (
                cleaned_df[column].fillna(median)
            )

            logger.info(
                "Filled missing numerical values "
                "in column '%s' using median.",
                column
            )

    for column in categorical_columns:

        if cleaned_df[column].isnull().any():

            mode = cleaned_df[column].mode()

            if not mode.empty:

                cleaned_df[column] = (
                    cleaned_df[column].fillna(mode[0])
                )

                logger.info(
                    "Filled missing categorical values "
                    "in column '%s' using mode.",
                    column
                )

    missing_after = int(
        cleaned_df.isnull().sum().sum()
    )

    logger.info(
        "Missing-value handling completed. "
        "Remaining missing values: %s.",
        missing_after
    )

    return cleaned_df


def clean_dataset(df):
    """
    Execute the complete dataset cleaning pipeline.
    """

    logger.info(
        "Starting dataset cleaning pipeline."
    )

    cleaned_df = remove_duplicates(df)

    cleaned_df = handle_missing_values(
        cleaned_df
    )

    logger.info(
        "Dataset cleaning pipeline completed. "
        "Final rows: %s.",
        len(cleaned_df)
    )

    return cleaned_df