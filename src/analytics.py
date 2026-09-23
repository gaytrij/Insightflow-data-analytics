import pandas as pd

from src.logger import get_logger


logger = get_logger(__name__)


def generate_summary_statistics(df):
    """
    Generate descriptive statistics for numerical columns.
    """

    logger.info(
        "Generating summary statistics."
    )

    numerical_df = df.select_dtypes(
        include="number"
    )

    if numerical_df.empty:

        logger.warning(
            "No numerical columns available "
            "for summary statistics."
        )

        return pd.DataFrame()

    result = numerical_df.describe().T

    logger.info(
        "Summary statistics generated successfully."
    )

    return result


def generate_correlation_matrix(df):
    """
    Generate correlation matrix for numerical columns.
    """

    logger.info(
        "Generating correlation matrix."
    )

    numerical_df = df.select_dtypes(
        include="number"
    )

    if numerical_df.empty:

        logger.warning(
            "No numerical columns available "
            "for correlation analysis."
        )

        return pd.DataFrame()

    result = numerical_df.corr()

    logger.info(
        "Correlation matrix generated successfully."
    )

    return result


def generate_categorical_summary(df):
    """
    Generate frequency summaries for categorical columns.
    """

    logger.info(
        "Generating categorical summary."
    )

    categorical_columns = (
        df.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    summaries = {}

    for column in categorical_columns:

        summaries[column] = (
            df[column]
            .value_counts(dropna=False)
            .to_dict()
        )

    logger.info(
        "Categorical summary generated for %s columns.",
        len(categorical_columns)
    )

    return summaries


def generate_kpis(df):
    """
    Generate key dataset-level metrics.
    """

    logger.info(
        "Generating dataset KPIs."
    )

    numerical_df = df.select_dtypes(
        include="number"
    )

    kpis = {
        "Total Records": len(df),
        "Total Columns": len(df.columns),
        "Numerical Columns": len(
            df.select_dtypes(
                include="number"
            ).columns
        ),
        "Categorical Columns": len(
            df.select_dtypes(
                include=["object", "category"]
            ).columns
        ),
        "Missing Values": int(
            df.isnull().sum().sum()
        ),
        "Duplicate Rows": int(
            df.duplicated().sum()
        )
    }

    if not numerical_df.empty:

        kpis["Average Numerical Value"] = round(
            numerical_df.mean().mean(),
            2
        )

    logger.info(
        "Generated %s KPIs.",
        len(kpis)
    )

    return kpis