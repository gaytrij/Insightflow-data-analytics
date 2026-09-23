import pandas as pd

from src.logger import get_logger


logger = get_logger(__name__)


def load_csv(file):
    """
    Load a CSV file into a pandas DataFrame.
    """

    logger.info("Attempting to load CSV dataset.")

    try:
        df = pd.read_csv(file)

        logger.info(
            "Dataset loaded successfully: %s rows, %s columns.",
            df.shape[0],
            df.shape[1]
        )

        return df

    except FileNotFoundError:

        logger.error(
            "CSV file was not found: %s",
            file
        )

        raise ValueError(
            f"CSV file was not found: {file}"
        )

    except pd.errors.EmptyDataError:

        logger.error(
            "CSV file is empty: %s",
            file
        )

        raise ValueError(
            "The CSV file is empty."
        )

    except Exception as e:

        logger.exception(
            "Unable to load CSV file."
        )

        raise ValueError(
            f"Unable to load CSV file: {e}"
        )


def get_dataset_profile(df):
    """
    Generate a basic profile of the dataset.
    """

    logger.info(
        "Generating dataset profile for %s rows and %s columns.",
        df.shape[0],
        df.shape[1]
    )

    profile = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(
            df.isnull().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "numerical_columns": (
            df.select_dtypes(
                include="number"
            ).columns.tolist()
        ),
        "categorical_columns": (
            df.select_dtypes(
                include="object"
            ).columns.tolist()
        ),
    }

    logger.info(
        "Dataset profile generated successfully."
    )

    return profile