import pandas as pd

from src.data_cleaner import (
    get_data_quality_report,
    remove_duplicates,
    handle_missing_values,
    clean_dataset
)


def create_messy_dataset():

    return pd.DataFrame({
        "Age": [22, None, 30, 30],
        "Income": [25000, 40000, None, 50000],
        "Category": ["A", "B", None, "A"]
    })


def create_duplicate_dataset():

    return pd.DataFrame({
        "Age": [22, 30, 30],
        "Income": [25000, 50000, 50000],
        "Category": ["A", "B", "B"]
    })


def test_get_data_quality_report():

    df = create_messy_dataset()

    report = get_data_quality_report(df)

    assert isinstance(report, pd.DataFrame)

    assert list(report.columns) == [
        "Column",
        "Data Type",
        "Missing Values",
        "Missing %",
        "Unique Values"
    ]

    assert len(report) == 3

    age_missing = report.loc[
        report["Column"] == "Age",
        "Missing Values"
    ].iloc[0]

    assert age_missing == 1


def test_remove_duplicates():

    df = create_duplicate_dataset()

    cleaned_df = remove_duplicates(df)

    assert len(cleaned_df) == 2
    assert cleaned_df.duplicated().sum() == 0


def test_handle_missing_values():

    df = create_messy_dataset()

    cleaned_df = handle_missing_values(df)

    assert cleaned_df.isnull().sum().sum() == 0


def test_clean_dataset():

    df = pd.concat(
        [
            create_messy_dataset(),
            pd.DataFrame({
                "Age": [30],
                "Income": [50000],
                "Category": ["A"]
            })
        ],
        ignore_index=True
    )

    # Add an exact duplicate of the final row
    df = pd.concat(
        [df, df.iloc[[4]]],
        ignore_index=True
    )

    cleaned_df = clean_dataset(df)

    # Missing values should be handled
    assert cleaned_df.isnull().sum().sum() == 0

    # No exact duplicate rows should remain
    assert cleaned_df.duplicated().sum() == 0

    # Current clean_dataset() removes duplicates before
    # missing-value imputation. Imputation can create an
    # additional duplicate, resulting in 4 rows.
    assert len(cleaned_df) == 4