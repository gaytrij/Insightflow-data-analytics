import pandas as pd

from src.data_loader import (
    load_csv,
    get_dataset_profile
)


def test_load_csv(tmp_path):

    csv_file = tmp_path / "test_data.csv"

    csv_file.write_text(
        "ID,Age,Category\n"
        "1,22,A\n"
        "2,30,B\n"
        "3,25,A\n"
    )

    df = load_csv(csv_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == [
        "ID",
        "Age",
        "Category"
    ]


def test_get_dataset_profile():

    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "Age": [22, 30, 25],
        "Category": ["A", "B", "A"]
    })

    profile = get_dataset_profile(df)

    assert profile["rows"] == 3
    assert profile["columns"] == 3
    assert profile["missing_values"] == 0
    assert profile["duplicate_rows"] == 0

    assert "ID" in profile["numerical_columns"]
    assert "Age" in profile["numerical_columns"]

    assert "Category" in profile["categorical_columns"]


def test_dataset_profile_detects_missing_and_duplicates():

    df = pd.DataFrame({
        "Age": [22, 30, 30],
        "Income": [25000, 50000, 50000]
    })

    profile = get_dataset_profile(df)

    assert profile["rows"] == 3
    assert profile["columns"] == 2
    assert profile["missing_values"] == 0
    assert profile["duplicate_rows"] == 1