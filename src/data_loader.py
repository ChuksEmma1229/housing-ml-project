from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "kc_house_data.csv"

TARGET_CANDIDATES = [
    "MedHouseVal",
    "median_house_value",
    "median_house_value_clean",
    "house_value",
    "price",
    "sale_price",
    "SalePrice",
    "target",
]


def identify_target_column(
    dataframe: pd.DataFrame,
    target_column: Optional[str] = None,
) -> str:
    """
    Identify the target column in the housing dataset.
    """
    if target_column:
        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' is not in the dataset. "
                f"Available columns: {list(dataframe.columns)}"
            )

        return target_column

    for candidate in TARGET_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        "The target column could not be identified automatically. "
        "Supply it explicitly using the target_column argument. "
        f"Available columns: {list(dataframe.columns)}"
    )


def load_housing_data(
    data_path: Optional[str | Path] = None,
    target_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Load a local housing CSV when available.

    If no local CSV exists, load the California Housing dataset from
    scikit-learn.
    """
    path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    if path.exists():
        print(f"Loading local housing data from: {path}")
        dataframe = pd.read_csv(path)
        source_name = str(path)
    else:
        print(
            f"No CSV found at {path}. "
            "Loading the scikit-learn California Housing dataset."
        )

        housing = fetch_california_housing(as_frame=True)
        dataframe = housing.frame.copy()
        source_name = "sklearn.datasets.fetch_california_housing"

    if dataframe.empty:
        raise ValueError("The housing dataset is empty.")

    dataframe = dataframe.drop_duplicates().reset_index(drop=True)

    resolved_target = identify_target_column(
        dataframe=dataframe,
        target_column=target_column,
    )

    dataframe = dataframe.dropna(subset=[resolved_target]).reset_index(
        drop=True
    )

    X = dataframe.drop(columns=[resolved_target])
    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = X[col].astype(float)
    y = dataframe[resolved_target].astype(float)

    print(f"Dataset source: {source_name}")
    print(f"Rows: {len(dataframe):,}")
    print(f"Input columns: {X.shape[1]}")
    print(f"Target column: {resolved_target}")

    return X, y, resolved_target


def split_housing_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create reproducible training and test datasets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and inspect the housing dataset."
    )

    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional path to housing.csv.",
    )

    parser.add_argument(
        "--target",
        default=None,
        help="Optional name of the target column.",
    )

    args = parser.parse_args()

    X, y, target = load_housing_data(
        data_path=args.data_path,
        target_column=args.target,
    )

    X_train, X_test, y_train, y_test = split_housing_data(X, y)

    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Target mean: {y.mean():.4f}")
    print(f"Target name: {target}")


if __name__ == "__main__":
    main()