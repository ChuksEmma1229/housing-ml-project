from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_groups(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Separate numeric and categorical feature columns.
    """
    numeric_features = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns were found.")

    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing suitable for mixed housing data.

    Numeric values:
    - Median imputation
    - Standard scaling

    Categorical values:
    - Most-frequent imputation
    - One-hot encoding
    """
    numeric_features, categorical_features = get_feature_groups(X)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def describe_preprocessing(X: pd.DataFrame) -> None:
    """
    Print a simple summary for terminal evidence.
    """
    numeric_features, categorical_features = get_feature_groups(X)

    print("Preprocessing summary")
    print("---------------------")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    print("Numeric missing values: median imputation")
    print("Categorical missing values: most-frequent imputation")
    print("Numeric scaling: StandardScaler")
    print("Categorical encoding: OneHotEncoder")