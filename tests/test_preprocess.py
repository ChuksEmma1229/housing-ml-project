import pandas as pd

from src.preprocess import build_preprocessor


def test_preprocessor_created():

    df = pd.DataFrame(
        {
            "bedrooms": [3, 4],
            "bathrooms": [2, 3],
        }
    )

    pipeline = build_preprocessor(df)

    assert pipeline is not None


def test_missing_values():

    df = pd.DataFrame(
        {
            "bedrooms": [3, None, 5],
            "bathrooms": [2, 3, None],
        }
    )

    pipeline = build_preprocessor(df)

    result = pipeline.fit_transform(df)

    assert result is not None