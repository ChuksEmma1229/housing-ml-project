from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.data_loader import load_housing_data
from src.register_model import (
    MLFLOW_DB_PATH,
    REGISTERED_MODEL_NAME,
    configure_mlflow,
    get_version_by_alias,
)

DEFAULT_ALIAS = "champion"


def load_registered_model(
    alias: str = DEFAULT_ALIAS,
):
    """
    Load a registered MLflow model using an alias.
    """
    configure_mlflow()

    clean_alias = alias.removeprefix("@")

    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}@{clean_alias}"
    )

    model = mlflow.pyfunc.load_model(model_uri)

    return model, model_uri


def predict_dataframe(
    input_dataframe: pd.DataFrame,
    alias: str = DEFAULT_ALIAS,
) -> list:
    """
    Generate predictions from a pandas DataFrame.
    """
    model, model_uri = load_registered_model(alias)

    predictions = model.predict(input_dataframe)

    print(f"Loaded model: {model_uri}")

    return [float(value) for value in predictions]


def predict_from_json(
    payload: dict[str, Any],
    alias: str = DEFAULT_ALIAS,
) -> list:
    """
    Generate predictions from one JSON object.
    """
    input_dataframe = pd.DataFrame([payload])

    # Convert integer columns to float
    for col in input_dataframe.select_dtypes(include=["int64"]).columns:
        input_dataframe[col] = input_dataframe[col].astype(float)

    return predict_dataframe(
        input_dataframe=input_dataframe,
        alias=alias,
    )


def create_example_payload(
    data_path: str | None = None,
    target_column: str | None = None,
) -> dict[str, Any]:
    """
    Create a valid prediction payload using the first dataset row.
    """
    X, _, _ = load_housing_data(
        data_path=data_path,
        target_column=target_column,
    )

    payload = X.iloc[0].to_dict()

    serialisable_payload = {}

    for key, value in payload.items():
        if pd.isna(value):
            serialisable_payload[key] = None

        elif hasattr(value, "item"):
            serialisable_payload[key] = value.item()

        else:
            serialisable_payload[key] = value

    return serialisable_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference using the current MLflow champion."
    )

    parser.add_argument(
        "--alias",
        default="champion",
        help="Registry alias to load.",
    )

    parser.add_argument(
        "--json",
        default=None,
        help=(
            'JSON payload, for example '
            '\'{"MedInc":8.3,"HouseAge":20}\'.'
        ),
    )

    parser.add_argument(
        "--json-file",
        default=None,
        help="Path to a JSON payload file.",
    )

    parser.add_argument(
        "--data-path",
        default=None,
    )

    parser.add_argument(
        "--target",
        default=None,
    )

    args = parser.parse_args()

    configure_mlflow()

    version = get_version_by_alias(args.alias)

    print(f"MLflow database: {MLFLOW_DB_PATH}")
    print(f"Registry model: {REGISTERED_MODEL_NAME}")
    print(f"Alias: @{args.alias.removeprefix('@')}")
    print(f"Version: {version.version}")

    if args.json:

        payload = json.loads(args.json)

    elif args.json_file:

        payload_path = Path(args.json_file)

        payload = json.loads(
            payload_path.read_text(
                encoding="utf-8"
            )
        )

    else:

        payload = create_example_payload(
            data_path=args.data_path,
            target_column=args.target,
        )

        print(
            "No payload supplied. Using the first row "
            "from the housing dataset."
        )

    predictions = predict_from_json(
        payload=payload,
        alias=args.alias,
    )

    output = {
        "model_name": REGISTERED_MODEL_NAME,
        "alias": f"@{args.alias.removeprefix('@')}",
        "version": str(version.version),
        "input": payload,
        "predictions": predictions,
    }

    print(
        json.dumps(
            output,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()