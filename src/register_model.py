from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

# An absolute SQLite URI prevents MLflow from creating different databases
# depending on the terminal's current working directory.
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

EXPERIMENT_NAME = "housing-model-experiments"
REGISTERED_MODEL_NAME = "housing_price_prediction"


def configure_mlflow() -> MlflowClient:
    """
    Configure MLflow tracking and registry to use the project's shared
    SQLite database.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

    return MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI,
        registry_uri=MLFLOW_TRACKING_URI,
    )


def get_or_create_experiment(
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """
    Return the MLflow experiment ID, creating the experiment if necessary.
    """
    client = configure_mlflow()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is not None:
        return experiment.experiment_id

    return client.create_experiment(experiment_name)


def ensure_registered_model(
    model_name: str = REGISTERED_MODEL_NAME,
) -> None:
    """
    Create the registered model if it does not already exist.
    """
    client = configure_mlflow()

    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            model_name,
            description=(
                "Housing price regression models tracked for the "
                "MLflow champion and challenger workflow."
            ),
        )


def get_model_version_for_run(
    run_id: str,
    model_name: str = REGISTERED_MODEL_NAME,
) -> ModelVersion:
    """
    Locate the registered model version produced by a specific MLflow run.
    """
    client = configure_mlflow()

    versions = client.search_model_versions(
        filter_string=f"name = '{model_name}'"
    )

    matching_versions = [
        version
        for version in versions
        if version.run_id == run_id
    ]

    if not matching_versions:
        raise RuntimeError(
            f"No registered version of '{model_name}' was found "
            f"for run ID '{run_id}'."
        )

    return max(
        matching_versions,
        key=lambda version: int(version.version),
    )


def assign_alias(
    alias: str,
    version: str | int,
    model_name: str = REGISTERED_MODEL_NAME,
) -> None:
    """
    Assign an MLflow registry alias to a model version.
    """
    client = configure_mlflow()

    alias_clean = alias.removeprefix("@")

    client.set_registered_model_alias(
        name=model_name,
        alias=alias_clean,
        version=str(version),
    )

    client.set_model_version_tag(
        name=model_name,
        version=str(version),
        key="lifecycle_alias",
        value=alias_clean,
    )

    print(
        f"Assigned alias '@{alias_clean}' to "
        f"'{model_name}' version {version}."
    )


def get_version_by_alias(
    alias: str,
    model_name: str = REGISTERED_MODEL_NAME,
) -> ModelVersion:
    """
    Retrieve a model version through an MLflow alias.
    """
    client = configure_mlflow()

    return client.get_model_version_by_alias(
        name=model_name,
        alias=alias.removeprefix("@"),
    )


def get_metric_for_version(
    model_version: ModelVersion,
    metric_name: str = "test_rmse",
) -> float:
    """
    Retrieve a logged evaluation metric from the model version's source run.
    """
    client = configure_mlflow()
    run = client.get_run(model_version.run_id)

    if metric_name not in run.data.metrics:
        raise KeyError(
            f"Metric '{metric_name}' was not found in run "
            f"'{model_version.run_id}'. Available metrics: "
            f"{list(run.data.metrics.keys())}"
        )

    return float(run.data.metrics[metric_name])


def show_alias(
    alias: str,
    model_name: str = REGISTERED_MODEL_NAME,
) -> None:
    """
    Print the model version and metrics associated with an alias.
    """
    model_version = get_version_by_alias(alias, model_name)
    client = configure_mlflow()
    run = client.get_run(model_version.run_id)

    print(f"Model: {model_name}")
    print(f"Alias: @{alias.removeprefix('@')}")
    print(f"Version: {model_version.version}")
    print(f"Run ID: {model_version.run_id}")
    print(f"Metrics: {run.data.metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage MLflow aliases for the housing model."
    )

    parser.add_argument(
        "--alias",
        required=True,
        help="Alias name, for example champion or challenger.",
    )

    parser.add_argument(
        "--version",
        required=False,
        help="Model version to assign. Omit this to inspect an alias.",
    )

    parser.add_argument(
        "--model-name",
        default=REGISTERED_MODEL_NAME,
    )

    args = parser.parse_args()

    print(f"MLflow database: {MLFLOW_DB_PATH}")
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    if args.version:
        assign_alias(
            alias=args.alias,
            version=args.version,
            model_name=args.model_name,
        )
    else:
        show_alias(
            alias=args.alias,
            model_name=args.model_name,
        )


if __name__ == "__main__":
    main()