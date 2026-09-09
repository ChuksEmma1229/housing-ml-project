from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from src.data_loader import load_housing_data, split_housing_data
from src.preprocess import build_preprocessor, describe_preprocessing
from src.register_model import (
    EXPERIMENT_NAME,
    MLFLOW_DB_PATH,
    REGISTERED_MODEL_NAME,
    assign_alias,
    configure_mlflow,
    ensure_registered_model,
    get_model_version_for_run,
    get_or_create_experiment,
)
import joblib


RANDOM_STATE = 42
TEST_SIZE = 0.20


def calculate_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate regression evaluation metrics.
    """
    mse = mean_squared_error(y_true, y_pred)

    return {
        "test_rmse": float(np.sqrt(mse)),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_r2": float(r2_score(y_true, y_pred)),
    }


def train_and_log_baseline(
    model_name: str,
    estimator,
    model_parameters: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_column: str,
    experiment_id: str,
) -> dict:
    """
    Train and log one baseline model inside a nested MLflow run.

    MLflow 3 returns a LoggedModel object from log_model().
    Its model_uri must be retained for later registry registration.
    """
    run_timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    run_name = f"baseline_{model_name.lower()}_{run_timestamp}"

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        nested=True,
    ) as child_run:

        pipeline.fit(X_train, y_train)
        joblib.dump(
            pipeline,
            "model.pkl"
        )
        predictions = pipeline.predict(X_test)

        metrics = calculate_regression_metrics(
            y_true=y_test,
            y_pred=predictions,
        )

        mlflow.log_params(
            {
                "model_type": model_name,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "target_column": target_column,
                "training_rows": len(X_train),
                "test_rows": len(X_test),
                **model_parameters,
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                "workflow_stage": "baseline",
                "candidate_role": "champion_candidate",
                "task_type": "regression",
            }
        )

        # Convert integer columns to float in the input example.
        # This avoids schema problems when inference input contains nulls.
        input_example = X_train.head(5).copy()

        integer_columns = input_example.select_dtypes(
            include=["integer"]
        ).columns

        input_example[integer_columns] = input_example[
            integer_columns
        ].astype("float64")

        # MLflow 3 returns a ModelInfo object.
        logged_model = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        print()
        print(f"Completed: {model_name}")
        print(f"Run ID: {child_run.info.run_id}")
        print(f"Logged model ID: {logged_model.model_id}")
        print(f"Logged model URI: {logged_model.model_uri}")
        print(f"RMSE: {metrics['test_rmse']:.6f}")
        print(f"MAE: {metrics['test_mae']:.6f}")
        print(f"R2: {metrics['test_r2']:.6f}")

        return {
            "model_name": model_name,
            "run_id": child_run.info.run_id,
            "logged_model_id": logged_model.model_id,
            "model_uri": logged_model.model_uri,
            "metrics": metrics,
        }


def register_best_baseline(result: dict) -> str:
    """
    Register the best baseline LoggedModel and assign @champion.

    MLflow 3 models are registered using the URI returned by
    mlflow.sklearn.log_model(), not runs:/<run_id>/model.
    """
    model_uri = result["model_uri"]

    print()
    print("Registering best baseline")
    print("-------------------------")
    print(f"Selected model: {result['model_name']}")
    print(f"Source run: {result['run_id']}")
    print(f"Logged model ID: {result['logged_model_id']}")
    print(f"Model URI: {model_uri}")

    registered_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    assign_alias(
        alias="champion",
        version=registered_version.version,
        model_name=REGISTERED_MODEL_NAME,
    )

    client = configure_mlflow()

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="evaluation_rmse",
        value=str(result["metrics"]["test_rmse"]),
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="evaluation_mae",
        value=str(result["metrics"]["test_mae"]),
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="evaluation_r2",
        value=str(result["metrics"]["test_r2"]),
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="model_type",
        value=result["model_name"],
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="model_role",
        value="baseline_champion",
    )

    print(
        f"Registered '{REGISTERED_MODEL_NAME}' "
        f"version {registered_version.version}."
    )

    print(
        f"Assigned @champion to version "
        f"{registered_version.version}."
    )

    return str(registered_version.version)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Linear Regression and Ridge housing baselines."
        )
    )

    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional path to a housing CSV.",
    )

    parser.add_argument(
        "--target",
        default=None,
        help="Optional name of the target column.",
    )

    args = parser.parse_args()

    configure_mlflow()
    ensure_registered_model()
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)

    print(f"MLflow database: {MLFLOW_DB_PATH}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Registered model: {REGISTERED_MODEL_NAME}")

    X, y, target_column = load_housing_data(
        data_path=args.data_path,
        target_column=args.target,
    )

    describe_preprocessing(X)

    X_train, X_test, y_train, y_test = split_housing_data(
        X=X,
        y=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    baseline_models = {
        "LinearRegression": {
            "estimator": LinearRegression(),
            "parameters": {
                "fit_intercept": True,
            },
        },
        "Ridge": {
            "estimator": Ridge(
                alpha=1.0,
                random_state=RANDOM_STATE,
            ),
            "parameters": {
                "alpha": 1.0,
                "fit_intercept": True,
            },
        },
    }

    parent_timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    parent_run_name = f"housing_baseline_comparison_{parent_timestamp}"

    results = []

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=parent_run_name,
    ) as parent_run:
        mlflow.set_tags(
            {
                "workflow_stage": "baseline_comparison",
                "run_structure": "parent",
                "task_type": "regression",
            }
        )

        for model_name, configuration in baseline_models.items():
            result = train_and_log_baseline(
                model_name=model_name,
                estimator=clone(configuration["estimator"]),
                model_parameters=configuration["parameters"],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                target_column=target_column,
                experiment_id=experiment_id,
            )

            results.append(result)

        best_result = min(
            results,
            key=lambda result: result["metrics"]["test_rmse"],
        )

        summary = {
            result["model_name"]: result["metrics"]
            for result in results
        }

        summary_path = (
            MLFLOW_DB_PATH.parent / "baseline_results.json"
        )

        summary_path.write_text(
            json.dumps(summary, indent=4),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(summary_path))

        mlflow.log_metric(
            "best_baseline_rmse",
            best_result["metrics"]["test_rmse"],
        )

        mlflow.set_tag(
            "best_baseline_model",
            best_result["model_name"],
        )

    champion_version = register_best_baseline(best_result)

    print()
    print("Baseline training complete")
    print("--------------------------")
    print(f"Best model: {best_result['model_name']}")
    print(
        f"Best RMSE: "
        f"{best_result['metrics']['test_rmse']:.6f}"
    )
    print(f"Registered version: {champion_version}")
    print("Alias assigned: @champion")


if __name__ == "__main__":
    main()