from __future__ import annotations

import argparse
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import (
    load_housing_data,
    split_housing_data,
)
from src.preprocess import build_preprocessor
from src.register_model import (
    EXPERIMENT_NAME,
    MLFLOW_DB_PATH,
    REGISTERED_MODEL_NAME,
    assign_alias,
    configure_mlflow,
    ensure_registered_model,
    get_metric_for_version,
    get_or_create_experiment,
    get_version_by_alias,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate the regression evaluation metrics.

    Lower RMSE and MAE are better.
    Higher R-squared is better.
    """
    mse = mean_squared_error(y_true, y_pred)

    return {
        "test_rmse": float(np.sqrt(mse)),
        "test_mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "test_r2": float(r2_score(y_true, y_pred)),
    }


def prepare_input_example(
    X_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an MLflow input example.

    Integer columns are converted to float64 so the inferred
    signature can also accept missing numeric values at inference.
    """
    input_example = X_train.head(5).copy()

    integer_columns = input_example.select_dtypes(
        include=["integer"]
    ).columns

    if len(integer_columns) > 0:
        input_example[integer_columns] = input_example[
            integer_columns
        ].astype("float64")

    return input_example


def train_challenger(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_column: str,
    experiment_id: str,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
) -> tuple[str, dict[str, float]]:
    """
    Train, log and register the Decision Tree challenger.

    Returns:
        Registered version number and evaluation metrics.
    """
    parameters = {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "random_state": RANDOM_STATE,
    }

    challenger = DecisionTreeRegressor(**parameters)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(X_train),
            ),
            (
                "regressor",
                challenger,
            ),
        ]
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    run_name = f"challenger_decision_tree_{timestamp}"

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
    ) as run:
        print()
        print("Training Decision Tree challenger")
        print("---------------------------------")
        print(f"Run name: {run_name}")

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=predictions,
        )

        mlflow.log_params(
            {
                "model_type": "DecisionTreeRegressor",
                "target_column": target_column,
                "test_size": TEST_SIZE,
                "training_rows": len(X_train),
                "test_rows": len(X_test),
                **parameters,
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                "workflow_stage": "challenger",
                "candidate_role": "challenger",
                "task_type": "regression",
                "comparison_metric": "test_rmse",
            }
        )

        input_example = prepare_input_example(X_train)

        # MLflow 3 returns a ModelInfo object.
        # The returned model_uri is required for registration.
        logged_model = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        run_id = run.info.run_id
        model_uri = logged_model.model_uri
        logged_model_id = logged_model.model_id

        print(f"Run ID: {run_id}")
        print(f"LoggedModel ID: {logged_model_id}")
        print(f"LoggedModel URI: {model_uri}")
        print(f"RMSE: {metrics['test_rmse']:.6f}")
        print(f"MAE: {metrics['test_mae']:.6f}")
        print(f"R2: {metrics['test_r2']:.6f}")

    # Register the MLflow 3 LoggedModel URI.
    # Do not use runs:/<run_id>/model here.
    registered_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client = configure_mlflow()

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="evaluation_rmse",
        value=str(metrics["test_rmse"]),
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="evaluation_mae",
        value=str(metrics["test_mae"]),
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="evaluation_r2",
        value=str(metrics["test_r2"]),
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="model_type",
        value="DecisionTreeRegressor",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="model_role",
        value="challenger",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="logged_model_id",
        value=str(logged_model_id),
    )

    assign_alias(
        alias="challenger",
        version=registered_version.version,
        model_name=REGISTERED_MODEL_NAME,
    )

    print()
    print("Challenger registered")
    print("---------------------")
    print(f"Model: {REGISTERED_MODEL_NAME}")
    print(f"Version: {registered_version.version}")
    print("Alias: @challenger")
    print(f"Run ID: {run_id}")
    print(f"RMSE: {metrics['test_rmse']:.6f}")
    print(f"MAE: {metrics['test_mae']:.6f}")
    print(f"R2: {metrics['test_r2']:.6f}")

    return str(registered_version.version), metrics


def compare_and_promote() -> None:
    """
    Compare the registered champion and challenger using test RMSE.

    The model with the lower test RMSE is the winner.

    If the challenger wins:
        - Existing champion becomes @previous-champion.
        - Challenger becomes @champion.
        - @challenger is retained for clear MLflow UI evidence.

    If the current champion wins:
        - @champion remains unchanged.
        - @challenger remains on the challenger version.
    """
    client = configure_mlflow()

    champion = get_version_by_alias(
        alias="champion",
        model_name=REGISTERED_MODEL_NAME,
    )

    challenger = get_version_by_alias(
        alias="challenger",
        model_name=REGISTERED_MODEL_NAME,
    )

    champion_rmse = get_metric_for_version(
        champion,
        metric_name="test_rmse",
    )

    challenger_rmse = get_metric_for_version(
        challenger,
        metric_name="test_rmse",
    )

    rmse_improvement = champion_rmse - challenger_rmse

    print()
    print("Champion versus challenger")
    print("--------------------------")
    print(
        f"Champion version {champion.version}: "
        f"RMSE {champion_rmse:.6f}"
    )
    print(
        f"Challenger version {challenger.version}: "
        f"RMSE {challenger_rmse:.6f}"
    )
    print(
        f"RMSE improvement if challenger is promoted: "
        f"{rmse_improvement:.6f}"
    )

    comparison_run_name = (
        "alias_comparison_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )

    experiment_id = get_or_create_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=comparison_run_name,
    ):
        mlflow.log_metrics(
            {
                "champion_rmse": champion_rmse,
                "challenger_rmse": challenger_rmse,
                "rmse_improvement": rmse_improvement,
            }
        )

        mlflow.log_params(
            {
                "champion_version": str(
                    champion.version
                ),
                "challenger_version": str(
                    challenger.version
                ),
                "comparison_metric": "test_rmse",
                "promotion_rule": "lower_is_better",
            }
        )

        mlflow.set_tags(
            {
                "workflow_stage": (
                    "champion_challenger_comparison"
                ),
                "task_type": "regression",
                "automated_decision": "true",
            }
        )

        if challenger_rmse < champion_rmse:
            # Preserve the current champion before promotion.
            assign_alias(
                alias="previous-champion",
                version=champion.version,
                model_name=REGISTERED_MODEL_NAME,
            )

            # Promote the challenger.
            assign_alias(
                alias="champion",
                version=challenger.version,
                model_name=REGISTERED_MODEL_NAME,
            )

            # Keep @challenger on the promoted version.
            # This makes the complete workflow visible in the UI.
            assign_alias(
                alias="challenger",
                version=challenger.version,
                model_name=REGISTERED_MODEL_NAME,
            )

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=challenger.version,
                key="promotion_result",
                value="promoted_to_champion",
            )

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=challenger.version,
                key="model_role",
                value="champion",
            )

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=champion.version,
                key="promotion_result",
                value="moved_to_previous_champion",
            )

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=champion.version,
                key="model_role",
                value="previous_champion",
            )

            mlflow.set_tags(
                {
                    "comparison_result": (
                        "challenger_promoted"
                    ),
                    "comparison_winner": (
                        f"version_{challenger.version}"
                    ),
                }
            )

            print()
            print("Promotion completed")
            print("-------------------")
            print(
                f"Version {challenger.version} "
                "is now @champion."
            )
            print(
                f"Version {champion.version} "
                "is now @previous-champion."
            )
            print(
                f"Version {challenger.version} "
                "also retains @challenger for evidence."
            )

        else:
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=challenger.version,
                key="promotion_result",
                value="not_promoted",
            )

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=challenger.version,
                key="model_role",
                value="challenger",
            )

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=champion.version,
                key="promotion_result",
                value="remained_champion",
            )

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=champion.version,
                key="model_role",
                value="champion",
            )

            mlflow.set_tags(
                {
                    "comparison_result": (
                        "champion_retained"
                    ),
                    "comparison_winner": (
                        f"version_{champion.version}"
                    ),
                }
            )

            print()
            print("No promotion made")
            print("-----------------")
            print(
                f"Version {champion.version} "
                "remains @champion."
            )
            print(
                f"Version {challenger.version} "
                "remains @challenger."
            )


def main() -> None:
    """
    Execute the complete challenger and promotion workflow.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a tuned Decision Tree challenger, register "
            "the model, compare it with the current champion "
            "and automatically promote the lower-RMSE model."
        )
    )

    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional path to the housing CSV.",
    )

    parser.add_argument(
        "--target",
        default=None,
        help="Optional target column name.",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum Decision Tree depth.",
    )

    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=10,
        help=(
            "Minimum samples required to split an "
            "internal tree node."
        ),
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=5,
        help=(
            "Minimum samples required at a tree leaf."
        ),
    )

    args = parser.parse_args()

    configure_mlflow()
    ensure_registered_model()

    print(f"MLflow database: {MLFLOW_DB_PATH}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Registered model: {REGISTERED_MODEL_NAME}")

    # A champion must exist before a challenger can be compared.
    try:
        current_champion = get_version_by_alias(
            alias="champion",
            model_name=REGISTERED_MODEL_NAME,
        )

        print(
            f"Current champion is version "
            f"{current_champion.version}."
        )

    except mlflow.exceptions.MlflowException as error:
        raise RuntimeError(
            "No @champion alias was found. Run "
            "'python -m src.train' successfully before "
            "running the challenger workflow."
        ) from error

    X, y, target_column = load_housing_data(
        data_path=args.data_path,
        target_column=args.target,
    )

    X_train, X_test, y_train, y_test = (
        split_housing_data(
            X=X,
            y=y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    experiment_id = get_or_create_experiment(
        EXPERIMENT_NAME
    )

    train_challenger(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_column=target_column,
        experiment_id=experiment_id,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )

    compare_and_promote()


if __name__ == "__main__":
    main()