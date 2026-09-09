import mlflow

from src.register_model import (
    configure_mlflow,
    get_version_by_alias,
)


def test_champion_performance():

    configure_mlflow()

    champion = get_version_by_alias("champion")

    client = mlflow.MlflowClient()

    run = client.get_run(champion.run_id)

    rmse = run.data.metrics["test_rmse"]
    r2 = run.data.metrics["test_r2"]

    assert rmse < 250000
    assert r2 > 0.65