import logging

import mlflow
import mlflow.pytorch
import mlflow.sklearn

from src.config import MLFLOW_EXPERIMENT_CHURN

logger = logging.getLogger(__name__)


def setup_experiment(experiment_name: str = MLFLOW_EXPERIMENT_CHURN):
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow experiment: %s", experiment_name)


def log_sklearn_run(name: str, params: dict, metrics: dict, model=None):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model is not None:
            mlflow.sklearn.log_model(model, name="model")
        logger.info("Logged run '%s' to MLflow", name)


def log_mlp_run(name: str, params: dict, metrics: dict, model=None, history: dict = None):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if history:
            for i, (tl, vl) in enumerate(zip(history["train_loss"], history["val_loss"])):
                mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=i)
        if model is not None:
            mlflow.pytorch.log_model(model, name="model")
        logger.info("Logged MLP run '%s' to MLflow", name)
