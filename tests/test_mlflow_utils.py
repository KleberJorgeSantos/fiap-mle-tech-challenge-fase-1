"""Tests for src/tracking/mlflow_utils.py — cobertura via mock do mlflow."""

from unittest.mock import MagicMock, patch

from src.tracking.mlflow_utils import log_mlp_run, log_sklearn_run, setup_experiment


def _mock_run_context(mock_mlflow):
    """Configura o mock do context manager mlflow.start_run."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock())
    ctx.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value = ctx
    return ctx


# ---------------------------------------------------------------------------
# setup_experiment
# ---------------------------------------------------------------------------

@patch("src.tracking.mlflow_utils.mlflow")
def test_setup_experiment_calls_set_experiment(mock_mlflow):
    setup_experiment("meu-experimento")
    mock_mlflow.set_experiment.assert_called_once_with("meu-experimento")


@patch("src.tracking.mlflow_utils.mlflow")
def test_setup_experiment_default_name(mock_mlflow):
    setup_experiment()
    mock_mlflow.set_experiment.assert_called_once()
    args, _ = mock_mlflow.set_experiment.call_args
    assert isinstance(args[0], str) and len(args[0]) > 0


# ---------------------------------------------------------------------------
# log_sklearn_run
# ---------------------------------------------------------------------------

@patch("src.tracking.mlflow_utils.mlflow")
def test_log_sklearn_run_logs_params(mock_mlflow):
    _mock_run_context(mock_mlflow)
    log_sklearn_run("run1", {"alpha": 0.1}, {"roc_auc": 0.85})
    mock_mlflow.log_params.assert_called_once_with({"alpha": 0.1})


@patch("src.tracking.mlflow_utils.mlflow")
def test_log_sklearn_run_logs_metrics(mock_mlflow):
    _mock_run_context(mock_mlflow)
    log_sklearn_run("run1", {}, {"roc_auc": 0.85, "f1": 0.78})
    mock_mlflow.log_metrics.assert_called_once_with({"roc_auc": 0.85, "f1": 0.78})


@patch("src.tracking.mlflow_utils.mlflow")
def test_log_sklearn_run_with_model_calls_log_model(mock_mlflow):
    _mock_run_context(mock_mlflow)
    fake_model = MagicMock()
    log_sklearn_run("run_model", {}, {}, model=fake_model)
    mock_mlflow.sklearn.log_model.assert_called_once_with(fake_model, name="model")


@patch("src.tracking.mlflow_utils.mlflow")
def test_log_sklearn_run_without_model_skips_log_model(mock_mlflow):
    _mock_run_context(mock_mlflow)
    log_sklearn_run("run_no_model", {}, {})
    mock_mlflow.sklearn.log_model.assert_not_called()


@patch("src.tracking.mlflow_utils.mlflow")
def test_log_sklearn_run_uses_run_name(mock_mlflow):
    _mock_run_context(mock_mlflow)
    log_sklearn_run("meu_run", {}, {})
    mock_mlflow.start_run.assert_called_once_with(run_name="meu_run")


# ---------------------------------------------------------------------------
# log_mlp_run
# ---------------------------------------------------------------------------

@patch("src.tracking.mlflow_utils.mlflow")
def test_log_mlp_run_logs_params_and_metrics(mock_mlflow):
    _mock_run_context(mock_mlflow)
    log_mlp_run("mlp_run", {"lr": 0.001}, {"roc_auc": 0.90})
    mock_mlflow.log_params.assert_called_once_with({"lr": 0.001})
    # log_metrics chamado ao menos uma vez para as métricas principais
    assert mock_mlflow.log_metrics.call_count >= 1


@patch("src.tracking.mlflow_utils.mlflow")
def test_log_mlp_run_with_history_logs_per_step(mock_mlflow):
    _mock_run_context(mock_mlflow)
    history = {"train_loss": [0.6, 0.5, 0.4], "val_loss": [0.7, 0.6, 0.5]}
    log_mlp_run("mlp_run_hist", {}, {"roc_auc": 0.9}, history=history)
    # 1 chamada para métricas principais + 3 chamadas para os steps
    assert mock_mlflow.log_metrics.call_count == 4


@patch("src.tracking.mlflow_utils.mlflow")
def test_log_mlp_run_without_history_single_metrics_call(mock_mlflow):
    _mock_run_context(mock_mlflow)
    log_mlp_run("mlp_no_hist", {}, {"roc_auc": 0.88})
    assert mock_mlflow.log_metrics.call_count == 1


@patch("src.tracking.mlflow_utils.mlflow")
def test_log_mlp_run_with_model_calls_pytorch_log(mock_mlflow):
    _mock_run_context(mock_mlflow)
    fake_model = MagicMock()
    log_mlp_run("mlp_with_model", {}, {}, model=fake_model)
    mock_mlflow.pytorch.log_model.assert_called_once_with(fake_model, name="model")


@patch("src.tracking.mlflow_utils.mlflow")
def test_log_mlp_run_without_model_skips_pytorch_log(mock_mlflow):
    _mock_run_context(mock_mlflow)
    log_mlp_run("mlp_no_model", {}, {})
    mock_mlflow.pytorch.log_model.assert_not_called()
