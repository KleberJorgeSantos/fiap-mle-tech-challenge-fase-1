"""Tests adicionais para src/models/trainer.py — cobre train_mlp, save_model
e load_model."""

import os
import tempfile

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.models.mlp import ChurnMLP
from src.models.trainer import load_model, save_model, train_mlp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_data(n: int = 200, dim: int = 10):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    return X, y


def _tiny_model_and_data(dim: int = 10):
    X_train, y_train = _synthetic_data(128, dim)
    X_val, y_val = _synthetic_data(32, dim)
    model = ChurnMLP(input_dim=dim)
    return model, X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# train_mlp — histórico
# ---------------------------------------------------------------------------

def test_train_mlp_returns_history_dict():
    model, X_train, y_train, X_val, y_val = _tiny_model_and_data()
    history = train_mlp(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32)
    assert isinstance(history, dict)
    assert "train_loss" in history and "val_loss" in history


def test_train_mlp_history_lengths_match():
    model, X_train, y_train, X_val, y_val = _tiny_model_and_data()
    history = train_mlp(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32)
    assert len(history["train_loss"]) == len(history["val_loss"])


def test_train_mlp_losses_are_floats():
    model, X_train, y_train, X_val, y_val = _tiny_model_and_data()
    history = train_mlp(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32)
    for loss in history["train_loss"] + history["val_loss"]:
        assert isinstance(loss, float)


def test_train_mlp_loss_is_positive():
    model, X_train, y_train, X_val, y_val = _tiny_model_and_data()
    history = train_mlp(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32)
    assert all(loss > 0 for loss in history["train_loss"])
    assert all(loss > 0 for loss in history["val_loss"])


# ---------------------------------------------------------------------------
# train_mlp — early stopping
# ---------------------------------------------------------------------------

def test_train_mlp_early_stopping_terminates():
    """Com patience=2 e 50 epochs max, deve terminar antes das 50 epochs."""
    model, X_train, y_train, X_val, y_val = _tiny_model_and_data()
    history = train_mlp(
        model, X_train, y_train, X_val, y_val,
        epochs=50, batch_size=32, patience=2,
    )
    assert len(history["train_loss"]) <= 50


def test_train_mlp_runs_requested_epochs_when_no_early_stop():
    """Com patience alto, deve rodar os epochs pedidos."""
    model, X_train, y_train, X_val, y_val = _tiny_model_and_data()
    history = train_mlp(
        model, X_train, y_train, X_val, y_val,
        epochs=5, batch_size=32, patience=1000,
    )
    assert len(history["train_loss"]) == 5


# ---------------------------------------------------------------------------
# save_model / load_model — round-trip
# ---------------------------------------------------------------------------

def _dummy_pipeline(n_features: int = 3):
    """Retorna um StandardScaler fitado — serve como pipeline placeholder."""
    pipe = StandardScaler()
    pipe.fit(np.random.rand(10, n_features))
    return pipe


def test_save_model_creates_pt_file():
    model = ChurnMLP(input_dim=10)
    pipeline = _dummy_pipeline()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model")
        save_model(model, pipeline, path)
        assert os.path.exists(path + ".pt")


def test_save_model_creates_pipeline_file():
    model = ChurnMLP(input_dim=10)
    pipeline = _dummy_pipeline()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model")
        save_model(model, pipeline, path)
        assert os.path.exists(path + "_pipeline.joblib")


def test_load_model_returns_churn_mlp():
    dim = 10
    model = ChurnMLP(input_dim=dim)
    pipeline = _dummy_pipeline()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model")
        save_model(model, pipeline, path)
        loaded_model, _ = load_model(path, input_dim=dim)
    assert isinstance(loaded_model, ChurnMLP)


def test_load_model_is_in_eval_mode():
    dim = 10
    model = ChurnMLP(input_dim=dim)
    pipeline = _dummy_pipeline()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model")
        save_model(model, pipeline, path)
        loaded_model, _ = load_model(path, input_dim=dim)
    assert not loaded_model.training


def test_save_load_model_preserves_weights():
    import torch

    dim = 10
    model = ChurnMLP(input_dim=dim)
    pipeline = _dummy_pipeline()

    # Captura os pesos antes de salvar
    original_weights = {k: v.clone() for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model")
        save_model(model, pipeline, path)
        loaded_model, _ = load_model(path, input_dim=dim)

    loaded_weights = loaded_model.state_dict()
    for key in original_weights:
        assert torch.allclose(original_weights[key], loaded_weights[key]), (
            f"Pesos diferentes para a camada '{key}'"
        )
