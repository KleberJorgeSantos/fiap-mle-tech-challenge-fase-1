import numpy as np
import torch

from src.models.mlp import ChurnMLP
from src.models.trainer import predict_proba


def test_mlp_forward_pass():
    model = ChurnMLP(input_dim=10)
    X = torch.randn(5, 10)
    out = model(X)
    assert out.shape == (5,), f"Expected shape (5,), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Output not in [0,1]"


def test_predict_proba_shape():
    model = ChurnMLP(input_dim=20)
    model.eval()
    X = np.random.rand(8, 20).astype(np.float32)
    proba = predict_proba(model, X)
    assert proba.shape == (8,), f"Expected (8,), got {proba.shape}"
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_mlp_different_input_dims():
    for dim in [5, 30, 100]:
        model = ChurnMLP(input_dim=dim)
        X = torch.randn(3, dim)
        out = model(X)
        assert out.shape == (3,)
