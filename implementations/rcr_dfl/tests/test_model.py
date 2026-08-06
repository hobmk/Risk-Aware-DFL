import pytest
import torch

from implementations.rcr_dfl.src.model import ReturnMLP


def test_return_mlp_output_shape() -> None:
    model = ReturnMLP(n_assets=5, lookback=10, hidden_dim=16, dropout=0.0)
    output = model(torch.randn(3, 10, 5))
    assert output.shape == (3, 5)


def test_return_mlp_rejects_wrong_shape() -> None:
    model = ReturnMLP(n_assets=5, lookback=10, hidden_dim=16)
    with pytest.raises(ValueError):
        model(torch.randn(10, 5))
