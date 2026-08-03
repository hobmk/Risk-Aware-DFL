import pytest
import torch

from implementations.rcr_dfl.src.model import ReturnMLP


def test_return_mlp_output_shape() -> None:
    model = ReturnMLP(
        n_assets=4,
        lookback=6,
        hidden_dim=16,
        dropout=0.1,
    )
    features = torch.randn(3, 6, 4)
    output = model(features)

    assert output.shape == (3, 4)


def test_return_mlp_rejects_invalid_shape() -> None:
    model = ReturnMLP(n_assets=4, lookback=6, hidden_dim=16)

    with pytest.raises(ValueError):
        model(torch.randn(6, 4))
