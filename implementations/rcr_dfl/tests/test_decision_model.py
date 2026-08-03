import pytest
import torch

pytest.importorskip("cvxpy")
pytest.importorskip("cvxpylayers")

from implementations.rcr_dfl.src.decision_model import (
    RCRMLPWithMarkowitz,
)
from implementations.rcr_dfl.src.losses import compute_rcr_losses


def _spd(batch_size: int, n_assets: int) -> torch.Tensor:
    matrix = torch.randn(
        batch_size,
        n_assets,
        n_assets,
        dtype=torch.float64,
    )
    return (
        matrix @ matrix.transpose(-1, -2)
        + 0.1 * torch.eye(n_assets, dtype=torch.float64)
    )


def test_decision_model_backward_reaches_mlp() -> None:
    torch.manual_seed(29)
    batch_size = 2
    lookback = 5
    n_assets = 3

    model = RCRMLPWithMarkowitz(
        n_assets=n_assets,
        lookback=lookback,
        hidden_dim=12,
        risk_aversion=20.0,
        max_weight=1.0,
        eta=0.5,
    ).float()
    features = torch.randn(batch_size, lookback, n_assets)
    targets = torch.randn(batch_size, n_assets) * 0.01
    covariance = _spd(batch_size, n_assets)
    residual_covariance = _spd(batch_size, n_assets)

    output = model(
        features=features,
        covariance=covariance,
        residual_covariance=residual_covariance,
    )
    oracle_weights = model.solve_oracle(
        true_returns=targets,
        risk_factor=output.risk_factor,
    )
    losses = compute_rcr_losses(
        predicted_returns=output.predicted_returns,
        predicted_weights=output.predicted_weights,
        oracle_weights=oracle_weights,
        true_returns=targets,
        risk_factor=output.risk_factor,
        alpha=0.5,
        mse_scale=15.0,
    )
    losses.total.backward()

    gradients = [
        parameter.grad
        for parameter in model.return_model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
