import pytest
import torch

pytest.importorskip("cvxpy")
pytest.importorskip("cvxpylayers")

from implementations.rcr_dfl.src.optimization import build_markowitz_layer, covariance_to_risk_factor


def test_risk_factor_reconstructs_scaled_covariance() -> None:
    covariance = torch.tensor([[0.04, 0.01], [0.01, 0.09]], dtype=torch.float64)
    factor = covariance_to_risk_factor(covariance, risk_aversion=2.0)
    assert torch.allclose(factor.T @ factor, 2.0 * covariance, atol=1e-12)


def test_markowitz_layer_respects_constraints() -> None:
    layer = build_markowitz_layer(n_assets=3, max_weight=0.6)
    expected_returns = torch.tensor([[0.02, 0.01, -0.01]], dtype=torch.float64)
    risk_factor = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    weights, = layer(expected_returns, risk_factor)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, dtype=torch.float64), atol=1e-5)
    assert weights.min() >= -1e-5
    assert weights.max() <= 0.60001
