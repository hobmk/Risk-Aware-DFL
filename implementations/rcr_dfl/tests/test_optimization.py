import pytest
import torch

pytest.importorskip("cvxpy")
pytest.importorskip("cvxpylayers")

from implementations.rcr_dfl.src.optimization import (
    build_markowitz_layer,
    covariance_to_risk_factor,
)


def test_risk_factor_reconstructs_scaled_covariance() -> None:
    dtype = torch.float64
    covariance = torch.tensor(
        [[0.04, 0.01], [0.01, 0.09]],
        dtype=dtype,
    )
    risk_aversion = 3.0

    risk_factor = covariance_to_risk_factor(
        covariance,
        risk_aversion=risk_aversion,
    )
    reconstructed = risk_factor.T @ risk_factor

    assert torch.allclose(
        reconstructed,
        risk_aversion * covariance,
        atol=1e-12,
    )


def test_markowitz_layer_satisfies_constraints() -> None:
    dtype = torch.float64
    layer = build_markowitz_layer(
        n_assets=3,
        max_weight=0.6,
    )
    expected_returns = torch.tensor(
        [[0.02, 0.01, -0.01]],
        dtype=dtype,
    )
    covariance = torch.tensor(
        [[
            [0.04, 0.01, 0.00],
            [0.01, 0.05, 0.01],
            [0.00, 0.01, 0.06],
        ]],
        dtype=dtype,
    )
    risk_factor = covariance_to_risk_factor(
        covariance,
        risk_aversion=10.0,
    )
    weights, = layer(expected_returns, risk_factor)

    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(1, dtype=dtype),
        atol=1e-5,
    )
    assert weights.min() >= -1e-5
    assert weights.max() <= 0.6 + 1e-5
