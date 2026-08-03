import torch

from implementations.rcr_dfl.src.losses import (
    combined_loss,
    compute_rcr_losses,
    markowitz_cost,
    rcr_regret,
)


def test_markowitz_cost_matches_direct_formula() -> None:
    dtype = torch.float64
    weights = torch.tensor([[0.25, 0.75]], dtype=dtype)
    true_returns = torch.tensor([[0.01, 0.02]], dtype=dtype)
    risk_factor = torch.tensor(
        [[[2.0, 0.0], [0.0, 3.0]]],
        dtype=dtype,
    )

    actual = markowitz_cost(weights, true_returns, risk_factor)
    direct = (
        (risk_factor @ weights.unsqueeze(-1))
        .squeeze(-1)
        .square()
        .sum(dim=-1)
        - (weights * true_returns).sum(dim=-1)
    )
    assert torch.allclose(actual, direct, atol=1e-12)


def test_regret_is_zero_for_identical_portfolios() -> None:
    dtype = torch.float64
    weights = torch.tensor([[0.4, 0.6]], dtype=dtype)
    true_returns = torch.tensor([[0.01, -0.01]], dtype=dtype)
    risk_factor = torch.eye(2, dtype=dtype).unsqueeze(0)

    regret = rcr_regret(
        predicted_weights=weights,
        oracle_weights=weights,
        true_returns=true_returns,
        risk_factor=risk_factor,
    )
    assert torch.allclose(regret, torch.zeros_like(regret), atol=1e-12)


def test_combined_loss_uses_benchmark_alpha_convention() -> None:
    regret = torch.tensor(2.0, dtype=torch.float64)
    mse = torch.tensor(3.0, dtype=torch.float32)

    pure_mse = combined_loss(regret, mse, alpha=0.0, mse_scale=10.0)
    pure_rcr = combined_loss(regret, mse, alpha=1.0, mse_scale=10.0)

    assert torch.allclose(
        pure_mse,
        torch.tensor(30.0, dtype=torch.float64),
    )
    assert torch.allclose(
        pure_rcr,
        torch.tensor(2.0, dtype=torch.float64),
    )


def test_compute_rcr_losses_returns_all_components() -> None:
    predicted_returns = torch.tensor([[0.01, 0.02]], dtype=torch.float32)
    true_returns = torch.tensor([[0.02, 0.01]], dtype=torch.float32)
    predicted_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float64)
    oracle_weights = torch.tensor([[0.4, 0.6]], dtype=torch.float64)
    risk_factor = torch.eye(2, dtype=torch.float64).unsqueeze(0)

    output = compute_rcr_losses(
        predicted_returns=predicted_returns,
        predicted_weights=predicted_weights,
        oracle_weights=oracle_weights,
        true_returns=true_returns,
        risk_factor=risk_factor,
        alpha=0.5,
        mse_scale=15.0,
    )

    assert output.total.ndim == 0
    assert output.regret.ndim == 0
    assert output.mse.ndim == 0
    assert output.predicted_cost.shape == (1,)
    assert output.oracle_cost.shape == (1,)
