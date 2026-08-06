import torch

from implementations.rcr_dfl.src.losses import combined_loss, markowitz_cost, rcr_regret


def test_markowitz_cost_matches_manual_value() -> None:
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    returns = torch.tensor([[0.01, 0.02]], dtype=torch.float64)
    factor = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    expected = weights.square().sum(dim=-1) - (weights * returns).sum(dim=-1)
    assert torch.allclose(markowitz_cost(weights, returns, factor), expected)


def test_regret_is_zero_for_same_portfolio() -> None:
    weights = torch.tensor([[0.4, 0.6]], dtype=torch.float64)
    returns = torch.tensor([[0.01, -0.01]], dtype=torch.float64)
    factor = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    assert rcr_regret(weights, weights, returns, factor).abs() < 1e-12


def test_combined_loss_formula() -> None:
    regret = torch.tensor(2.0, dtype=torch.float64)
    mse = torch.tensor(0.1, dtype=torch.float32)
    result = combined_loss(regret, mse, alpha=0.25, mse_scale=10.0)
    assert torch.allclose(result, torch.tensor(1.25, dtype=torch.float64))
