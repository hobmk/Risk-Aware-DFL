from __future__ import annotations

import math

import torch


def calculate_portfolio_metrics(
    portfolio_returns: torch.Tensor,
    periods_per_year: int = 252,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """
    일간 포트폴리오 수익률로 성능 지표를 계산한다.

    Returns
    -------
    metrics:
        Final wealth, 연환산 수익률, 변동성, Sharpe, MDD

    wealth:
        초기자산 1 기준 누적 wealth 시계열

    drawdown:
        날짜별 drawdown 시계열
    """
    returns = portfolio_returns.detach().cpu().to(torch.float64).flatten()

    if returns.numel() == 0:
        raise ValueError("portfolio_returns가 비어 있습니다.")

    wealth = torch.cumprod(1.0 + returns, dim=0)
    running_max = torch.cummax(wealth, dim=0).values
    drawdown = 1.0 - wealth / running_max

    final_wealth = wealth[-1].item()
    n_periods = returns.numel()

    if final_wealth > 0:
        annualized_return_cagr = (
            final_wealth ** (periods_per_year / n_periods)
            - 1.0
        )
    else:
        annualized_return_cagr = float("nan")

    mean_return = returns.mean().item()

    if n_periods > 1:
        return_std = returns.std(unbiased=True).item()
    else:
        return_std = 0.0

    annualized_return_mean = periods_per_year * mean_return
    annualized_volatility = math.sqrt(periods_per_year) * return_std

    if return_std > 0:
        sharpe_ratio = (
            math.sqrt(periods_per_year)
            * mean_return
            / return_std
        )
    else:
        sharpe_ratio = float("nan")

    maximum_drawdown = drawdown.max().item()

    metrics = {
        "final_wealth": final_wealth,
        "annualized_return_cagr": annualized_return_cagr,
        "annualized_return_mean": annualized_return_mean,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown": maximum_drawdown,
    }

    return metrics, wealth, drawdown
