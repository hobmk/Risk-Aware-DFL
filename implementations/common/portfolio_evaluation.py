from __future__ import annotations

import math
from typing import Iterable

import numpy as np


TRADING_DAYS = 252


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _validate(
    portfolio_returns: np.ndarray,
    weights: np.ndarray,
    asset_returns: np.ndarray | None,
) -> None:
    if portfolio_returns.ndim != 1:
        raise ValueError("portfolio_returns는 [T]여야 합니다.")
    if weights.ndim != 2:
        raise ValueError("weights는 [T, N]이어야 합니다.")
    if len(portfolio_returns) != len(weights):
        raise ValueError("portfolio_returns와 weights의 T가 다릅니다.")

    if asset_returns is not None:
        if asset_returns.ndim != 2:
            raise ValueError("asset_returns는 [T, N]이어야 합니다.")
        if asset_returns.shape != weights.shape:
            raise ValueError("asset_returns와 weights의 shape이 다릅니다.")


def naive_turnover(weights: np.ndarray) -> np.ndarray:
    weights = _to_numpy(weights)
    turnover = np.zeros(len(weights), dtype=np.float64)

    if len(weights) > 1:
        turnover[1:] = 0.5 * np.abs(weights[1:] - weights[:-1]).sum(axis=1)

    return turnover


def drift_adjusted_turnover(
    weights: np.ndarray,
    asset_returns: np.ndarray,
) -> np.ndarray:
    weights = _to_numpy(weights)
    asset_returns = _to_numpy(asset_returns)

    if weights.shape != asset_returns.shape:
        raise ValueError("weights와 asset_returns의 shape이 다릅니다.")

    turnover = np.zeros(len(weights), dtype=np.float64)

    for t in range(1, len(weights)):
        prev_w = weights[t - 1]
        prev_r = asset_returns[t - 1]

        denom = 1.0 + float(np.dot(prev_w, prev_r))

        if abs(denom) < 1e-12:
            drifted = prev_w.copy()
        else:
            drifted = prev_w * (1.0 + prev_r) / denom

        turnover[t] = 0.5 * np.abs(weights[t] - drifted).sum()

    return turnover


def concentration_metrics(
    weights: np.ndarray,
    active_threshold: float = 1e-3,
) -> dict[str, float]:
    weights = _to_numpy(weights)

    active_assets = (weights > active_threshold).sum(axis=1)

    hhi = np.square(weights).sum(axis=1)

    effective_n = np.divide(
        1.0,
        hhi,
        out=np.full_like(hhi, np.nan),
        where=hhi > 0,
    )

    return {
        "active_assets": float(active_assets.mean()),
        "hhi": float(hhi.mean()),
        "effective_n": float(np.nanmean(effective_n)),
    }


def performance_metrics(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
    cvar_level: float = 0.95,
) -> dict[str, float]:
    returns = _to_numpy(returns).reshape(-1)

    if len(returns) == 0:
        raise ValueError("returns가 비어 있습니다.")

    wealth = np.cumprod(1.0 + returns)
    cumulative_return = float(wealth[-1] - 1.0)

    if wealth[-1] > 0:
        annualized_return = float(
            wealth[-1] ** (periods_per_year / len(returns)) - 1.0
        )
    else:
        annualized_return = np.nan

    if len(returns) > 1:
        daily_std = float(np.std(returns, ddof=1))
        volatility = daily_std * math.sqrt(periods_per_year)
    else:
        daily_std = np.nan
        volatility = np.nan

    mean_daily = float(np.mean(returns))

    if np.isfinite(daily_std) and daily_std > 0:
        sharpe = mean_daily / daily_std * math.sqrt(periods_per_year)
    else:
        sharpe = np.nan

    downside = np.minimum(returns, 0.0)
    downside_daily = float(np.sqrt(np.mean(np.square(downside))))
    downside_deviation = downside_daily * math.sqrt(periods_per_year)

    if downside_daily > 0:
        sortino = mean_daily / downside_daily * math.sqrt(periods_per_year)
    else:
        sortino = np.nan

    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    max_drawdown = float(-np.min(drawdown))

    if max_drawdown > 0 and np.isfinite(annualized_return):
        calmar = annualized_return / max_drawdown
    else:
        calmar = np.nan

    tail_quantile = np.quantile(returns, 1.0 - cvar_level)
    tail_returns = returns[returns <= tail_quantile]

    if len(tail_returns) > 0:
        cvar_loss = max(0.0, -float(np.mean(tail_returns)))
    else:
        cvar_loss = np.nan

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "downside_deviation": float(downside_deviation),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "cvar_95_daily": float(cvar_loss),
    }


def evaluate_portfolio(
    portfolio_returns,
    weights,
    asset_returns=None,
    cost_bps: Iterable[float] = (0.0, 5.0, 10.0),
    active_threshold: float = 1e-3,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    portfolio_returns = _to_numpy(portfolio_returns).reshape(-1)
    weights = _to_numpy(weights)

    if asset_returns is not None:
        asset_returns = _to_numpy(asset_returns)

    _validate(
        portfolio_returns=portfolio_returns,
        weights=weights,
        asset_returns=asset_returns,
    )

    naive_to = naive_turnover(weights)

    if asset_returns is not None:
        turnover = drift_adjusted_turnover(
            weights=weights,
            asset_returns=asset_returns,
        )
    else:
        turnover = naive_to

    result = concentration_metrics(
        weights=weights,
        active_threshold=active_threshold,
    )

    result.update({
        "mean_turnover_naive": float(naive_to.mean()),
        "annualized_turnover_naive": float(naive_to.mean() * periods_per_year),
        "mean_turnover": float(turnover.mean()),
        "annualized_turnover": float(turnover.mean() * periods_per_year),
    })

    for bps in cost_bps:
        cost_rate = float(bps) / 10000.0

        net_returns = (
            portfolio_returns
            - cost_rate * turnover
        )

        metrics = performance_metrics(
            returns=net_returns,
            periods_per_year=periods_per_year,
        )

        prefix = (
            "gross"
            if float(bps) == 0.0
            else f"net_{int(bps)}bp"
        )

        for key, value in metrics.items():
            result[f"{prefix}_{key}"] = value

    return result


def equal_weight_portfolio(
    asset_returns,
    cost_bps: Iterable[float] = (0.0, 5.0, 10.0),
    active_threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    asset_returns = _to_numpy(asset_returns)

    if asset_returns.ndim != 2:
        raise ValueError("asset_returns는 [T, N]이어야 합니다.")

    n_periods, n_assets = asset_returns.shape

    weights = np.full(
        (n_periods, n_assets),
        1.0 / n_assets,
        dtype=np.float64,
    )

    portfolio_returns = np.sum(
        weights * asset_returns,
        axis=1,
    )

    metrics = evaluate_portfolio(
        portfolio_returns=portfolio_returns,
        weights=weights,
        asset_returns=asset_returns,
        cost_bps=cost_bps,
        active_threshold=active_threshold,
    )

    return portfolio_returns, weights, metrics