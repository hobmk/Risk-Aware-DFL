from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import EigenRCRDFL


TRADING_DAYS = 252


def portfolio_metrics(
    portfolio_returns: np.ndarray,
    weights: np.ndarray,
    active_threshold: float = 1e-6,
) -> dict[str, float]:
    if portfolio_returns.ndim != 1:
        raise ValueError("portfolio_returns는 1차원이어야 합니다.")

    if weights.ndim != 2:
        raise ValueError("weights는 [T, N]이어야 합니다.")

    if len(portfolio_returns) != len(weights):
        raise ValueError("return과 weight의 기간 길이가 다릅니다.")

    n_periods = len(portfolio_returns)

    wealth = np.cumprod(
        1.0 + portfolio_returns
    )

    cumulative_return = (
        wealth[-1] - 1.0
    )

    if wealth[-1] > 0:
        annualized_return = (
            wealth[-1] ** (
                TRADING_DAYS / n_periods
            ) - 1.0
        )
    else:
        annualized_return = np.nan

    if n_periods > 1:
        volatility = (
            np.std(
                portfolio_returns,
                ddof=1,
            )
            * math.sqrt(TRADING_DAYS)
        )
    else:
        volatility = np.nan

    daily_std = np.std(
        portfolio_returns,
        ddof=1,
    )

    if daily_std > 0:
        sharpe = (
            np.mean(portfolio_returns)
            / daily_std
            * math.sqrt(TRADING_DAYS)
        )
    else:
        sharpe = np.nan

    running_max = np.maximum.accumulate(
        wealth
    )

    drawdown = (
        wealth / running_max
        - 1.0
    )

    max_drawdown = (
        -np.min(drawdown)
    )

    if max_drawdown > 0:
        calmar = (
            annualized_return
            / max_drawdown
        )
    else:
        calmar = np.nan

    if len(weights) > 1:
        daily_turnover = (
            0.5
            * np.abs(
                weights[1:]
                - weights[:-1]
            ).sum(axis=1)
        )

        mean_turnover = (
            daily_turnover.mean()
        )

        annualized_turnover = (
            mean_turnover
            * TRADING_DAYS
        )
    else:
        mean_turnover = 0.0
        annualized_turnover = 0.0

    active_assets = (
        weights > active_threshold
    ).sum(axis=1)

    return {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "mean_turnover": float(mean_turnover),
        "annualized_turnover": float(annualized_turnover),
        "active_assets": float(active_assets.mean()),
        "n_periods": int(n_periods),
    }


def evaluate_portfolio(
    model: EigenRCRDFL,
    dataloader: DataLoader,
    device: torch.device,
    tickers: list[str],
    output_dir: str | Path,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    dates = []
    all_weights = []
    all_predictions = []
    all_targets = []

    for batch in dataloader:
        features = batch["features"].to(
            device=device,
            dtype=torch.float32,
        )

        covariance = batch["covariance"].to(
            device="cpu",
            dtype=torch.float64,
        )

        eigen_risk = batch["eigen_risk"].to(
            device="cpu",
            dtype=torch.float64,
        )

        with torch.no_grad():
            output = model(
                features=features,
                covariance=covariance,
                eigen_risk=eigen_risk,
            )

        weights = (
            output.predicted_weights
            .detach()
            .cpu()
            .numpy()
        )

        predictions = (
            output.predicted_returns
            .detach()
            .cpu()
            .numpy()
        )

        targets = (
            batch["target"]
            .detach()
            .cpu()
            .numpy()
        )

        all_weights.append(weights)
        all_predictions.append(predictions)
        all_targets.append(targets)

        dates.extend(
            list(batch["target_date"])
        )

    weights = np.concatenate(
        all_weights,
        axis=0,
    )

    predictions = np.concatenate(
        all_predictions,
        axis=0,
    )

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    portfolio_returns = np.sum(
        weights * targets,
        axis=1,
    )

    wealth = np.cumprod(
        1.0 + portfolio_returns
    )

    running_max = np.maximum.accumulate(
        wealth
    )

    drawdown = (
        wealth / running_max
        - 1.0
    )

    metrics = portfolio_metrics(
        portfolio_returns=portfolio_returns,
        weights=weights,
    )

    # ---------------------------------------------------------
    # Daily portfolio result
    # ---------------------------------------------------------
    daily_df = pd.DataFrame({
        "date": dates,
        "portfolio_return": portfolio_returns,
        "wealth": wealth,
        "drawdown": drawdown,
    })

    daily_df.to_csv(
        output_dir / "portfolio_daily.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Portfolio weights
    # ---------------------------------------------------------
    weights_df = pd.DataFrame(
        weights,
        columns=tickers,
    )

    weights_df.insert(
        0,
        "date",
        dates,
    )

    weights_df.to_csv(
        output_dir / "portfolio_weights.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Predicted returns
    # ---------------------------------------------------------
    predictions_df = pd.DataFrame(
        predictions,
        columns=tickers,
    )

    predictions_df.insert(
        0,
        "date",
        dates,
    )

    predictions_df.to_csv(
        output_dir / "predicted_returns.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # True returns
    # ---------------------------------------------------------
    targets_df = pd.DataFrame(
        targets,
        columns=tickers,
    )

    targets_df.insert(
        0,
        "date",
        dates,
    )

    targets_df.to_csv(
        output_dir / "true_returns.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    summary_df = pd.DataFrame(
        [metrics]
    )

    summary_df.to_csv(
        output_dir / "portfolio_summary.csv",
        index=False,
    )

    return metrics