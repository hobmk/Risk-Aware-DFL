from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CAPMResult:
    alpha: torch.Tensor
    beta: torch.Tensor
    residuals: torch.Tensor


def fit_capm(
    asset_returns: torch.Tensor,
    market_returns: torch.Tensor,
    risk_free_rate: float = 0.0,
    fit_intercept: bool = True,
    eps: float = 1e-12,
) -> CAPMResult:
    """
    Rolling window CAPM.

    asset_returns:
        [T, N] 또는 [B, T, N]

    market_returns:
        [T] 또는 [B, T]

    residuals:
        asset excess return 중 market factor로 설명되지 않는 부분
    """
    single = asset_returns.ndim == 2

    if single:
        asset_returns = asset_returns.unsqueeze(0)
        market_returns = market_returns.unsqueeze(0)

    if asset_returns.ndim != 3:
        raise ValueError("asset_returns shape은 [T,N] 또는 [B,T,N]이어야 합니다.")

    if market_returns.ndim != 2:
        raise ValueError("market_returns shape은 [T] 또는 [B,T]이어야 합니다.")

    if asset_returns.shape[:2] != market_returns.shape:
        raise ValueError("asset_returns와 market_returns의 batch/time 차원이 다릅니다.")

    asset_excess = asset_returns - risk_free_rate
    market_excess = market_returns - risk_free_rate

    x = market_excess.unsqueeze(-1)

    if fit_intercept:
        x_mean = x.mean(dim=-2, keepdim=True)
        y_mean = asset_excess.mean(dim=-2, keepdim=True)

        x_centered = x - x_mean
        y_centered = asset_excess - y_mean

        denominator = (x_centered ** 2).sum(dim=-2).clamp_min(eps)

        beta = (
            x_centered * y_centered
        ).sum(dim=-2) / denominator

        alpha = (
            y_mean.squeeze(-2)
            - beta * x_mean.squeeze(-2)
        )
    else:
        denominator = (x ** 2).sum(dim=-2).clamp_min(eps)

        beta = (
            x * asset_excess
        ).sum(dim=-2) / denominator

        alpha = torch.zeros_like(beta)

    fitted = (
        alpha.unsqueeze(-2)
        + x * beta.unsqueeze(-2)
    )

    residuals = asset_excess - fitted

    if single:
        alpha = alpha.squeeze(0)
        beta = beta.squeeze(0)
        residuals = residuals.squeeze(0)

    return CAPMResult(
        alpha=alpha,
        beta=beta,
        residuals=residuals,
    )