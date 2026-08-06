from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CAPMResult:
    """Rolling-window CAPM estimation result."""

    alpha: torch.Tensor
    beta: torch.Tensor
    residuals: torch.Tensor
    fitted_returns: torch.Tensor


def _validate_inputs(
    asset_returns: torch.Tensor,
    market_returns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if asset_returns.ndim not in {2, 3}:
        raise ValueError(
            "asset_returns shape은 [T, N] 또는 [B, T, N]이어야 합니다. "
            f"현재 shape={tuple(asset_returns.shape)}"
        )
    if market_returns.ndim not in {1, 2}:
        raise ValueError(
            "market_returns shape은 [T] 또는 [B, T]이어야 합니다. "
            f"현재 shape={tuple(market_returns.shape)}"
        )
    squeeze_batch = asset_returns.ndim == 2
    assets = asset_returns.unsqueeze(0) if squeeze_batch else asset_returns
    market = market_returns.unsqueeze(0) if market_returns.ndim == 1 else market_returns
    if assets.shape[:2] != market.shape:
        raise ValueError(
            "asset_returns와 market_returns의 batch/time 차원이 일치해야 합니다. "
            f"asset_returns={tuple(asset_returns.shape)}, "
            f"market_returns={tuple(market_returns.shape)}"
        )
    if assets.size(1) < 2:
        raise ValueError("CAPM 추정을 위해 time window는 2 이상이어야 합니다.")
    if not torch.is_floating_point(assets) or not torch.is_floating_point(market):
        raise TypeError("asset_returns와 market_returns는 부동소수점 Tensor여야 합니다.")
    if not torch.isfinite(assets).all() or not torch.isfinite(market).all():
        raise ValueError("입력 수익률에 NaN 또는 inf가 존재합니다.")
    market = market.to(device=assets.device, dtype=assets.dtype)
    return assets, market, squeeze_batch


def _prepare_risk_free(
    risk_free_rates: float | torch.Tensor,
    market_returns: torch.Tensor,
) -> torch.Tensor:
    risk_free = torch.as_tensor(
        risk_free_rates,
        dtype=market_returns.dtype,
        device=market_returns.device,
    )
    try:
        return torch.broadcast_to(risk_free, market_returns.shape)
    except RuntimeError as error:
        raise ValueError(
            "risk_free_rates는 market_returns shape에 broadcast 가능해야 합니다. "
            f"risk_free_rates={tuple(risk_free.shape)}, "
            f"market_returns={tuple(market_returns.shape)}"
        ) from error


def fit_capm(
    asset_returns: torch.Tensor,
    market_returns: torch.Tensor,
    risk_free_rates: float | torch.Tensor = 0.0,
    fit_intercept: bool = True,
    eps: float = 1e-12,
) -> CAPMResult:
    """
    각 rolling window에서 자산별 CAPM을 OLS로 추정한다.

    asset_returns: [T, N] 또는 [B, T, N]
    market_returns: [T] 또는 [B, T]
    risk_free_rates: scalar 또는 market_returns에 broadcast 가능한 Tensor
    """
    if eps <= 0:
        raise ValueError(f"eps는 0보다 커야 합니다: {eps}")

    assets, market, squeeze_batch = _validate_inputs(asset_returns, market_returns)
    risk_free = _prepare_risk_free(risk_free_rates, market)
    asset_excess = assets - risk_free.unsqueeze(-1)
    market_excess = market - risk_free

    if fit_intercept:
        market_mean = market_excess.mean(dim=1, keepdim=True)
        asset_mean = asset_excess.mean(dim=1, keepdim=True)
        market_centered = market_excess - market_mean
        asset_centered = asset_excess - asset_mean
        denominator = market_centered.square().sum(dim=1)
        if torch.any(denominator <= eps):
            raise ValueError("시장 초과수익률의 분산이 0에 가까워 CAPM을 추정할 수 없습니다.")
        beta = (market_centered.unsqueeze(-1) * asset_centered).sum(dim=1) / denominator.unsqueeze(-1)
        alpha = asset_mean.squeeze(1) - beta * market_mean
    else:
        denominator = market_excess.square().sum(dim=1)
        if torch.any(denominator <= eps):
            raise ValueError("시장 초과수익률의 제곱합이 0에 가까워 CAPM을 추정할 수 없습니다.")
        beta = (market_excess.unsqueeze(-1) * asset_excess).sum(dim=1) / denominator.unsqueeze(-1)
        alpha = torch.zeros_like(beta)

    fitted_excess = alpha.unsqueeze(1) + beta.unsqueeze(1) * market_excess.unsqueeze(-1)
    residuals = asset_excess - fitted_excess
    fitted_returns = fitted_excess + risk_free.unsqueeze(-1)
    if squeeze_batch:
        return CAPMResult(
            alpha=alpha.squeeze(0),
            beta=beta.squeeze(0),
            residuals=residuals.squeeze(0),
            fitted_returns=fitted_returns.squeeze(0),
        )
    return CAPMResult(alpha=alpha, beta=beta, residuals=residuals, fitted_returns=fitted_returns)
