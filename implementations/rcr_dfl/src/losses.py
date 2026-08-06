from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

Reduction = Literal["none", "mean", "sum"]


@dataclass(frozen=True)
class RCRLossOutput:
    total: torch.Tensor
    regret: torch.Tensor
    mse: torch.Tensor
    predicted_cost: torch.Tensor
    oracle_cost: torch.Tensor


def markowitz_cost(
    weights: torch.Tensor,
    true_returns: torch.Tensor,
    risk_factor: torch.Tensor,
) -> torch.Tensor:
    """
    실제 수익률 기준 Markowitz cost.

    f(w, mu_true) = lambda * w^T Sigma_eff w - mu_true^T w
                  = ||Lw||_2^2 - mu_true^T w
    """
    if weights.shape != true_returns.shape:
        raise ValueError(
            "weights와 true_returns shape이 일치해야 합니다. "
            f"weights={tuple(weights.shape)}, true_returns={tuple(true_returns.shape)}"
        )
    if risk_factor.shape[-2:] != (weights.size(-1), weights.size(-1)):
        raise ValueError(
            "risk_factor의 마지막 두 차원은 [N, N]이어야 합니다. "
            f"weights={tuple(weights.shape)}, risk_factor={tuple(risk_factor.shape)}"
        )
    realized_return = torch.sum(true_returns * weights, dim=-1)
    transformed_weights = torch.matmul(risk_factor, weights.unsqueeze(-1)).squeeze(-1)
    risk = torch.sum(transformed_weights.square(), dim=-1)
    return risk - realized_return


def rcr_regret(
    predicted_weights: torch.Tensor,
    oracle_weights: torch.Tensor,
    true_returns: torch.Tensor,
    risk_factor: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Sigma_eff에 기반한 predicted portfolio와 oracle portfolio의 cost 차이."""
    predicted_cost = markowitz_cost(predicted_weights, true_returns, risk_factor)
    oracle_cost = markowitz_cost(oracle_weights, true_returns, risk_factor)
    regret = predicted_cost - oracle_cost
    if reduction == "none":
        return regret
    if reduction == "mean":
        return regret.mean()
    if reduction == "sum":
        return regret.sum()
    raise ValueError(f"지원하지 않는 reduction입니다: {reduction}")


def combined_loss(
    regret_loss: torch.Tensor,
    mse_loss: torch.Tensor,
    alpha: float,
    mse_scale: float = 1.0,
) -> torch.Tensor:
    """L = alpha * L_RCR + (1-alpha) * mse_scale * L_MSE."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha는 0 이상 1 이하여야 합니다: {alpha}")
    if mse_scale <= 0:
        raise ValueError(f"mse_scale은 0보다 커야 합니다: {mse_scale}")
    mse_solver = mse_loss.to(device=regret_loss.device, dtype=regret_loss.dtype)
    return alpha * regret_loss + (1.0 - alpha) * mse_scale * mse_solver


def compute_rcr_losses(
    predicted_returns: torch.Tensor,
    predicted_weights: torch.Tensor,
    oracle_weights: torch.Tensor,
    true_returns: torch.Tensor,
    risk_factor: torch.Tensor,
    alpha: float,
    mse_scale: float,
) -> RCRLossOutput:
    """한 batch의 MSE, RCR regret, combined loss를 계산한다."""
    true_returns_solver = true_returns.to(
        device=predicted_weights.device,
        dtype=predicted_weights.dtype,
    )
    predicted_cost = markowitz_cost(predicted_weights, true_returns_solver, risk_factor)
    oracle_cost = markowitz_cost(oracle_weights, true_returns_solver, risk_factor)
    regret = (predicted_cost - oracle_cost).mean()
    mse = F.mse_loss(predicted_returns, true_returns)
    total = combined_loss(regret, mse, alpha=alpha, mse_scale=mse_scale)
    return RCRLossOutput(
        total=total,
        regret=regret,
        mse=mse,
        predicted_cost=predicted_cost,
        oracle_cost=oracle_cost,
    )
