from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class EigenRCRLossOutput:
    total: torch.Tensor
    mse: torch.Tensor
    regret: torch.Tensor
    predicted_cost: torch.Tensor
    oracle_cost: torch.Tensor


def markowitz_cost(
    weights: torch.Tensor,
    true_returns: torch.Tensor,
    risk_factor: torch.Tensor,
) -> torch.Tensor:
    true_returns = true_returns.to(
        device=weights.device,
        dtype=weights.dtype,
    )

    transformed = torch.matmul(
        risk_factor,
        weights.unsqueeze(-1),
    ).squeeze(-1)

    risk = transformed.square().sum(dim=-1)

    realized_return = (
        true_returns
        * weights
    ).sum(dim=-1)

    return risk - realized_return


def compute_losses(
    predicted_returns: torch.Tensor,
    predicted_weights: torch.Tensor,
    oracle_weights: torch.Tensor,
    true_returns: torch.Tensor,
    risk_factor: torch.Tensor,
    alpha: float,
    mse_scale: float,
) -> EigenRCRLossOutput:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha는 0 이상 1 이하여야 합니다.")

    if mse_scale <= 0:
        raise ValueError("mse_scale은 0보다 커야 합니다.")

    predicted_cost = markowitz_cost(
        predicted_weights,
        true_returns,
        risk_factor,
    )

    oracle_cost = markowitz_cost(
        oracle_weights,
        true_returns,
        risk_factor,
    )

    regret = (
        predicted_cost
        - oracle_cost
    ).mean()

    mse = F.mse_loss(
        predicted_returns,
        true_returns,
    )

    mse_solver = mse.to(
        device=regret.device,
        dtype=regret.dtype,
    )

    total = (
        alpha * regret
        + (1.0 - alpha)
        * mse_scale
        * mse_solver
    )

    return EigenRCRLossOutput(
        total=total,
        mse=mse,
        regret=regret,
        predicted_cost=predicted_cost,
        oracle_cost=oracle_cost,
    )