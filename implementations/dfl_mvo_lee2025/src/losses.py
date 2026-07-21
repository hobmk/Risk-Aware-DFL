from __future__ import annotations

from typing import Literal

import torch


Reduction = Literal[
    "none",
    "mean",
    "sum",
]


def markowitz_cost(
    weights: torch.Tensor,
    true_returns: torch.Tensor,
    risk_factor: torch.Tensor,
) -> torch.Tensor:
    """
    실제 수익률 기준 Markowitz 비용함수를 계산한다.

        f(w, mu_true)
        = lambda * w.T Sigma w - mu_true.T w
    """
    if weights.shape != true_returns.shape:
        raise ValueError(
            "weights와 true_returns shape이 일치해야 합니다. "
            f"weights={tuple(weights.shape)}, "
            f"true_returns={tuple(true_returns.shape)}"
        )

    realized_return = torch.sum(
        true_returns * weights,
        dim=-1,
    )

    transformed_weights = torch.matmul(
        risk_factor,
        weights.unsqueeze(-1),
    ).squeeze(-1)

    risk = torch.sum(
        transformed_weights.square(),
        dim=-1,
    )

    return risk - realized_return


def mvo_regret(
    predicted_weights: torch.Tensor,
    oracle_weights: torch.Tensor,
    true_returns: torch.Tensor,
    risk_factor: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    MVO Regret을 계산한다.

        Regret
        = f(w_pred, mu_true)
        - f(w_oracle, mu_true)
    """
    predicted_cost = markowitz_cost(
        weights=predicted_weights,
        true_returns=true_returns,
        risk_factor=risk_factor,
    )

    oracle_cost = markowitz_cost(
        weights=oracle_weights,
        true_returns=true_returns,
        risk_factor=risk_factor,
    )

    regret = predicted_cost - oracle_cost

    if reduction == "none":
        return regret

    if reduction == "mean":
        return regret.mean()

    if reduction == "sum":
        return regret.sum()

    raise ValueError(
        f"지원하지 않는 reduction입니다: {reduction}"
    )


def combined_loss(
    regret_loss: torch.Tensor,
    mse_loss: torch.Tensor,
    alpha: float,
    mse_scale: float = 1.0,
) -> torch.Tensor:
    """
    L_combined
    = alpha * L_MVO
    + (1 - alpha) * mse_scale * L_MSE

    regret_loss는 CPU float64이고 mse_loss는 GPU float32일 수 있으므로,
    MSE를 regret_loss와 같은 device와 dtype으로 변환한다.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"alpha는 0 이상 1 이하여야 합니다: {alpha}"
        )

    if mse_scale <= 0:
        raise ValueError(
            f"mse_scale은 0보다 커야 합니다: {mse_scale}"
        )

    mse_solver = mse_loss.to(
        device=regret_loss.device,
        dtype=regret_loss.dtype,
    )

    return (
        alpha * regret_loss
        + (1.0 - alpha) * mse_scale * mse_solver
    )
