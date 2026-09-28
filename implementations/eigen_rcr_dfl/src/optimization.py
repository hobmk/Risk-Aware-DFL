from __future__ import annotations

import math

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


def build_markowitz_layer(
    n_assets: int,
    max_weight: float = 0.2,
) -> CvxpyLayer:
    if n_assets < 2:
        raise ValueError("n_assets는 2 이상이어야 합니다.")

    if not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight는 0보다 크고 1 이하여야 합니다.")

    if n_assets * max_weight < 1.0 - 1e-12:
        raise ValueError(
            "max_weight 제약으로 비중 합계 1을 만들 수 없습니다."
        )

    w = cp.Variable(n_assets)

    mu = cp.Parameter(n_assets)
    risk_factor = cp.Parameter((n_assets, n_assets))

    objective = cp.Minimize(
        cp.sum_squares(risk_factor @ w)
        - mu @ w
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
    ]

    problem = cp.Problem(
        objective,
        constraints,
    )

    if not problem.is_dpp():
        raise RuntimeError("Markowitz problem이 DPP 조건을 만족하지 않습니다.")

    return CvxpyLayer(
        problem,
        parameters=[mu, risk_factor],
        variables=[w],
    )


def covariance_to_risk_factor(
    covariance: torch.Tensor,
    risk_aversion: float = 1.0,
    jitter: float = 0.0,
) -> torch.Tensor:
    if covariance.ndim not in {2, 3}:
        raise ValueError(
            "covariance는 [N,N] 또는 [B,N,N]이어야 합니다."
        )

    if risk_aversion <= 0:
        raise ValueError("risk_aversion은 0보다 커야 합니다.")

    symmetric = 0.5 * (
        covariance
        + covariance.transpose(-1, -2)
    )

    if jitter > 0:
        eye = torch.eye(
            symmetric.size(-1),
            dtype=symmetric.dtype,
            device=symmetric.device,
        )

        symmetric = symmetric + jitter * eye

    try:
        chol = torch.linalg.cholesky(symmetric)

    except RuntimeError as error:
        min_eigenvalue = torch.linalg.eigvalsh(
            symmetric.detach()
        ).min().item()

        raise RuntimeError(
            "Effective covariance Cholesky 분해 실패. "
            f"minimum eigenvalue={min_eigenvalue:.6e}"
        ) from error

    return (
        math.sqrt(risk_aversion)
        * chol.transpose(-1, -2)
    )