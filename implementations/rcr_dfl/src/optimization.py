from __future__ import annotations

import math

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


def build_markowitz_layer(n_assets: int, max_weight: float = 1.0) -> CvxpyLayer:
    """
    min_w ||Lw||_2^2 - mu^T w
    s.t. sum(w)=1, 0<=w_i<=max_weight.
    """
    if n_assets < 2:
        raise ValueError(f"n_assets는 2 이상이어야 합니다: {n_assets}")
    if not 0.0 < max_weight <= 1.0:
        raise ValueError(f"max_weight는 0보다 크고 1 이하여야 합니다: {max_weight}")
    if n_assets * max_weight < 1.0 - 1e-12:
        raise ValueError(
            "현재 max_weight로는 비중 합계 1을 만들 수 없습니다. "
            f"n_assets={n_assets}, max_weight={max_weight}"
        )

    weights = cp.Variable(n_assets)
    expected_returns = cp.Parameter(n_assets)
    risk_factor = cp.Parameter((n_assets, n_assets))
    objective = cp.Minimize(cp.sum_squares(risk_factor @ weights) - expected_returns @ weights)
    constraints = [cp.sum(weights) == 1, weights >= 0, weights <= max_weight]
    problem = cp.Problem(objective, constraints)
    if not problem.is_dpp():
        raise ValueError("Markowitz problem이 DPP 조건을 만족하지 않습니다.")
    return CvxpyLayer(
        problem,
        parameters=[expected_returns, risk_factor],
        variables=[weights],
    )


def covariance_to_risk_factor(
    covariance: torch.Tensor,
    risk_aversion: float,
    jitter: float = 0.0,
) -> torch.Tensor:
    """L^T L = risk_aversion * covariance를 만족하는 upper-triangular L을 반환한다."""
    if covariance.ndim not in {2, 3}:
        raise ValueError(
            "covariance shape은 [N, N] 또는 [B, N, N]이어야 합니다. "
            f"현재 shape={tuple(covariance.shape)}"
        )
    if covariance.size(-1) != covariance.size(-2):
        raise ValueError(f"covariance는 정방행렬이어야 합니다: {tuple(covariance.shape)}")
    if not torch.is_floating_point(covariance):
        raise TypeError("covariance는 부동소수점 Tensor여야 합니다.")
    if not torch.isfinite(covariance).all():
        raise ValueError("covariance에 NaN 또는 inf가 존재합니다.")
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion은 0보다 커야 합니다: {risk_aversion}")
    if jitter < 0:
        raise ValueError(f"jitter는 0 이상이어야 합니다: {jitter}")

    symmetric_covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    if jitter > 0:
        eye = torch.eye(covariance.size(-1), dtype=covariance.dtype, device=covariance.device)
        symmetric_covariance = symmetric_covariance + jitter * eye
    try:
        cholesky = torch.linalg.cholesky(symmetric_covariance)
    except RuntimeError as error:
        minimum_eigenvalue = torch.linalg.eigvalsh(symmetric_covariance.detach()).min().item()
        raise RuntimeError(
            "Covariance의 Cholesky 분해에 실패했습니다. Dataset covariance_jitter, "
            "effective covariance 또는 PSD 처리를 확인하세요. "
            f"minimum_eigenvalue={minimum_eigenvalue:.6e}"
        ) from error
    return math.sqrt(risk_aversion) * cholesky.transpose(-1, -2)
