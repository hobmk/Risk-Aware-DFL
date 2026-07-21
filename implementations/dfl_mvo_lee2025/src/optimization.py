from __future__ import annotations

import math

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


def build_markowitz_layer(
    n_assets: int,
    max_weight: float = 1.0,
) -> CvxpyLayer:
    """
    Constrained mean-variance optimization layer를 생성한다.

    최소화 형태:
        lambda * w^T Sigma w - mu^T w

    risk_factor가 다음을 만족하도록 구성한다.
        risk_factor.T @ risk_factor = lambda * Sigma

    따라서:
        lambda * w^T Sigma w
        = ||risk_factor @ w||^2

    제약조건:
        sum(w) = 1
        0 <= w_i <= max_weight
    """
    if n_assets < 2:
        raise ValueError(
            f"n_assets는 2 이상이어야 합니다: {n_assets}"
        )

    if not 0.0 < max_weight <= 1.0:
        raise ValueError(
            "max_weight는 0보다 크고 1 이하여야 합니다: "
            f"{max_weight}"
        )

    # 모든 종목에 최대 비중을 적용했을 때
    # 합계 1을 만들 수 있는지 확인한다.
    if n_assets * max_weight < 1.0 - 1e-12:
        raise ValueError(
            "현재 max_weight로는 비중 합계 1을 만들 수 없습니다. "
            f"n_assets={n_assets}, max_weight={max_weight}"
        )

    weights = cp.Variable(n_assets)

    expected_returns = cp.Parameter(n_assets)
    risk_factor = cp.Parameter((n_assets, n_assets))

    objective = cp.Minimize(
        cp.sum_squares(risk_factor @ weights)
        - expected_returns @ weights
    )

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight,
    ]

    problem = cp.Problem(
        objective,
        constraints,
    )

    if not problem.is_dpp():
        raise ValueError(
            "Markowitz problem이 DPP 조건을 만족하지 않습니다."
        )

    return CvxpyLayer(
        problem,
        parameters=[
            expected_returns,
            risk_factor,
        ],
        variables=[weights],
    )


def covariance_to_risk_factor(
    covariance: torch.Tensor,
    risk_aversion: float,
    jitter: float = 0.0,
) -> torch.Tensor:
    """
    covariance를 MVO layer에서 사용할 risk factor로 변환한다.

    반환되는 L은 다음을 만족한다.

        L.T @ L = lambda * Sigma

    따라서:

        ||Lw||^2 = lambda * w.T Sigma w

    covariance shape:
        단일 샘플: [N, N]
        배치:      [B, N, N]
    """
    if covariance.ndim not in {2, 3}:
        raise ValueError(
            "covariance shape은 [N, N] 또는 [B, N, N]이어야 합니다. "
            f"현재 shape={tuple(covariance.shape)}"
        )

    if covariance.size(-1) != covariance.size(-2):
        raise ValueError(
            "covariance는 정방행렬이어야 합니다. "
            f"현재 shape={tuple(covariance.shape)}"
        )

    if risk_aversion <= 0:
        raise ValueError(
            "risk_aversion은 0보다 커야 합니다: "
            f"{risk_aversion}"
        )

    if jitter < 0:
        raise ValueError(
            f"jitter는 0 이상이어야 합니다: {jitter}"
        )

    # 수치오차로 발생할 수 있는 미세한 비대칭을 제거한다.
    symmetric_covariance = 0.5 * (
        covariance
        + covariance.transpose(-1, -2)
    )

    if jitter > 0:
        n_assets = covariance.size(-1)

        eye = torch.eye(
            n_assets,
            dtype=covariance.dtype,
            device=covariance.device,
        )

        symmetric_covariance = (
            symmetric_covariance
            + jitter * eye
        )

    try:
        chol = torch.linalg.cholesky(
            symmetric_covariance
        )
    except RuntimeError as error:
        raise RuntimeError(
            "Covariance의 Cholesky 분해에 실패했습니다. "
            "covariance_jitter 또는 covariance 계산을 확인하세요."
        ) from error

    return (
        math.sqrt(risk_aversion)
        * chol.transpose(-1, -2)
    )