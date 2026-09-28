from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ScaleMode = Literal[
    "none",
    "preserve_residual_trace",
    "match_base_trace",
]


@dataclass
class EigenRiskResult:
    residual_covariance: torch.Tensor
    risk_matrix: torch.Tensor
    eigenvalues: torch.Tensor
    explained_ratio: torch.Tensor
    cumulative_explained_ratio: torch.Tensor
    retained_ratio: torch.Tensor


def covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Sample covariance.

    x:
        [T, N] 또는 [B, T, N]

    return:
        [N, N] 또는 [B, N, N]
    """
    if x.ndim not in {2, 3}:
        raise ValueError("입력 shape은 [T,N] 또는 [B,T,N]이어야 합니다.")

    n_obs = x.size(-2)

    if n_obs < 2:
        raise ValueError("covariance 계산에는 최소 2개 관측치가 필요합니다.")

    centered = x - x.mean(dim=-2, keepdim=True)

    covariance = (
        centered.transpose(-1, -2) @ centered
    ) / (n_obs - 1)

    return 0.5 * (
        covariance + covariance.transpose(-1, -2)
    )


def _trace(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.diagonal(
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)


def _rescale_trace(
    matrix: torch.Tensor,
    target_trace: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    current_trace = _trace(matrix)

    factor = (
        target_trace
        / current_trace.clamp_min(eps)
    )

    return matrix * factor[..., None, None]


def build_eigenmode_risk(
    residuals: torch.Tensor,
    k: int,
    scale_mode: ScaleMode = "none",
    base_covariance: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> EigenRiskResult:
    """
    CAPM residual covariance에서 dominant top-K eigenmode를 추출한다.

    Sigma_eps = U Lambda U^T

    A_res^(K)
        = sum_{j=1}^K lambda_j u_j u_j^T
    """
    residual_covariance = covariance_matrix(residuals)

    n_assets = residual_covariance.size(-1)

    if not 1 <= k <= n_assets:
        raise ValueError(
            f"k는 1~{n_assets} 범위여야 합니다. 현재 k={k}"
        )

    # PSD symmetric matrix용 eigendecomposition.
    # eigh는 eigenvalue를 오름차순으로 반환한다.
    eigenvalues, eigenvectors = torch.linalg.eigh(
        residual_covariance
    )

    # floating-point 오차로 발생하는 작은 음수 제거
    eigenvalues = eigenvalues.clamp_min(0.0)

    # 큰 eigenvalue부터 정렬
    eigenvalues = eigenvalues.flip(-1)
    eigenvectors = eigenvectors.flip(-1)

    total = eigenvalues.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)

    explained_ratio = eigenvalues / total

    cumulative_explained_ratio = (
        explained_ratio.cumsum(dim=-1)
    )

    if k == n_assets:
        # Full eigenmode는 residual covariance와 수학적으로 동일하므로
        # 재구성 과정의 불필요한 floating-point 오차를 제거한다.
        risk_matrix = residual_covariance.clone()
    else:
        top_values = eigenvalues[..., :k]
        top_vectors = eigenvectors[..., :, :k]

        risk_matrix = (
            top_vectors
            * top_values.unsqueeze(-2)
        ) @ top_vectors.transpose(-1, -2)

        risk_matrix = 0.5 * (
            risk_matrix
            + risk_matrix.transpose(-1, -2)
        )

    retained_ratio = explained_ratio[
        ..., :k
    ].sum(dim=-1)

    if scale_mode == "none":
        pass

    elif scale_mode == "preserve_residual_trace":
        if k < n_assets:
            risk_matrix = _rescale_trace(
                risk_matrix,
                _trace(residual_covariance),
                eps,
            )

    elif scale_mode == "match_base_trace":
        if base_covariance is None:
            raise ValueError(
                "scale_mode='match_base_trace'에는 "
                "base_covariance가 필요합니다."
            )

        risk_matrix = _rescale_trace(
            risk_matrix,
            _trace(base_covariance),
            eps,
        )

    else:
        raise ValueError(
            f"지원하지 않는 scale_mode입니다: {scale_mode}"
        )

    return EigenRiskResult(
        residual_covariance=residual_covariance,
        risk_matrix=risk_matrix,
        eigenvalues=eigenvalues,
        explained_ratio=explained_ratio,
        cumulative_explained_ratio=cumulative_explained_ratio,
        retained_ratio=retained_ratio,
    )