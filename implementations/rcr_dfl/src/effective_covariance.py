from __future__ import annotations

import torch


def _validate_square_matrices(
    covariance: torch.Tensor,
    residual_risk_matrix: torch.Tensor,
) -> None:
    if covariance.shape != residual_risk_matrix.shape:
        raise ValueError(
            "covariance와 residual_risk_matrix shape이 일치해야 합니다. "
            f"covariance={tuple(covariance.shape)}, "
            f"residual_risk_matrix={tuple(residual_risk_matrix.shape)}"
        )
    if covariance.ndim not in {2, 3} or covariance.size(-1) != covariance.size(-2):
        raise ValueError("행렬 shape은 [N, N] 또는 [B, N, N]이어야 합니다.")
    if not torch.is_floating_point(covariance) or not torch.is_floating_point(residual_risk_matrix):
        raise TypeError("입력 행렬은 부동소수점 Tensor여야 합니다.")
    if not torch.isfinite(covariance).all() or not torch.isfinite(residual_risk_matrix).all():
        raise ValueError("입력 행렬에 NaN 또는 inf가 존재합니다.")


def project_to_psd(matrix: torch.Tensor, minimum_eigenvalue: float = 0.0) -> torch.Tensor:
    """고유값 clipping으로 대칭행렬을 PSD 영역에 투영한다."""
    if matrix.ndim not in {2, 3} or matrix.size(-1) != matrix.size(-2):
        raise ValueError("matrix shape은 [N, N] 또는 [B, N, N]이어야 합니다.")
    if minimum_eigenvalue < 0:
        raise ValueError("minimum_eigenvalue는 0 이상이어야 합니다.")
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    clipped = eigenvalues.clamp_min(minimum_eigenvalue)
    projected = eigenvectors @ torch.diag_embed(clipped) @ eigenvectors.transpose(-1, -2)
    return 0.5 * (projected + projected.transpose(-1, -2))


def build_effective_covariance(
    covariance: torch.Tensor,
    residual_risk_matrix: torch.Tensor,
    eta: float,
    jitter: float = 0.0,
    project_psd: bool = False,
    minimum_eigenvalue: float = 0.0,
) -> torch.Tensor:
    """
    Sigma_eff = Sigma + eta * A_res + jitter * I.

    residual_risk_matrix는 residual correlation을 covariance 단위로 변환한 A_res이다.
    따라서 이 함수 안에서는 추가 normalization을 수행하지 않는다.
    """
    _validate_square_matrices(covariance, residual_risk_matrix)
    if eta < 0:
        raise ValueError(f"eta는 0 이상이어야 합니다: {eta}")
    if jitter < 0:
        raise ValueError(f"jitter는 0 이상이어야 합니다: {jitter}")
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    residual_risk_matrix = 0.5 * (
        residual_risk_matrix + residual_risk_matrix.transpose(-1, -2)
    )
    effective = covariance + eta * residual_risk_matrix
    if jitter > 0:
        eye = torch.eye(effective.size(-1), dtype=effective.dtype, device=effective.device)
        effective = effective + jitter * eye
    effective = 0.5 * (effective + effective.transpose(-1, -2))
    if project_psd:
        effective = project_to_psd(effective, minimum_eigenvalue=minimum_eigenvalue)
    return effective
