from __future__ import annotations

from typing import Literal

import torch

from .residual_risk import normalize_covariance_trace

Normalization = Literal["none", "trace"]


def _validate_square_matrices(
    covariance: torch.Tensor,
    residual_covariance: torch.Tensor,
) -> None:
    if covariance.shape != residual_covariance.shape:
        raise ValueError(
            "covariance와 residual_covariance shape이 일치해야 합니다. "
            f"covariance={tuple(covariance.shape)}, "
            f"residual_covariance={tuple(residual_covariance.shape)}"
        )
    if covariance.ndim not in {2, 3} or covariance.size(-1) != covariance.size(-2):
        raise ValueError("행렬 shape은 [N, N] 또는 [B, N, N]이어야 합니다.")
    if not torch.is_floating_point(covariance) or not torch.is_floating_point(residual_covariance):
        raise TypeError("covariance 입력은 부동소수점 Tensor여야 합니다.")
    if not torch.isfinite(covariance).all() or not torch.isfinite(residual_covariance).all():
        raise ValueError("covariance 입력에 NaN 또는 inf가 존재합니다.")


def project_to_psd(
    matrix: torch.Tensor,
    minimum_eigenvalue: float = 0.0,
) -> torch.Tensor:
    """고유값 clipping으로 대칭행렬을 PSD 영역에 투영한다."""
    if matrix.ndim not in {2, 3} or matrix.size(-1) != matrix.size(-2):
        raise ValueError(
            "matrix shape은 [N, N] 또는 [B, N, N]이어야 합니다."
        )
    if minimum_eigenvalue < 0:
        raise ValueError("minimum_eigenvalue는 0 이상이어야 합니다.")

    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    clipped = eigenvalues.clamp_min(minimum_eigenvalue)
    projected = (
        eigenvectors
        @ torch.diag_embed(clipped)
        @ eigenvectors.transpose(-1, -2)
    )
    return 0.5 * (projected + projected.transpose(-1, -2))


def build_effective_covariance(
    covariance: torch.Tensor,
    residual_covariance: torch.Tensor,
    eta: float,
    normalization: Normalization = "trace",
    jitter: float = 0.0,
    project_psd: bool = False,
    minimum_eigenvalue: float = 0.0,
) -> torch.Tensor:
    """
    Sigma_eff = Sigma + eta * Sigma_res + jitter * I

    normalization="trace"이면 residual covariance의 trace를
    기존 covariance의 trace와 맞춘다.

    eta=0일 때 residual covariance의 trace와 무관하게
    기존 covariance를 그대로 사용한다.
    """
    _validate_square_matrices(covariance, residual_covariance)
    if eta < 0:
        raise ValueError(f"eta는 0 이상이어야 합니다: {eta}")
    if jitter < 0:
        raise ValueError(f"jitter는 0 이상이어야 합니다: {jitter}")
    if normalization not in {"none", "trace"}:
        raise ValueError(
            f"지원하지 않는 normalization입니다: {normalization}"
        )

    covariance = 0.5 * (
        covariance + covariance.transpose(-1, -2)
    )

    if eta == 0.0:
        effective = covariance
    else:
        residual_covariance = 0.5 * (
            residual_covariance
            + residual_covariance.transpose(-1, -2)
        )
        if normalization == "trace":
            residual_covariance = normalize_covariance_trace(
                covariance=residual_covariance,
                reference_covariance=covariance,
            )
        effective = covariance + eta * residual_covariance

    if jitter > 0:
        eye = torch.eye(
            effective.size(-1),
            dtype=effective.dtype,
            device=effective.device,
        )
        effective = effective + jitter * eye

    effective = 0.5 * (
        effective + effective.transpose(-1, -2)
    )
    if project_psd:
        effective = project_to_psd(
            effective,
            minimum_eigenvalue=minimum_eigenvalue,
        )
    return effective
