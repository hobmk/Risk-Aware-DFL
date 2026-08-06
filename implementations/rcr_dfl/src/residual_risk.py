from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

CorrelationScaling = Literal["none", "trace"]


@dataclass(frozen=True)
class MatrixDiagnostics:
    symmetry_error: torch.Tensor
    minimum_eigenvalue: torch.Tensor
    maximum_eigenvalue: torch.Tensor
    condition_number: torch.Tensor


def _as_batched_observations(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if values.ndim not in {2, 3}:
        raise ValueError(
            "values shape은 [T, N] 또는 [B, T, N]이어야 합니다. "
            f"현재 shape={tuple(values.shape)}"
        )
    if values.size(-2) < 2:
        raise ValueError("공분산 계산을 위해 관측치는 2개 이상이어야 합니다.")
    if not torch.is_floating_point(values):
        raise TypeError("values는 부동소수점 Tensor여야 합니다.")
    if not torch.isfinite(values).all():
        raise ValueError("values에 NaN 또는 inf가 존재합니다.")
    squeeze_batch = values.ndim == 2
    return (values.unsqueeze(0) if squeeze_batch else values), squeeze_batch


def _validate_square_matrix(matrix: torch.Tensor, name: str) -> None:
    if matrix.ndim not in {2, 3} or matrix.size(-1) != matrix.size(-2):
        raise ValueError(f"{name} shape은 [N, N] 또는 [B, N, N]이어야 합니다.")
    if not torch.is_floating_point(matrix):
        raise TypeError(f"{name}는 부동소수점 Tensor여야 합니다.")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name}에 NaN 또는 inf가 존재합니다.")


def covariance_matrix(values: torch.Tensor, correction: int = 1) -> torch.Tensor:
    """[T, N] 또는 [B, T, N] 관측치의 자산 간 공분산을 계산한다."""
    batched, squeeze_batch = _as_batched_observations(values)
    denominator = batched.size(-2) - correction
    if denominator <= 0:
        raise ValueError(
            "관측치 수가 correction보다 커야 합니다. "
            f"n_observations={batched.size(-2)}, correction={correction}"
        )
    centered = batched - batched.mean(dim=-2, keepdim=True)
    covariance = centered.transpose(-1, -2) @ centered / denominator
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    return covariance.squeeze(0) if squeeze_batch else covariance


def correlation_matrix(
    values: torch.Tensor,
    correction: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """관측치로부터 Pearson correlation matrix를 계산한다."""
    if eps <= 0:
        raise ValueError(f"eps는 0보다 커야 합니다: {eps}")
    covariance = covariance_matrix(values, correction=correction)
    variances = torch.diagonal(covariance, dim1=-2, dim2=-1)
    if torch.any(variances <= eps):
        raise ValueError("분산이 0에 가까운 자산이 있어 correlation을 계산할 수 없습니다.")
    standard_deviations = variances.sqrt()
    denominator = standard_deviations.unsqueeze(-1) * standard_deviations.unsqueeze(-2)
    correlation = covariance / denominator
    correlation = 0.5 * (correlation + correlation.transpose(-1, -2))
    identity = torch.eye(correlation.size(-1), dtype=correlation.dtype, device=correlation.device)
    diagonal = torch.diag_embed(torch.diagonal(correlation, dim1=-2, dim2=-1))
    return correlation - diagonal + identity


def shrink_correlation(correlation: torch.Tensor, shrinkage: float = 0.1) -> torch.Tensor:
    """C_bar = (1-rho)C + rho I 형태의 correlation shrinkage를 적용한다."""
    _validate_square_matrix(correlation, "correlation")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage는 0 이상 1 이하여야 합니다: {shrinkage}")
    correlation = 0.5 * (correlation + correlation.transpose(-1, -2))
    identity = torch.eye(correlation.size(-1), dtype=correlation.dtype, device=correlation.device)
    shrunk = (1.0 - shrinkage) * correlation + shrinkage * identity
    diagonal = torch.diag_embed(torch.diagonal(shrunk, dim1=-2, dim2=-1))
    return 0.5 * (shrunk - diagonal + identity + (shrunk - diagonal + identity).transpose(-1, -2))


def scale_correlation_to_covariance(
    correlation: torch.Tensor,
    reference_covariance: torch.Tensor,
    scaling: CorrelationScaling = "trace",
) -> torch.Tensor:
    """
    무차원 correlation을 covariance와 합산 가능한 수익률² 단위로 변환한다.

    scaling="trace":
        A_res = tr(Sigma) / N * C_res
        correlation의 trace가 N이므로 tr(A_res)=tr(Sigma)가 된다.
    scaling="none":
        correlation을 그대로 반환한다. 단위가 맞지 않으므로 진단용으로만 사용한다.
    """
    _validate_square_matrix(correlation, "correlation")
    _validate_square_matrix(reference_covariance, "reference_covariance")
    if correlation.shape != reference_covariance.shape:
        raise ValueError(
            "correlation과 reference_covariance shape이 일치해야 합니다. "
            f"correlation={tuple(correlation.shape)}, "
            f"reference={tuple(reference_covariance.shape)}"
        )
    if scaling not in {"none", "trace"}:
        raise ValueError(f"지원하지 않는 scaling입니다: {scaling}")
    correlation = 0.5 * (correlation + correlation.transpose(-1, -2))
    if scaling == "none":
        return correlation
    reference_trace = torch.diagonal(
        reference_covariance, dim1=-2, dim2=-1
    ).sum(dim=-1)
    if torch.any(reference_trace <= 0):
        raise ValueError("reference_covariance의 trace는 양수여야 합니다.")
    scale = reference_trace / correlation.size(-1)
    return correlation * scale[..., None, None]


def normalize_covariance_trace(
    covariance: torch.Tensor,
    reference_covariance: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """기존 코드 호환용: covariance의 trace를 reference와 맞춘다."""
    _validate_square_matrix(covariance, "covariance")
    _validate_square_matrix(reference_covariance, "reference_covariance")
    if covariance.shape != reference_covariance.shape:
        raise ValueError(
            "covariance와 reference_covariance shape이 일치해야 합니다. "
            f"covariance={tuple(covariance.shape)}, "
            f"reference={tuple(reference_covariance.shape)}"
        )
    if eps <= 0:
        raise ValueError(f"eps는 0보다 커야 합니다: {eps}")
    source_trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)
    reference_trace = torch.diagonal(
        reference_covariance, dim1=-2, dim2=-1
    ).sum(dim=-1)
    if torch.any(source_trace <= eps):
        raise ValueError("정규화할 covariance의 trace가 0에 가깝거나 음수입니다.")
    if torch.any(reference_trace <= 0):
        raise ValueError("reference_covariance의 trace는 양수여야 합니다.")
    scale = reference_trace / (source_trace + eps)
    return covariance * scale[..., None, None]


def matrix_diagnostics(matrix: torch.Tensor, psd_tolerance: float = 1e-12) -> MatrixDiagnostics:
    """대칭 오차, 최소/최대 고유값, PSD condition number를 계산한다."""
    _validate_square_matrix(matrix, "matrix")
    if psd_tolerance <= 0:
        raise ValueError(f"psd_tolerance는 0보다 커야 합니다: {psd_tolerance}")
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    symmetry_error = (matrix - matrix.transpose(-1, -2)).abs().amax(dim=(-2, -1))
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    minimum = eigenvalues[..., 0]
    maximum = eigenvalues[..., -1]
    condition = torch.where(
        minimum > psd_tolerance,
        maximum / minimum,
        torch.full_like(maximum, float("inf")),
    )
    return MatrixDiagnostics(symmetry_error, minimum, maximum, condition)
