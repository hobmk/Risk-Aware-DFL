from __future__ import annotations

import torch


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


def covariance_matrix(values: torch.Tensor, correction: int = 1) -> torch.Tensor:
    """[T, N] 또는 [B, T, N] 관측치의 자산 간 공분산을 계산한다."""
    batched, squeeze_batch = _as_batched_observations(values)
    n_observations = batched.size(-2)
    denominator = n_observations - correction
    if denominator <= 0:
        raise ValueError(
            "관측치 수가 correction보다 커야 합니다. "
            f"n_observations={n_observations}, correction={correction}"
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
    """공분산을 표준편차로 나누어 correlation matrix를 계산한다."""
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

    diagonal = torch.diagonal(correlation, dim1=-2, dim2=-1)
    diagonal.copy_(torch.ones_like(diagonal))
    return correlation


def normalize_covariance_trace(
    covariance: torch.Tensor,
    reference_covariance: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    covariance의 trace를 reference_covariance의 trace와 맞춘다.

    normalized = covariance * tr(reference) / (tr(covariance) + eps)
    """
    if covariance.shape != reference_covariance.shape:
        raise ValueError(
            "covariance와 reference_covariance shape이 일치해야 합니다. "
            f"covariance={tuple(covariance.shape)}, "
            f"reference={tuple(reference_covariance.shape)}"
        )
    if covariance.ndim not in {2, 3} or covariance.size(-1) != covariance.size(-2):
        raise ValueError("입력 행렬 shape은 [N, N] 또는 [B, N, N]이어야 합니다.")
    if eps <= 0:
        raise ValueError(f"eps는 0보다 커야 합니다: {eps}")

    source_trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)
    reference_trace = torch.diagonal(
        reference_covariance,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)

    if torch.any(source_trace <= eps):
        raise ValueError("정규화할 covariance의 trace가 0에 가깝거나 음수입니다.")
    if torch.any(reference_trace <= 0):
        raise ValueError("reference_covariance의 trace는 양수여야 합니다.")

    scale = reference_trace / (source_trace + eps)
    return covariance * scale[..., None, None]
