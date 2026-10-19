from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RankTailRiskResult:
    residual_covariance: torch.Tensor
    rank_tail_dependence: torch.Tensor
    risk_matrix: torch.Tensor
    tail_scores: torch.Tensor
    tail_fraction: torch.Tensor
    mean_offdiag_dependence: torch.Tensor
    mean_abs_offdiag_dependence: torch.Tensor
    diagonal_preservation_error: torch.Tensor


def _validate_residuals(residuals: torch.Tensor) -> None:
    if residuals.ndim != 3:
        raise ValueError(
            "residuals는 [batch, lookback, n_assets] 형태여야 합니다."
        )

    if residuals.shape[1] < 3:
        raise ValueError("lookback은 3 이상이어야 합니다.")

    if residuals.shape[2] < 2:
        raise ValueError("n_assets는 2 이상이어야 합니다.")

    if not torch.isfinite(residuals).all():
        raise ValueError("residuals에 NaN 또는 inf가 있습니다.")


def _within_asset_rank_percentiles(
    residuals: torch.Tensor,
) -> torch.Tensor:
    """
    각 rolling window 안에서 자산별 CAPM residual을 작은 값부터 순위화한다.

    가장 음수인 residual -> 작은 percentile
    가장 양수인 residual -> 큰 percentile

    u = rank / (T + 1)
    """
    batch_size, lookback, n_assets = residuals.shape

    order = torch.argsort(
        residuals,
        dim=1,
    )

    ranks = torch.empty_like(
        order,
        dtype=residuals.dtype,
    )

    ordinal = torch.arange(
        1,
        lookback + 1,
        device=residuals.device,
        dtype=residuals.dtype,
    ).view(1, lookback, 1).expand(
        batch_size,
        lookback,
        n_assets,
    )

    ranks.scatter_(
        dim=1,
        index=order,
        src=ordinal,
    )

    return ranks / float(lookback + 1)


def _residual_covariance(
    residuals: torch.Tensor,
) -> torch.Tensor:
    centered = (
        residuals
        - residuals.mean(dim=1, keepdim=True)
    )

    denominator = residuals.shape[1] - 1

    covariance = (
        centered.transpose(1, 2)
        @ centered
    ) / denominator

    return 0.5 * (
        covariance
        + covariance.transpose(-1, -2)
    )


def build_rank_tail_residual_risk(
    residuals: torch.Tensor,
    tail_q: float = 0.20,
    tail_side: str = "down",
    eps: float = 1e-12,
) -> RankTailRiskResult:
    """
    Rank-Tail Residual Risk.

    Downside:
        z = max(0, (q - u) / q)

    Upside:
        z = max(0, (u - (1 - q)) / q)

    이후 과정은 동일:
        tail score centering
        -> score covariance
        -> score correlation
        -> residual volatility scaling

    최종:
        A_RT = D_sigma R_RT D_sigma

    따라서
        diag(A_RT) = diag(Sigma_epsilon)
    """
    _validate_residuals(residuals)

    if not 0.0 < tail_q < 1.0:
        raise ValueError(
            "tail_q는 0과 1 사이여야 합니다."
        )

    if tail_side not in {"down", "up"}:
        raise ValueError(
            "tail_side는 'down' 또는 'up'이어야 합니다."
        )

    if eps <= 0.0:
        raise ValueError(
            "eps는 0보다 커야 합니다."
        )

    lookback = residuals.shape[1]
    n_assets = residuals.shape[2]

    rank_percentile = (
        _within_asset_rank_percentiles(
            residuals
        )
    )

    if tail_side == "down":
        tail_scores = torch.clamp(
            (tail_q - rank_percentile) / tail_q,
            min=0.0,
        )
    else:
        tail_scores = torch.clamp(
            (
                rank_percentile
                - (1.0 - tail_q)
            )
            / tail_q,
            min=0.0,
        )

    centered_tail = (
        tail_scores
        - tail_scores.mean(
            dim=1,
            keepdim=True,
        )
    )

    tail_covariance = (
        centered_tail.transpose(1, 2)
        @ centered_tail
    ) / (lookback - 1)

    tail_covariance = 0.5 * (
        tail_covariance
        + tail_covariance.transpose(-1, -2)
    )

    tail_variance = torch.diagonal(
        tail_covariance,
        dim1=-2,
        dim2=-1,
    )

    if (tail_variance <= eps).any():
        raise ValueError(
            f"tail_q가 너무 작아 일부 자산의 "
            f"{tail_side}-tail score 분산이 0입니다. "
            "tail_q를 키우세요."
        )

    inv_tail_std = torch.rsqrt(
        tail_variance.clamp_min(eps)
    )

    rank_tail_dependence = (
        inv_tail_std.unsqueeze(-1)
        * tail_covariance
        * inv_tail_std.unsqueeze(-2)
    )

    rank_tail_dependence = 0.5 * (
        rank_tail_dependence
        + rank_tail_dependence.transpose(-1, -2)
    )

    # 수치오차를 없애고 correlation diagonal을 정확히 1로 둔다.
    diagonal = torch.diagonal(
        rank_tail_dependence,
        dim1=-2,
        dim2=-1,
    )
    diagonal.copy_(
        torch.ones_like(diagonal)
    )

    residual_covariance = (
        _residual_covariance(
            residuals
        )
    )

    residual_variance = torch.diagonal(
        residual_covariance,
        dim1=-2,
        dim2=-1,
    ).clamp_min(0.0)

    residual_std = torch.sqrt(
        residual_variance
    )

    risk_matrix = (
        residual_std.unsqueeze(-1)
        * rank_tail_dependence
        * residual_std.unsqueeze(-2)
    )

    risk_matrix = 0.5 * (
        risk_matrix
        + risk_matrix.transpose(-1, -2)
    )

    risk_diagonal = torch.diagonal(
        risk_matrix,
        dim1=-2,
        dim2=-1,
    )

    diagonal_preservation_error = (
        risk_diagonal
        - residual_variance
    ).abs().amax(dim=-1)

    offdiag_mask = (
        ~torch.eye(
            n_assets,
            dtype=torch.bool,
            device=residuals.device,
        )
    ).unsqueeze(0)

    offdiag_values = (
        rank_tail_dependence[
            offdiag_mask.expand_as(
                rank_tail_dependence
            )
        ]
        .reshape(
            residuals.shape[0],
            -1,
        )
    )

    tail_fraction = (
        (tail_scores > 0.0)
        .to(residuals.dtype)
        .mean(dim=(1, 2))
    )

    return RankTailRiskResult(
        residual_covariance=residual_covariance.contiguous(),
        rank_tail_dependence=rank_tail_dependence.contiguous(),
        risk_matrix=risk_matrix.contiguous(),
        tail_scores=tail_scores.contiguous(),
        tail_fraction=tail_fraction.contiguous(),
        mean_offdiag_dependence=(
            offdiag_values.mean(dim=-1)
        ).contiguous(),
        mean_abs_offdiag_dependence=(
            offdiag_values.abs().mean(dim=-1)
        ).contiguous(),
        diagonal_preservation_error=(
            diagonal_preservation_error
        ).contiguous(),
    )