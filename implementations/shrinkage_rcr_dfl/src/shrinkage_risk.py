from __future__ import annotations

from dataclasses import dataclass

import torch

from implementations.eigen_rcr_dfl.src.eigen_risk import (
    ScaleMode,
    covariance_matrix,
)


@dataclass
class ShrinkageRiskResult:
    residual_covariance: torch.Tensor
    risk_matrix: torch.Tensor
    eigenvalues: torch.Tensor
    shrunk_eigenvalues: torch.Tensor
    shrinkage_factors: torch.Tensor
    tau: torch.Tensor
    explained_ratio: torch.Tensor
    cumulative_explained_ratio: torch.Tensor
    retained_ratio: torch.Tensor


def build_shrinkage_risk(
    residuals: torch.Tensor,
    shrinkage_c: float,
    scale_mode: ScaleMode = "none",
    base_covariance: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> ShrinkageRiskResult:
    """
    Adaptive Eigenvalue Shrinkage RCR.

    Residual covariance:
        Sigma_eps = U Lambda U^T

    Adaptive threshold:
        tau_t = c * mean(lambda_t)

    Shrinkage factor:
        s_k = lambda_k / (lambda_k + tau_t)

    Shrunk eigenvalue:
        lambda_tilde_k = s_k * lambda_k

    Residual risk:
        A_res = U Lambda_tilde U^T
    """
    if shrinkage_c < 0:
        raise ValueError(
            "shrinkage_c는 0 이상이어야 합니다."
        )

    if scale_mode not in {
        "none",
        "preserve_residual_trace",
        "match_base_trace",
    }:
        raise ValueError(
            f"지원하지 않는 scale_mode입니다: "
            f"{scale_mode}"
        )

    residual_covariance = covariance_matrix(
        residuals
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(
        residual_covariance
    )

    # torch.linalg.eigh는 ascending order이므로 descending으로 변경.
    eigenvalues = eigenvalues.flip(
        dims=(-1,)
    ).clamp_min(0.0)

    eigenvectors = eigenvectors.flip(
        dims=(-1,)
    )

    total_eigenvalue = eigenvalues.sum(
        dim=-1,
        keepdim=True,
    )

    mean_eigenvalue = (
        total_eigenvalue
        / eigenvalues.shape[-1]
    )

    tau = (
        float(shrinkage_c)
        * mean_eigenvalue
    )

    if shrinkage_c == 0.0:
        # c=0은 Full Residual Covariance와 정확히 동일하게 둔다.
        shrinkage_factors = torch.ones_like(
            eigenvalues
        )

        shrunk_eigenvalues = (
            eigenvalues.clone()
        )

        raw_risk_matrix = (
            residual_covariance.clone()
        )

    else:
        denominator = (
            eigenvalues
            + tau
        )

        shrinkage_factors = torch.where(
            denominator > eps,
            eigenvalues / denominator,
            torch.zeros_like(eigenvalues),
        )

        shrunk_eigenvalues = (
            shrinkage_factors
            * eigenvalues
        )

        # U Lambda_tilde U^T
        weighted_vectors = (
            eigenvectors
            * shrunk_eigenvalues.unsqueeze(-2)
        )

        raw_risk_matrix = (
            weighted_vectors
            @ eigenvectors.transpose(-1, -2)
        )

        raw_risk_matrix = 0.5 * (
            raw_risk_matrix
            + raw_risk_matrix.transpose(-1, -2)
        )

    raw_trace = shrunk_eigenvalues.sum(
        dim=-1,
        keepdim=True,
    )

    retained_ratio = torch.where(
        total_eigenvalue > eps,
        raw_trace / total_eigenvalue,
        torch.zeros_like(raw_trace),
    ).squeeze(-1)

    risk_matrix = raw_risk_matrix

    if scale_mode == "preserve_residual_trace":
        scale = torch.where(
            raw_trace > eps,
            total_eigenvalue / raw_trace,
            torch.ones_like(raw_trace),
        )

        risk_matrix = (
            risk_matrix
            * scale.unsqueeze(-1)
        )

    elif scale_mode == "match_base_trace":
        if base_covariance is None:
            raise ValueError(
                "match_base_trace에는 "
                "base_covariance가 필요합니다."
            )

        base_trace = torch.diagonal(
            base_covariance,
            dim1=-2,
            dim2=-1,
        ).sum(
            dim=-1,
            keepdim=True,
        )

        scale = torch.where(
            raw_trace > eps,
            base_trace / raw_trace,
            torch.ones_like(raw_trace),
        )

        risk_matrix = (
            risk_matrix
            * scale.unsqueeze(-1)
        )

    risk_matrix = 0.5 * (
        risk_matrix
        + risk_matrix.transpose(-1, -2)
    )

    explained_ratio = torch.where(
        total_eigenvalue > eps,
        eigenvalues / total_eigenvalue,
        torch.zeros_like(eigenvalues),
    )

    cumulative_explained_ratio = torch.cumsum(
        explained_ratio,
        dim=-1,
    )

    return ShrinkageRiskResult(
        residual_covariance=residual_covariance,
        risk_matrix=risk_matrix,
        eigenvalues=eigenvalues,
        shrunk_eigenvalues=shrunk_eigenvalues,
        shrinkage_factors=shrinkage_factors,
        tau=tau.squeeze(-1),
        explained_ratio=explained_ratio,
        cumulative_explained_ratio=cumulative_explained_ratio,
        retained_ratio=retained_ratio,
    )