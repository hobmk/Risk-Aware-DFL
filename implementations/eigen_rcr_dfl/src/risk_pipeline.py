from __future__ import annotations

from dataclasses import dataclass

import torch

from .capm import fit_capm
from .eigen_risk import (
    ScaleMode,
    build_eigenmode_risk,
    covariance_matrix,
)


@dataclass
class EigenRCRResult:
    base_covariance: torch.Tensor
    residual_covariance: torch.Tensor
    eigen_risk: torch.Tensor
    effective_covariance: torch.Tensor

    capm_alpha: torch.Tensor
    capm_beta: torch.Tensor
    residuals: torch.Tensor

    eigenvalues: torch.Tensor
    explained_ratio: torch.Tensor
    cumulative_explained_ratio: torch.Tensor
    retained_ratio: torch.Tensor


def build_eigen_rcr(
    asset_returns: torch.Tensor,
    market_returns: torch.Tensor,
    eigen_k: int,
    eta: float,
    risk_free_rate: float = 0.0,
    fit_intercept: bool = True,
    scale_mode: ScaleMode = "none",
    covariance_jitter: float = 1e-6,
) -> EigenRCRResult:
    """
    Eigenmode RCR 전체 risk pipeline.

    Historical returns
        -> base covariance
        -> CAPM
        -> residuals
        -> residual covariance
        -> top-K eigenmodes
        -> residual collective risk
        -> effective covariance

    Sigma_eff = Sigma + eta * A_res^(K)
    """
    if eta < 0:
        raise ValueError("eta는 0 이상이어야 합니다.")

    # ---------------------------------------------------------
    # 1. 기존 return covariance
    # ---------------------------------------------------------
    base_covariance = covariance_matrix(asset_returns)

    n_assets = base_covariance.size(-1)

    eye = torch.eye(
        n_assets,
        dtype=base_covariance.dtype,
        device=base_covariance.device,
    )

    base_covariance = (
        base_covariance
        + covariance_jitter * eye
    )

    # ---------------------------------------------------------
    # 2. CAPM residual 계산
    # ---------------------------------------------------------
    capm = fit_capm(
        asset_returns=asset_returns,
        market_returns=market_returns,
        risk_free_rate=risk_free_rate,
        fit_intercept=fit_intercept,
    )

    # ---------------------------------------------------------
    # 3. Residual covariance -> Top-K Eigenmode
    # ---------------------------------------------------------
    eigen = build_eigenmode_risk(
        residuals=capm.residuals,
        k=eigen_k,
        scale_mode=scale_mode,
        base_covariance=base_covariance,
    )

    # ---------------------------------------------------------
    # 4. Effective covariance
    #
    # Sigma_eff = Sigma + eta * A_res^(K)
    # ---------------------------------------------------------
    effective_covariance = (
        base_covariance
        + eta * eigen.risk_matrix
    )

    effective_covariance = 0.5 * (
        effective_covariance
        + effective_covariance.transpose(-1, -2)
    )

    return EigenRCRResult(
        base_covariance=base_covariance,
        residual_covariance=eigen.residual_covariance,
        eigen_risk=eigen.risk_matrix,
        effective_covariance=effective_covariance,
        capm_alpha=capm.alpha,
        capm_beta=capm.beta,
        residuals=capm.residuals,
        eigenvalues=eigen.eigenvalues,
        explained_ratio=eigen.explained_ratio,
        cumulative_explained_ratio=eigen.cumulative_explained_ratio,
        retained_ratio=eigen.retained_ratio,
    )