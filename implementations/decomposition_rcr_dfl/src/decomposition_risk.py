from __future__ import annotations

from dataclasses import dataclass

import torch

from implementations.eigen_rcr_dfl.src.eigen_risk import covariance_matrix


@dataclass
class DecompositionRiskResult:
    base_covariance: torch.Tensor
    market_covariance: torch.Tensor
    residual_covariance: torch.Tensor
    dominant_residual: torch.Tensor
    residual_diagonal: torch.Tensor
    reconstructed_residual: torch.Tensor
    effective_covariance: torch.Tensor
    eigenvalues: torch.Tensor
    explained_ratio: torch.Tensor
    cumulative_explained_ratio: torch.Tensor
    retained_ratio: torch.Tensor
    decomposition_error_fro: torch.Tensor


def build_decomposition_risk(
    asset_returns: torch.Tensor,
    market_returns: torch.Tensor,
    beta: torch.Tensor,
    residuals: torch.Tensor,
    k: int,
    covariance_jitter: float = 1e-6,
    eps: float = 1e-12,
) -> DecompositionRiskResult:
    """
    Residual Risk Decomposition & Reconstruction.

    Sigma_market = beta beta^T sigma_m^2

    Sigma_eps = U Lambda U^T

    Sigma_dom^(K) = U_K Lambda_K U_K^T

    D_K = Diag(
        diag(
            Sigma_eps - Sigma_dom^(K)
        )
    )

    Sigma_eps_tilde^(K)
        = Sigma_dom^(K) + D_K

    Sigma_eff^(K)
        = Sigma_market
        + Sigma_eps_tilde^(K)

    covariance_jitter는 최종 covariance에만 동일하게 추가한다.
    """

    if asset_returns.ndim != 3:
        raise ValueError(
            "asset_returns는 [B, T, N]이어야 합니다."
        )

    if market_returns.ndim != 2:
        raise ValueError(
            "market_returns는 [B, T]이어야 합니다."
        )

    n_assets = asset_returns.shape[-1]

    if not 0 <= k <= n_assets:
        raise ValueError(
            f"k는 0 이상 N 이하이어야 합니다: "
            f"k={k}, N={n_assets}"
        )

    if covariance_jitter < 0:
        raise ValueError(
            "covariance_jitter는 0 이상이어야 합니다."
        )

    # Raw total return covariance.
    base_raw = covariance_matrix(
        asset_returns
    )

    # CAPM residual covariance.
    residual_covariance = covariance_matrix(
        residuals
    )

    # 시장수익률도 기존 covariance_matrix를 사용해
    # covariance convention을 정확히 동일하게 맞춘다.
    market_variance = covariance_matrix(
        market_returns.unsqueeze(-1)
    ).squeeze(-1).squeeze(-1)

    # Sigma_market = beta beta^T sigma_m^2
    market_covariance = (
        beta.unsqueeze(-1)
        * beta.unsqueeze(-2)
        * market_variance[:, None, None]
    )

    # Residual eigendecomposition.
    eigenvalues, eigenvectors = torch.linalg.eigh(
        residual_covariance
    )

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

    explained_ratio = torch.where(
        total_eigenvalue > eps,
        eigenvalues / total_eigenvalue,
        torch.zeros_like(eigenvalues),
    )

    cumulative_explained_ratio = torch.cumsum(
        explained_ratio,
        dim=-1,
    )

    if k == 0:
        dominant_residual = torch.zeros_like(
            residual_covariance
        )

        retained_ratio = torch.zeros(
            residual_covariance.shape[0],
            dtype=residual_covariance.dtype,
            device=residual_covariance.device,
        )

    elif k == n_assets:
        # K=N parity를 위해 EVD reconstruction 대신
        # 원 residual covariance를 정확히 사용.
        dominant_residual = (
            residual_covariance.clone()
        )

        retained_ratio = torch.ones(
            residual_covariance.shape[0],
            dtype=residual_covariance.dtype,
            device=residual_covariance.device,
        )

    else:
        top_values = eigenvalues[..., :k]
        top_vectors = eigenvectors[..., :k]

        dominant_residual = (
            top_vectors
            * top_values.unsqueeze(-2)
        ) @ top_vectors.transpose(-1, -2)

        dominant_residual = 0.5 * (
            dominant_residual
            + dominant_residual.transpose(-1, -2)
        )

        retained_ratio = (
            top_values.sum(dim=-1)
            / total_eigenvalue.squeeze(-1).clamp_min(eps)
        )

    # Top-K 밖의 residual covariance.
    residual_remainder = (
        residual_covariance
        - dominant_residual
    )

    # Cross covariance는 제거하고 diagonal variance만 보존.
    residual_diagonal_values = torch.diagonal(
        residual_remainder,
        dim1=-2,
        dim2=-1,
    ).clamp_min(0.0)

    residual_diagonal = torch.diag_embed(
        residual_diagonal_values
    )

    reconstructed_residual = (
        dominant_residual
        + residual_diagonal
    )

    reconstructed_residual = 0.5 * (
        reconstructed_residual
        + reconstructed_residual.transpose(-1, -2)
    )

    # CAPM decomposition 자체의 numerical error.
    decomposition_error = (
        base_raw
        - (
            market_covariance
            + residual_covariance
        )
    )

    decomposition_error_fro = torch.linalg.matrix_norm(
        decomposition_error,
        ord="fro",
        dim=(-2, -1),
    )

    eye = torch.eye(
        n_assets,
        dtype=asset_returns.dtype,
        device=asset_returns.device,
    ).unsqueeze(0)

    base_covariance = (
        base_raw
        + covariance_jitter * eye
    )

    # K=N은 DFL-MVO의 exact special case.
    # market + residual을 다시 더하지 않고
    # baseline covariance 자체를 그대로 사용한다.
    if k == n_assets:
        effective_covariance = (
            base_covariance.clone()
        )
    else:
        effective_covariance = (
            market_covariance
            + reconstructed_residual
            + covariance_jitter * eye
        )

        effective_covariance = 0.5 * (
            effective_covariance
            + effective_covariance.transpose(-1, -2)
        )

    return DecompositionRiskResult(
        base_covariance=base_covariance,
        market_covariance=market_covariance,
        residual_covariance=residual_covariance,
        dominant_residual=dominant_residual,
        residual_diagonal=residual_diagonal,
        reconstructed_residual=reconstructed_residual,
        effective_covariance=effective_covariance,
        eigenvalues=eigenvalues,
        explained_ratio=explained_ratio,
        cumulative_explained_ratio=cumulative_explained_ratio,
        retained_ratio=retained_ratio,
        decomposition_error_fro=decomposition_error_fro,
    )