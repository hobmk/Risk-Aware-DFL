from __future__ import annotations

import torch

from implementations.rcr_dfl.src.capm import fit_capm
from implementations.rcr_dfl.src.effective_covariance import build_effective_covariance
from implementations.rcr_dfl.src.residual_risk import (
    correlation_matrix,
    covariance_matrix,
    matrix_diagnostics,
    scale_correlation_to_covariance,
    shrink_correlation,
)


def main() -> None:
    torch.manual_seed(42)
    dtype = torch.float64
    lookback = 60
    n_assets = 5
    market = torch.randn(lookback, dtype=dtype) * 0.01
    beta = torch.linspace(0.7, 1.3, n_assets, dtype=dtype)
    common_residual = torch.randn(lookback, 1, dtype=dtype) * 0.003
    idiosyncratic = torch.randn(lookback, n_assets, dtype=dtype) * 0.005
    returns = market.unsqueeze(-1) * beta + 0.5 * common_residual + idiosyncratic

    capm = fit_capm(returns, market)
    covariance = covariance_matrix(returns) + 1e-6 * torch.eye(n_assets, dtype=dtype)
    residual_correlation_raw = correlation_matrix(capm.residuals)
    residual_correlation = shrink_correlation(residual_correlation_raw, shrinkage=0.1)
    a_res = scale_correlation_to_covariance(
        residual_correlation,
        reference_covariance=covariance,
        scaling="trace",
    )
    effective = build_effective_covariance(covariance, a_res, eta=0.5, jitter=1e-8)

    market_residual_covariance = (
        (market - market.mean()).unsqueeze(-1) * capm.residuals
    ).sum(dim=0) / (lookback - 1)

    for name, matrix in {
        "Sigma": covariance,
        "Corr(residual)": residual_correlation_raw,
        "A_res": a_res,
        "Sigma_eff": effective,
    }.items():
        diagnostics = matrix_diagnostics(matrix)
        print(
            f"{name:15s} | symmetry={diagnostics.symmetry_error.item():.3e} | "
            f"min_eig={diagnostics.minimum_eigenvalue.item():.3e} | "
            f"condition={diagnostics.condition_number.item():.3e}"
        )

    print(f"returns shape: {tuple(returns.shape)}")
    print(f"residuals shape: {tuple(capm.residuals.shape)}")
    print(f"max |mean residual|: {capm.residuals.mean(dim=0).abs().max().item():.3e}")
    print(f"max |Cov(market, residual)|: {market_residual_covariance.abs().max().item():.3e}")
    print(f"max |diag(C_res)-1|: {(torch.diagonal(residual_correlation_raw) - 1).abs().max().item():.3e}")
    print(f"trace Sigma: {torch.trace(covariance).item():.3e}")
    print(f"trace A_res: {torch.trace(a_res).item():.3e}")


if __name__ == "__main__":
    main()
