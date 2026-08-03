from __future__ import annotations

import torch

from implementations.rcr_dfl.src.capm import fit_capm
from implementations.rcr_dfl.src.effective_covariance import build_effective_covariance
from implementations.rcr_dfl.src.residual_risk import covariance_matrix


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
    covariance = covariance_matrix(returns)
    residual_covariance = covariance_matrix(capm.residuals)
    effective = build_effective_covariance(
        covariance,
        residual_covariance,
        eta=0.5,
        normalization="trace",
        jitter=1e-8,
    )

    market_residual_covariance = (
        (market - market.mean()).unsqueeze(-1) * capm.residuals
    ).sum(dim=0) / (lookback - 1)

    print(f"returns shape: {tuple(returns.shape)}")
    print(f"residuals shape: {tuple(capm.residuals.shape)}")
    print(f"max |mean residual|: {capm.residuals.mean(dim=0).abs().max().item():.3e}")
    print(f"max |Cov(market, residual)|: {market_residual_covariance.abs().max().item():.3e}")
    print(f"min eig covariance: {torch.linalg.eigvalsh(covariance).min().item():.3e}")
    print(f"min eig residual covariance: {torch.linalg.eigvalsh(residual_covariance).min().item():.3e}")
    print(f"min eig effective covariance: {torch.linalg.eigvalsh(effective).min().item():.3e}")
    print(f"trace covariance: {torch.trace(covariance).item():.3e}")
    print(f"trace residual covariance: {torch.trace(residual_covariance).item():.3e}")
    print(f"trace effective covariance: {torch.trace(effective).item():.3e}")


if __name__ == "__main__":
    main()
