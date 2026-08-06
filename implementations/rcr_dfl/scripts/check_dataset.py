from __future__ import annotations

import argparse

import torch

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset
from implementations.rcr_dfl.src.effective_covariance import build_effective_covariance
from implementations.rcr_dfl.src.residual_risk import matrix_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling CAPM, residual correlation, A_res, Sigma_eff를 점검합니다.")
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--market-mode", choices=["equal_weight", "external"], default="equal_weight")
    parser.add_argument("--market-price-csv", default=None)
    parser.add_argument("--market-column", default=None)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--return-type", choices=["simple", "log"], default="simple")
    parser.add_argument("--covariance-jitter", type=float, default=1e-6)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--residual-correlation-shrinkage", type=float, default=0.1)
    parser.add_argument("--correlation-scaling", choices=["none", "trace"], default="trace")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--effective-jitter", type=float, default=0.0)
    return parser.parse_args()


def print_diagnostics(name: str, matrix: torch.Tensor) -> None:
    diagnostics = matrix_diagnostics(matrix)
    print(
        f"{name:24s} | symmetry={diagnostics.symmetry_error.item():.3e} | "
        f"min_eig={diagnostics.minimum_eigenvalue.item():.3e} | "
        f"max_eig={diagnostics.maximum_eigenvalue.item():.3e} | "
        f"condition={diagnostics.condition_number.item():.3e}"
    )


def main() -> None:
    args = parse_args()
    dataset = RCRRollingMVODataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type=args.return_type,
        covariance_jitter=args.covariance_jitter,
        market_mode=args.market_mode,
        market_price_csv=args.market_price_csv,
        market_column=args.market_column,
        risk_free_rate=args.risk_free_rate,
        residual_correlation_shrinkage=args.residual_correlation_shrinkage,
        correlation_scaling=args.correlation_scaling,
    )
    sample = dataset[args.sample_index]
    covariance = sample["covariance"]
    residual_correlation_raw = sample["residual_correlation_raw"]
    residual_correlation = sample["residual_correlation"]
    a_res = sample["a_res"]
    effective = build_effective_covariance(
        covariance=covariance,
        residual_risk_matrix=a_res,
        eta=args.eta,
        jitter=args.effective_jitter,
    )
    eta_zero = build_effective_covariance(
        covariance=covariance,
        residual_risk_matrix=a_res,
        eta=0.0,
    )
    market_centered = sample["market_window"] - sample["market_window"].mean()
    market_residual_covariance = (
        market_centered.unsqueeze(-1) * sample["residuals"]
    ).sum(dim=0) / (dataset.lookback - 1)

    print(f"samples: {len(dataset)}")
    print(f"assets: {dataset.n_assets}")
    print(f"lookback: {dataset.lookback}")
    print(f"market: {dataset.market_name} ({dataset.market_mode})")
    print(f"target date range: {dataset.target_dates[0].date()} ~ {dataset.target_dates[-1].date()}")
    print(f"sample target date: {sample['target_date']}")
    print(f"features shape: {tuple(sample['features'].shape)}")
    print(f"residuals shape: {tuple(sample['residuals'].shape)}")
    print(f"max |residual mean|: {sample['residuals'].mean(dim=0).abs().max().item():.3e}")
    print(f"max |Cov(market, residual)|: {market_residual_covariance.abs().max().item():.3e}")
    print(f"max |diag(C_res)-1|: {(torch.diagonal(residual_correlation_raw) - 1).abs().max().item():.3e}")
    print(f"max |Sigma_eff(eta=0)-Sigma|: {(eta_zero - covariance).abs().max().item():.3e}")
    print(f"trace Sigma: {torch.trace(covariance).item():.3e}")
    print(f"trace A_res: {torch.trace(a_res).item():.3e}")
    print_diagnostics("Sigma", covariance)
    print_diagnostics("raw residual correlation", residual_correlation_raw)
    print_diagnostics("shrunk residual correlation", residual_correlation)
    print_diagnostics("A_res", a_res)
    print_diagnostics("Sigma_eff", effective)


if __name__ == "__main__":
    main()
