from __future__ import annotations

import argparse

import torch

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset
from implementations.rcr_dfl.src.effective_covariance import build_effective_covariance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RCR Dataset과 위험행렬을 점검합니다.")
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--market-mode", choices=["equal_weight", "external"], default="equal_weight")
    parser.add_argument("--market-price-csv", default=None)
    parser.add_argument("--market-column", default=None)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--return-type", choices=["simple", "log"], default="simple")
    parser.add_argument("--covariance-jitter", type=float, default=1e-6)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = RCRRollingMVODataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        return_type=args.return_type,
        covariance_jitter=args.covariance_jitter,
        market_mode=args.market_mode,
        market_price_csv=args.market_price_csv,
        market_column=args.market_column,
        risk_free_rate=args.risk_free_rate,
    )
    sample = dataset[args.sample_index]

    covariance = sample["covariance"]
    residual_covariance = sample["residual_covariance"]
    effective = build_effective_covariance(
        covariance=covariance,
        residual_covariance=residual_covariance,
        eta=args.eta,
        normalization="trace",
    )
    eta_zero = build_effective_covariance(
        covariance=covariance,
        residual_covariance=residual_covariance,
        eta=0.0,
        normalization="trace",
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
    print(f"target shape: {tuple(sample['target'].shape)}")
    print(f"covariance shape: {tuple(covariance.shape)}")
    print(f"residual covariance shape: {tuple(residual_covariance.shape)}")
    print(f"max |residual mean|: {sample['residuals'].mean(dim=0).abs().max().item():.3e}")
    print(f"max |Cov(market, residual)|: {market_residual_covariance.abs().max().item():.3e}")
    print(f"min eig covariance: {torch.linalg.eigvalsh(covariance).min().item():.3e}")
    print(f"min eig residual covariance: {torch.linalg.eigvalsh(residual_covariance).min().item():.3e}")
    print(f"min eig effective covariance: {torch.linalg.eigvalsh(effective).min().item():.3e}")
    print(f"trace covariance: {torch.trace(covariance).item():.3e}")
    print(f"trace residual covariance: {torch.trace(residual_covariance).item():.3e}")
    print(f"trace effective covariance: {torch.trace(effective).item():.3e}")
    print(f"eta=0 max difference: {(eta_zero - covariance).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
