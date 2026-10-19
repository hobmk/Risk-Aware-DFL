from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from implementations.eigen_rcr_dfl.src.optimization import (
    build_markowitz_layer,
    covariance_to_risk_factor,
)
from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--validation-end", default="2022-12-31")
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.2)
    parser.add_argument(
        "--down-root",
        default="implementations/rank_tail_rcr_dfl/outputs/q_grid/q_0.35",
    )
    parser.add_argument(
        "--up-root",
        default="implementations/rank_tail_rcr_dfl/outputs/upside_placebo/q_0.35",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "implementations/rank_tail_rcr_dfl/outputs/mechanism/"
            "q_0.35/objective_scale"
        ),
    )
    return parser.parse_args()


def load_matrix(path: Path, tickers: list[str], value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if all(t in df.columns for t in tickers):
        return df.set_index("date").sort_index().reindex(columns=tickers).astype(float)

    return (
        df.pivot(index="date", columns="ticker", values=value_col)
        .sort_index()
        .reindex(columns=tickers)
        .astype(float)
    )


def build_dataset(q: float, side: str):
    return RankTailRCRDataset(
        price_csv="data/raw/dow30_adjusted_close.csv",
        lookback=60,
        date_column="Date",
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        tail_q=q,
        tail_side=side,
        dtype=torch.float64,
    )


def solve_weights(layer, mu, covariance, risk_aversion, batch_size=64):
    rows = []

    for start in range(0, len(mu), batch_size):
        end = start + batch_size

        mu_t = torch.tensor(mu[start:end], dtype=torch.float64)
        cov_t = torch.tensor(covariance[start:end], dtype=torch.float64)

        risk_factor = covariance_to_risk_factor(
            covariance=cov_t,
            risk_aversion=risk_aversion,
            jitter=0.0,
        )

        weights, = layer(mu_t, risk_factor)
        rows.append(weights.detach().cpu().numpy())

    return np.concatenate(rows, axis=0)


def objective_terms(mu, covariance, weights, risk_aversion):
    predicted_return = np.einsum("bi,bi->b", mu, weights)

    portfolio_variance = np.einsum(
        "bi,bij,bj->b",
        weights,
        covariance,
        weights,
        optimize=True,
    )

    risk_term = risk_aversion * portfolio_variance
    abs_return_term = np.abs(predicted_return)

    ratio = risk_term / np.maximum(abs_return_term, 1e-12)

    return predicted_return, risk_term, ratio


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    down_ds = build_dataset(args.tail_q, "down")
    up_ds = build_dataset(args.tail_q, "up")

    tickers = list(down_ds.tickers)
    dates = pd.DatetimeIndex(down_ds.target_dates).normalize()
    mask = dates > pd.Timestamp(args.validation_end)
    test_dates = dates[mask]

    down_cov = down_ds.covariances[mask].cpu().numpy()
    up_cov = up_ds.covariances[mask].cpu().numpy()

    layer = build_markowitz_layer(
        n_assets=len(tickers),
        max_weight=args.max_weight,
    )

    rows = []
    daily_rows = []

    for seed in args.seeds:
        down_pred = load_matrix(
            Path(args.down_root) / f"seed_{seed}" / "predicted_returns.csv",
            tickers,
            "predicted_return",
        )
        up_pred = load_matrix(
            Path(args.up_root) / f"seed_{seed}" / "predicted_returns.csv",
            tickers,
            "predicted_return",
        )

        common = test_dates.intersection(down_pred.index).intersection(up_pred.index)
        pos = test_dates.get_indexer(common)

        combinations = {
            "DownPred_DownCov": (down_pred.loc[common].to_numpy(), down_cov[pos]),
            "DownPred_UpCov": (down_pred.loc[common].to_numpy(), up_cov[pos]),
            "UpPred_DownCov": (up_pred.loc[common].to_numpy(), down_cov[pos]),
            "UpPred_UpCov": (up_pred.loc[common].to_numpy(), up_cov[pos]),
        }

        for name, (mu, cov) in combinations.items():
            weights = solve_weights(
                layer,
                mu,
                cov,
                args.risk_aversion,
            )

            return_term, risk_term, ratio = objective_terms(
                mu,
                cov,
                weights,
                args.risk_aversion,
            )

            rows.append({
                "seed": seed,
                "combination": name,
                "mean_abs_return_term": np.mean(np.abs(return_term)),
                "median_abs_return_term": np.median(np.abs(return_term)),
                "mean_risk_term": np.mean(risk_term),
                "median_risk_term": np.median(risk_term),
                "mean_ratio": np.mean(ratio),
                "median_ratio": np.median(ratio),
                "ratio_p25": np.quantile(ratio, 0.25),
                "ratio_p75": np.quantile(ratio, 0.75),
                "risk_gt_return_fraction": np.mean(ratio > 1.0),
            })

            for date, ret, risk, r in zip(common, return_term, risk_term, ratio):
                daily_rows.append({
                    "seed": seed,
                    "combination": name,
                    "date": date,
                    "predicted_portfolio_return": ret,
                    "risk_term": risk,
                    "risk_return_ratio": r,
                })

    summary = pd.DataFrame(rows)
    daily = pd.DataFrame(daily_rows)

    summary.to_csv(
        output_dir / "objective_scale_by_seed.csv",
        index=False,
    )
    daily.to_csv(
        output_dir / "objective_scale_daily.csv",
        index=False,
    )

    aggregate = (
        summary.groupby("combination")[
            [
                "mean_abs_return_term",
                "mean_risk_term",
                "median_ratio",
                "risk_gt_return_fraction",
            ]
        ]
        .agg(["mean", "std"])
    )

    aggregate.to_csv(
        output_dir / "objective_scale_summary.csv"
    )

    print("=" * 110)
    print("OBJECTIVE SCALE — BY SEED")
    print("=" * 110)
    print(summary.to_string(index=False))

    print()
    print("=" * 110)
    print("OBJECTIVE SCALE — 5-SEED SUMMARY")
    print("=" * 110)
    print(aggregate.to_string())

    print()
    print("Interpretation:")
    print("ratio << 1  : predicted-return term dominates")
    print("ratio ~ 1   : return/risk terms have similar scale")
    print("ratio >> 1  : covariance risk term dominates")

    print()
    print("Saved:")
    print(output_dir)


if __name__ == "__main__":
    main()