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
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1, 5, 10, 25, 50, 100, 150])
    parser.add_argument("--validation-end", default="2022-12-31")
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
            "q_0.35/lambda_sensitivity"
        ),
    )
    return parser.parse_args()


def load_matrix(path: Path, tickers: list[str], value_col: str):
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


def metrics(weights, true_returns, mu, covariance, risk_aversion):
    returns = np.sum(weights * true_returns, axis=1)

    cumulative = np.prod(1.0 + returns) - 1.0
    annualized_return = (1.0 + cumulative) ** (252.0 / len(returns)) - 1.0

    daily_std = returns.std(ddof=1)
    volatility = daily_std * np.sqrt(252.0)
    sharpe = returns.mean() / daily_std * np.sqrt(252.0)

    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    max_drawdown = np.max(1.0 - wealth / peak)

    turnover = 0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=1).mean() * 252.0

    predicted_return = np.abs(np.sum(mu * weights, axis=1))
    variance = np.einsum("bi,bij,bj->b", weights, covariance, weights)
    risk_term = risk_aversion * variance
    ratio = risk_term / np.maximum(predicted_return, 1e-12)

    return {
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "annualized_turnover": turnover,
        "active_assets": (weights > 1e-3).sum(axis=1).mean(),
        "median_objective_ratio": np.median(ratio),
        "mean_objective_ratio": np.mean(ratio),
    }


def cosine_rows(a, b):
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.mean(numerator / np.maximum(denominator, 1e-12))


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

    result_rows = []
    effect_rows = []

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
        true_ret = load_matrix(
            Path(args.down_root) / f"seed_{seed}" / "true_returns.csv",
            tickers,
            "true_return",
        )

        common = (
            test_dates.intersection(down_pred.index)
            .intersection(up_pred.index)
            .intersection(true_ret.index)
        )

        pos = test_dates.get_indexer(common)
        dp = down_pred.loc[common].to_numpy()
        up = up_pred.loc[common].to_numpy()
        tr = true_ret.loc[common].to_numpy()
        dc = down_cov[pos]
        uc = up_cov[pos]

        for risk_aversion in args.lambdas:
            for predictor_name, pred in [("DownPred", dp), ("UpPred", up)]:
                wd = solve_weights(layer, pred, dc, risk_aversion)
                wu = solve_weights(layer, pred, uc, risk_aversion)

                md = metrics(wd, tr, pred, dc, risk_aversion)
                mu = metrics(wu, tr, pred, uc, risk_aversion)

                result_rows.append({
                    "seed": seed,
                    "lambda": risk_aversion,
                    "combination": f"{predictor_name}_DownCov",
                    **md,
                })
                result_rows.append({
                    "seed": seed,
                    "lambda": risk_aversion,
                    "combination": f"{predictor_name}_UpCov",
                    **mu,
                })

                effect_rows.append({
                    "seed": seed,
                    "lambda": risk_aversion,
                    "predictor": predictor_name,
                    "down_minus_up_sharpe": md["sharpe"] - mu["sharpe"],
                    "down_minus_up_return": md["annualized_return"] - mu["annualized_return"],
                    "down_minus_up_volatility": md["volatility"] - mu["volatility"],
                    "down_minus_up_mdd": md["max_drawdown"] - mu["max_drawdown"],
                    "weight_cosine": cosine_rows(wd, wu),
                    "weight_l1": np.abs(wd - wu).sum(axis=1).mean(),
                })

        print(f"seed {seed} done")

    results = pd.DataFrame(result_rows)
    effects = pd.DataFrame(effect_rows)

    results.to_csv(
        output_dir / "lambda_sensitivity_by_seed.csv",
        index=False,
    )
    effects.to_csv(
        output_dir / "covariance_effect_by_seed.csv",
        index=False,
    )

    summary = (
        results.groupby(["lambda", "combination"])
        .agg({
            "annualized_return": ["mean", "std"],
            "volatility": ["mean", "std"],
            "sharpe": ["mean", "std"],
            "max_drawdown": ["mean", "std"],
            "median_objective_ratio": ["mean", "std"],
        })
    )

    effect_summary = (
        effects.groupby(["lambda", "predictor"])
        .agg({
            "down_minus_up_sharpe": ["mean", "std"],
            "down_minus_up_return": ["mean", "std"],
            "down_minus_up_volatility": ["mean", "std"],
            "down_minus_up_mdd": ["mean", "std"],
            "weight_cosine": ["mean", "std"],
            "weight_l1": ["mean", "std"],
        })
    )

    summary.to_csv(
        output_dir / "lambda_sensitivity_summary.csv"
    )
    effect_summary.to_csv(
        output_dir / "covariance_effect_summary.csv"
    )

    print()
    print("=" * 120)
    print("SHARPE — 5-SEED MEAN")
    print("=" * 120)
    print(
        results.groupby(["lambda", "combination"])["sharpe"]
        .mean()
        .unstack()
        .to_string()
    )

    print()
    print("=" * 120)
    print("MEDIAN RISK / RETURN OBJECTIVE RATIO — 5-SEED MEAN")
    print("=" * 120)
    print(
        results.groupby(["lambda", "combination"])["median_objective_ratio"]
        .mean()
        .unstack()
        .to_string()
    )

    print()
    print("=" * 120)
    print("COVARIANCE EFFECT — PREDICTOR FIXED")
    print("=" * 120)
    print(effect_summary.to_string())

    print()
    print("Saved:")
    print(output_dir)


if __name__ == "__main__":
    main()