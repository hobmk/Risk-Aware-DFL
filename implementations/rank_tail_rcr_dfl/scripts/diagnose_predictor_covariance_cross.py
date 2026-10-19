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
            "q_0.35/predictor_covariance_cross"
        ),
    )
    return parser.parse_args()


def load_matrix(path: Path, tickers: list[str], value_candidates: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if all(ticker in df.columns for ticker in tickers):
        return df.set_index("date").sort_index().reindex(columns=tickers).astype(float)

    if "ticker" in df.columns:
        value_col = next((c for c in value_candidates if c in df.columns), None)
        if value_col is None:
            raise KeyError(f"Value column not found: {path}")

        return (
            df.pivot(index="date", columns="ticker", values=value_col)
            .sort_index()
            .reindex(columns=tickers)
            .astype(float)
        )

    raise RuntimeError(f"Unsupported format: {path}")


def build_dataset(tail_q: float, side: str) -> RankTailRCRDataset:
    return RankTailRCRDataset(
        price_csv="data/raw/dow30_adjusted_close.csv",
        lookback=60,
        date_column="Date",
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        tail_q=tail_q,
        tail_side=side,
        dtype=torch.float64,
    )


def solve_weights(
    layer,
    predicted_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
    batch_size: int = 64,
) -> np.ndarray:
    weights = []

    for start in range(0, len(predicted_returns), batch_size):
        end = start + batch_size

        mu = torch.tensor(
            predicted_returns[start:end],
            dtype=torch.float64,
        )

        sigma = torch.tensor(
            covariance[start:end],
            dtype=torch.float64,
        )

        risk_factor = covariance_to_risk_factor(
            covariance=sigma,
            risk_aversion=risk_aversion,
            jitter=0.0,
        )

        w, = layer(mu, risk_factor)
        weights.append(w.detach().cpu().numpy())

    return np.concatenate(weights, axis=0)


def portfolio_metrics(weights: np.ndarray, true_returns: np.ndarray) -> dict:
    returns = np.sum(weights * true_returns, axis=1)

    cumulative = np.prod(1.0 + returns) - 1.0
    annualized_return = (1.0 + cumulative) ** (252.0 / len(returns)) - 1.0

    daily_std = returns.std(ddof=1)
    volatility = daily_std * np.sqrt(252.0)
    sharpe = returns.mean() / daily_std * np.sqrt(252.0)

    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    max_drawdown = np.max(1.0 - wealth / peak)

    turnover = np.abs(np.diff(weights, axis=0)).sum(axis=1).mean() * 252.0
    active_assets = (weights > 1e-3).sum(axis=1).mean()

    return {
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "annualized_turnover": turnover,
        "active_assets": active_assets,
    }


def prediction_similarity(down: np.ndarray, up: np.ndarray) -> dict:
    flat_down = down.ravel()
    flat_up = up.ravel()

    flat_corr = np.corrcoef(flat_down, flat_up)[0, 1]
    cosine = np.dot(flat_down, flat_up) / (
        np.linalg.norm(flat_down) * np.linalg.norm(flat_up)
    )

    daily_corr = []
    top5_overlap = []

    for d, u in zip(down, up):
        daily_corr.append(np.corrcoef(d, u)[0, 1])
        top5_overlap.append(
            len(set(np.argsort(d)[-5:]) & set(np.argsort(u)[-5:])) / 5.0
        )

    return {
        "flat_corr": flat_corr,
        "cosine": cosine,
        "mean_daily_cs_corr": np.nanmean(daily_corr),
        "rmse_difference": np.sqrt(np.mean((down - up) ** 2)),
        "mean_abs_difference": np.mean(np.abs(down - up)),
        "mean_top5_overlap": np.mean(top5_overlap),
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("PREDICTOR × COVARIANCE 2x2 CROSS-EVALUATION")
    print("=" * 110)

    down_dataset = build_dataset(args.tail_q, "down")
    up_dataset = build_dataset(args.tail_q, "up")

    tickers = list(down_dataset.tickers)
    dates = pd.DatetimeIndex(down_dataset.target_dates).normalize()
    test_mask = dates > pd.Timestamp(args.validation_end)
    test_dates = dates[test_mask]

    down_cov = down_dataset.covariances[test_mask].cpu().numpy()
    up_cov = up_dataset.covariances[test_mask].cpu().numpy()

    layer = build_markowitz_layer(
        n_assets=len(tickers),
        max_weight=args.max_weight,
    )

    similarity_rows = []
    result_rows = []

    for seed in args.seeds:
        down_dir = Path(args.down_root) / f"seed_{seed}"
        up_dir = Path(args.up_root) / f"seed_{seed}"

        down_pred = load_matrix(
            down_dir / "predicted_returns.csv",
            tickers,
            ["predicted_return"],
        )
        up_pred = load_matrix(
            up_dir / "predicted_returns.csv",
            tickers,
            ["predicted_return"],
        )
        true_ret = load_matrix(
            down_dir / "true_returns.csv",
            tickers,
            ["true_return"],
        )

        common = (
            test_dates.intersection(down_pred.index)
            .intersection(up_pred.index)
            .intersection(true_ret.index)
        )

        positions = test_dates.get_indexer(common)

        dp = down_pred.loc[common].to_numpy()
        up = up_pred.loc[common].to_numpy()
        tr = true_ret.loc[common].to_numpy()

        dc = down_cov[positions]
        uc = up_cov[positions]

        sim = prediction_similarity(dp, up)
        sim["seed"] = seed

        sim["down_mse"] = np.mean((dp - tr) ** 2)
        sim["up_mse"] = np.mean((up - tr) ** 2)
        similarity_rows.append(sim)

        combinations = {
            "DownPred_DownCov": (dp, dc),
            "DownPred_UpCov": (dp, uc),
            "UpPred_DownCov": (up, dc),
            "UpPred_UpCov": (up, uc),
        }

        for name, (pred, cov) in combinations.items():
            weights = solve_weights(
                layer=layer,
                predicted_returns=pred,
                covariance=cov,
                risk_aversion=args.risk_aversion,
            )

            metrics = portfolio_metrics(weights, tr)

            result_rows.append({
                "seed": seed,
                "combination": name,
                **metrics,
            })

        print(f"seed {seed} | dates={len(common)}")

    similarity_df = pd.DataFrame(similarity_rows)
    results_df = pd.DataFrame(result_rows)

    summary_df = (
        results_df
        .groupby("combination")
        .agg({
            "annualized_return": ["mean", "std"],
            "volatility": ["mean", "std"],
            "sharpe": ["mean", "std"],
            "max_drawdown": ["mean", "std"],
            "annualized_turnover": ["mean", "std"],
            "active_assets": ["mean", "std"],
        })
    )

    similarity_df.to_csv(
        output_dir / "predictor_similarity_by_seed.csv",
        index=False,
    )
    results_df.to_csv(
        output_dir / "cross_evaluation_by_seed.csv",
        index=False,
    )
    summary_df.to_csv(
        output_dir / "cross_evaluation_summary.csv",
    )

    print()
    print("=" * 110)
    print("PREDICTED RETURN SIMILARITY")
    print("=" * 110)

    print(
        similarity_df[
            [
                "seed",
                "flat_corr",
                "cosine",
                "mean_daily_cs_corr",
                "rmse_difference",
                "mean_top5_overlap",
                "down_mse",
                "up_mse",
            ]
        ].to_string(index=False)
    )

    print()
    print("Mean:")
    print(
        similarity_df[
            [
                "flat_corr",
                "cosine",
                "mean_daily_cs_corr",
                "rmse_difference",
                "mean_top5_overlap",
                "down_mse",
                "up_mse",
            ]
        ].mean().to_string()
    )

    print()
    print("=" * 110)
    print("2x2 CROSS-EVALUATION — 5-SEED MEAN ± STD")
    print("=" * 110)
    print(summary_df.to_string())

    print()
    print("=" * 110)
    print("SHARPE BY SEED")
    print("=" * 110)

    print(
        results_df.pivot(
            index="seed",
            columns="combination",
            values="sharpe",
        ).to_string()
    )

    print()
    print("Saved:")
    print(output_dir)


if __name__ == "__main__":
    main()