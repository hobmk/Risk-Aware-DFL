from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from implementations.final_benchmark.scripts.rcr_collective_cvar_test import (
    build_data,
    empirical_cvar,
    future_risk,
    load_dfl,
    max_drawdown,
    risk_reduction,
    solve_candidate,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--cvar-alpha", type=float, default=0.90)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.2)
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.05, 0.1])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    parser.add_argument(
        "--output-dir",
        default="implementations/final_benchmark/outputs/rcr_collective_cvar_20seed",
    )
    return parser.parse_args()


def drift_turnover(weights, asset_returns):
    turnover = np.zeros(len(weights), dtype=np.float64)

    for t in range(1, len(weights)):
        prev_w = weights[t - 1]
        prev_r = asset_returns[t - 1]
        denominator = 1.0 + np.dot(prev_w, prev_r)

        if abs(denominator) <= 1e-15:
            drifted = prev_w
        else:
            drifted = prev_w * (1.0 + prev_r) / denominator

        turnover[t] = 0.5 * np.abs(weights[t] - drifted).sum()

    return turnover


def metrics_from_returns(returns, alpha):
    returns = np.asarray(returns, dtype=np.float64)
    wealth = np.cumprod(1.0 + returns)

    annual_return = wealth[-1] ** (252.0 / len(returns)) - 1.0
    volatility = returns.std(ddof=1) * np.sqrt(252.0)

    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
        if returns.std(ddof=1) > 1e-15 else np.nan
    )

    return {
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "mdd": max_drawdown(returns),
        "cvar": empirical_cvar(returns, alpha),
    }


def evaluate_portfolio(weights, asset_returns, alpha):
    gross = np.sum(weights * asset_returns, axis=1)
    turnover = drift_turnover(weights, asset_returns)

    net_5bp = gross - 0.0005 * turnover
    net_10bp = gross - 0.0010 * turnover

    gross_metrics = metrics_from_returns(gross, alpha)
    metrics_5bp = metrics_from_returns(net_5bp, alpha)
    metrics_10bp = metrics_from_returns(net_10bp, alpha)

    return {
        **gross_metrics,
        "annualized_turnover": float(turnover.mean() * 252.0),
        "annual_return_5bp": metrics_5bp["annual_return"],
        "sharpe_5bp": metrics_5bp["sharpe"],
        "annual_return_10bp": metrics_10bp["annual_return"],
        "sharpe_10bp": metrics_10bp["sharpe"],
    }


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    result = np.full_like(pvalues, np.nan)

    valid = np.isfinite(pvalues)
    values = pvalues[valid]

    if len(values) == 0:
        return result

    order = np.argsort(values)
    ordered = values[order]
    adjusted = np.empty_like(ordered)

    running = 0.0
    m = len(ordered)

    for i, p in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[i] = running

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid] = restored

    return result


def paired_p(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    if np.allclose(values, 0.0):
        return 1.0

    try:
        return float(wilcoxon(values).pvalue)
    except ValueError:
        return np.nan


def build_paired_tests(by_seed, gammas):
    metrics = {
        "annual_return": "higher",
        "volatility": "lower",
        "sharpe": "higher",
        "mdd": "lower",
        "cvar": "lower",
        "annualized_turnover": "lower",
        "sharpe_5bp": "higher",
        "sharpe_10bp": "higher",
        "future_vol_reduction_h20": "higher",
        "future_mdd_reduction_h20": "higher",
        "future_cvar_reduction_h20": "higher",
    }

    rows = []

    for gamma in gammas:
        candidate = by_seed[
            (by_seed["method"] == "collective_rcr_cvar")
            & np.isclose(by_seed["gamma"], gamma)
        ]

        references = {
            "dfl_mvo": by_seed[by_seed["method"] == "dfl_mvo"],
            "uniform_residual_cvar": by_seed[
                (by_seed["method"] == "uniform_residual_cvar")
                & np.isclose(by_seed["gamma"], gamma)
            ],
            "rcr_weighted_cvar": by_seed[
                (by_seed["method"] == "rcr_weighted_cvar")
                & np.isclose(by_seed["gamma"], gamma)
            ],
        }

        for reference_name, reference in references.items():
            merged = candidate.merge(
                reference,
                on="seed",
                suffixes=("_candidate", "_reference"),
            )

            for metric, direction in metrics.items():
                cand = merged[f"{metric}_candidate"].to_numpy(dtype=np.float64)
                ref = merged[f"{metric}_reference"].to_numpy(dtype=np.float64)

                improvement = cand - ref if direction == "higher" else ref - cand

                rows.append({
                    "gamma": gamma,
                    "candidate": "collective_rcr_cvar",
                    "reference": reference_name,
                    "metric": metric,
                    "mean_improvement": float(np.nanmean(improvement)),
                    "median_improvement": float(np.nanmedian(improvement)),
                    "candidate_better_seeds": int(np.nansum(improvement > 0)),
                    "reference_better_seeds": int(np.nansum(improvement < 0)),
                    "p_raw": paired_p(improvement),
                })

    tests = pd.DataFrame(rows)
    tests["p_holm"] = np.nan

    for (_, metric), group in tests.groupby(["gamma", "metric"]):
        tests.loc[group.index, "p_holm"] = holm_adjust(
            group["p_raw"].to_numpy(dtype=np.float64)
        )

    return tests


def build_summary(by_seed):
    metrics = [
        "annual_return",
        "volatility",
        "sharpe",
        "mdd",
        "cvar",
        "annualized_turnover",
        "annual_return_5bp",
        "sharpe_5bp",
        "annual_return_10bp",
        "sharpe_10bp",
        "future_vol_reduction_h20",
        "future_mdd_reduction_h20",
        "future_cvar_reduction_h20",
    ]

    summary = (
        by_seed.groupby(["method", "gamma"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        tickers,
        date_to_index,
        base_covariance,
        residuals,
        rcr_probabilities,
        collective_probabilities,
    ) = build_data(args)

    uniform_probability = np.full(
        args.lookback,
        1.0 / args.lookback,
        dtype=np.float64,
    )

    rows = []

    print("=" * 120)
    print("20-SEED RESIDUAL CVaR VALIDATION")
    print("=" * 120)
    print(f"lambda     : {args.lambda_value:g}")
    print(f"tail_q     : {args.tail_q:g}")
    print(f"CVaR alpha : {args.cvar_alpha:g}")
    print(f"gammas     : {args.gammas}")
    print(f"seeds      : {args.seeds}")

    for seed in args.seeds:
        dfl = load_dfl(seed, args.lambda_value, tickers)

        dates = pd.DatetimeIndex(dfl["predictions"].index).normalize()
        indices = np.asarray([date_to_index[d] for d in dates], dtype=np.int64)

        predictions = dfl["predictions"].to_numpy(dtype=np.float64)
        baseline_weights = dfl["weights"].to_numpy(dtype=np.float64)
        returns = dfl["returns"].to_numpy(dtype=np.float64)

        covariance = base_covariance[indices]
        scenarios = residuals[indices]
        rcr_p = rcr_probabilities[indices]
        collective_p = collective_probabilities[indices]

        uniform_p = np.broadcast_to(
            uniform_probability,
            (len(dates), args.lookback),
        )

        baseline_metrics = evaluate_portfolio(
            baseline_weights,
            returns,
            args.cvar_alpha,
        )

        baseline_future = future_risk(
            baseline_weights,
            returns,
            horizon=20,
            alpha=args.cvar_alpha,
        )

        rows.append({
            "seed": seed,
            "method": "dfl_mvo",
            "gamma": 0.0,
            **baseline_metrics,
            "future_vol_reduction_h20": 0.0,
            "future_mdd_reduction_h20": 0.0,
            "future_cvar_reduction_h20": 0.0,
        })

        method_probabilities = {
            "uniform_residual_cvar": uniform_p,
            "rcr_weighted_cvar": rcr_p,
            "collective_rcr_cvar": collective_p,
        }

        for gamma in args.gammas:
            for method, probabilities in method_probabilities.items():
                weights = solve_candidate(
                    predictions=predictions,
                    base_covariance=covariance,
                    residuals=scenarios,
                    probabilities=probabilities,
                    gamma=gamma,
                    args=args,
                )

                metrics = evaluate_portfolio(
                    weights,
                    returns,
                    args.cvar_alpha,
                )

                candidate_future = future_risk(
                    weights,
                    returns,
                    horizon=20,
                    alpha=args.cvar_alpha,
                )

                row = {
                    "seed": seed,
                    "method": method,
                    "gamma": gamma,
                    **metrics,
                    "future_vol_reduction_h20": risk_reduction(
                        baseline_future["volatility"],
                        candidate_future["volatility"],
                    ),
                    "future_mdd_reduction_h20": risk_reduction(
                        baseline_future["mdd"],
                        candidate_future["mdd"],
                    ),
                    "future_cvar_reduction_h20": risk_reduction(
                        baseline_future["cvar"],
                        candidate_future["cvar"],
                    ),
                }

                rows.append(row)

                print(
                    f"seed={seed} | {method:23s} | γ={gamma:.2f} | "
                    f"Sharpe={row['sharpe']:.4f} | "
                    f"S5={row['sharpe_5bp']:.4f} | "
                    f"S10={row['sharpe_10bp']:.4f} | "
                    f"MDD={row['mdd']:.4f} | "
                    f"FCVaR20↓={row['future_cvar_reduction_h20']:.3f}%"
                )

    by_seed = pd.DataFrame(rows)
    summary = build_summary(by_seed)
    tests = build_paired_tests(by_seed, args.gammas)

    by_seed.to_csv(output_dir / "by_seed.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    tests.to_csv(output_dir / "paired_tests.csv", index=False)

    print()
    print("=" * 120)
    print("MEAN ± STD")
    print("=" * 120)

    key = summary[
        [
            "method",
            "gamma",
            "annual_return_mean",
            "annual_return_std",
            "volatility_mean",
            "volatility_std",
            "sharpe_mean",
            "sharpe_std",
            "mdd_mean",
            "mdd_std",
            "sharpe_5bp_mean",
            "sharpe_10bp_mean",
            "future_cvar_reduction_h20_mean",
        ]
    ]

    print(
        key.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 120)
    print("COLLECTIVE RCR — PAIRED TESTS")
    print("=" * 120)

    important = tests[
        tests["metric"].isin([
            "annual_return",
            "volatility",
            "sharpe",
            "mdd",
            "sharpe_5bp",
            "sharpe_10bp",
            "future_cvar_reduction_h20",
        ])
    ]

    print(
        important.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("Saved:")
    print(output_dir / "by_seed.csv")
    print(output_dir / "summary.csv")
    print(output_dir / "paired_tests.csv")


if __name__ == "__main__":
    main()