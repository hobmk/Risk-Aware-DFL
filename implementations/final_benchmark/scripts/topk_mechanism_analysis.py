from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from implementations.eigen_rcr_dfl.src.dataset import EigenRCRDataset
from implementations.final_benchmark.scripts.return_channel_analysis import (
    FINAL_ROOT,
    holm_adjust,
    lambda_name,
    load_dfl_asset_predictions,
    load_wide,
    safe_wilcoxon,
)
from implementations.rank_tail_rcr_dfl.scripts.validate_rank_tail_exposure import (
    future_fixed_weight_risk,
    hac_test,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 25.0])
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--topk-root", default=None)
    parser.add_argument(
        "--output-dir",
        default="implementations/final_benchmark/outputs/topk_mechanism",
    )
    return parser.parse_args()


def resolve_topk_root(args):
    if args.topk_root:
        path = Path(args.topk_root)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = [
        FINAL_ROOT / "topk",
        FINAL_ROOT / "top_k",
        FINAL_ROOT / "eigen_topk",
        FINAL_ROOT / "eigen_rcr",
    ]

    for path in candidates:
        check = path / "lambda_1" / "seed_42" / "predicted_returns.csv"
        if check.exists():
            return path

    raise FileNotFoundError(
        "Top-K final output을 찾지 못했습니다. "
        "--topk-root로 직접 지정하세요."
    )


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan

    return float(spearmanr(x, y).statistic)


def quadratic_exposure(weights, matrices):
    return np.einsum(
        "bi,bij,bj->b",
        weights,
        matrices,
        weights,
        optimize=True,
    )


def build_topk_maps(args):
    dataset = EigenRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        eigen_k=1,
        scale_mode="none",
        dtype=torch.float64,
    )

    matrix_map = {}
    residual_map = {}
    loading_map = {}

    for i, date in enumerate(dataset.target_dates):
        date = pd.Timestamp(date).normalize()

        top1 = (
            dataset.eigen_risk[i]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        residual = (
            dataset.residual_covariances[i]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        matrix_map[date] = top1
        residual_map[date] = residual

        # lambda_1 * u_i^2
        loading_map[date] = np.diag(top1).copy()

    return list(dataset.tickers), matrix_map, residual_map, loading_map


def load_dfl(seed, tickers):
    path = (
        FINAL_ROOT
        / "dfl_mvo"
        / "lambda_1"
        / f"seed_{seed}"
        / "asset_predictions.csv"
    )

    data = load_dfl_asset_predictions(path, tickers)

    return (
        data["weights"],
        data["true_returns"],
    )


def load_models(seed, lam, tickers, topk_root):
    dfl = load_dfl_asset_predictions(
        FINAL_ROOT
        / "dfl_mvo"
        / lambda_name(lam)
        / f"seed_{seed}"
        / "asset_predictions.csv",
        tickers,
    )

    topk_dir = (
        topk_root
        / lambda_name(lam)
        / f"seed_{seed}"
    )

    topk_pred = load_wide(
        topk_dir / "predicted_returns.csv",
        tickers,
    )

    topk_weight = load_wide(
        topk_dir / "portfolio_weights.csv",
        tickers,
    )

    topk_true = load_wide(
        topk_dir / "true_returns.csv",
        tickers,
    )

    dates = dfl["predicted_returns"].index

    for df in [
        dfl["weights"],
        dfl["true_returns"],
        topk_pred,
        topk_weight,
        topk_true,
    ]:
        dates = dates.intersection(df.index)

    return {
        "dates": dates.sort_values(),
        "dfl_pred": dfl["predicted_returns"],
        "dfl_weight": dfl["weights"],
        "dfl_true": dfl["true_returns"],
        "topk_pred": topk_pred,
        "topk_weight": topk_weight,
        "topk_true": topk_true,
    }


# =============================================================================
# 1. Top-K exposure -> future realized risk
# =============================================================================

def risk_validity(
    args,
    tickers,
    matrix_map,
    residual_map,
    output_dir,
):
    all_summaries = []

    for horizon in args.horizons:
        hac_lags = horizon - 1
        daily_frames = []
        test_rows = []

        for seed in args.seeds:
            weights_df, returns_df = load_dfl(
                seed,
                tickers,
            )

            dates = (
                weights_df.index
                .intersection(returns_df.index)
                .sort_values()
            )

            dates = pd.DatetimeIndex(dates).normalize()

            matrices = np.stack(
                [matrix_map[d] for d in dates]
            )

            residuals = np.stack(
                [residual_map[d] for d in dates]
            )

            weights = weights_df.loc[dates].to_numpy()
            returns = returns_df.loc[dates].to_numpy()

            topk_total = quadratic_exposure(
                weights,
                matrices,
            )

            residual_total = quadratic_exposure(
                weights,
                residuals,
            )

            topk_share = np.divide(
                topk_total,
                residual_total,
                out=np.full_like(topk_total, np.nan),
                where=np.abs(residual_total) > 1e-15,
            )

            (
                future_variance,
                future_volatility,
                future_mdd,
            ) = future_fixed_weight_risk(
                weights=weights,
                returns=returns,
                horizon=horizon,
                periods_per_year=252,
            )

            daily = pd.DataFrame({
                "seed": seed,
                "date": dates,
                "topk_total_exposure": topk_total,
                "topk_share": topk_share,
                "future_variance": future_variance,
                "future_volatility": future_volatility,
                "future_mdd": future_mdd,
            })

            daily_frames.append(daily)

            for exposure in [
                "topk_total_exposure",
                "topk_share",
            ]:
                for future in [
                    "future_variance",
                    "future_volatility",
                    "future_mdd",
                ]:
                    result = hac_test(
                        daily,
                        exposure,
                        future,
                        hac_lags,
                    )

                    test_rows.append({
                        "horizon": horizon,
                        "seed": seed,
                        "exposure_metric": exposure,
                        "future_metric": future,
                        **result,
                    })

        daily_all = pd.concat(
            daily_frames,
            ignore_index=True,
        )

        tests = pd.DataFrame(test_rows)

        tests["significant_positive"] = (
            (tests["hac_beta_q4"] > 0)
            & (tests["hac_p_q4"] < 0.05)
        ).astype(int)

        summary = (
            tests.groupby(
                [
                    "horizon",
                    "exposure_metric",
                    "future_metric",
                ],
                as_index=False,
            )
            .agg(
                mean_increase_pct=("increase_pct", "mean"),
                std_increase_pct=("increase_pct", "std"),
                positive_direction_seeds=(
                    "increase_pct",
                    lambda x: int((x > 0).sum()),
                ),
                significant_positive_seeds=(
                    "significant_positive",
                    "sum",
                ),
                mean_hac_beta_q4=(
                    "hac_beta_q4",
                    "mean",
                ),
            )
        )

        horizon_dir = (
            output_dir
            / f"risk_validity_h{horizon}"
        )

        horizon_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        daily_all.to_csv(
            horizon_dir / "topk_future_risk_daily.csv",
            index=False,
        )

        tests.to_csv(
            horizon_dir / "topk_future_risk_by_seed.csv",
            index=False,
        )

        summary.to_csv(
            horizon_dir / "topk_future_risk_summary.csv",
            index=False,
        )

        all_summaries.append(summary)

    result = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    print()
    print("=" * 110)
    print("TOP-K EXPOSURE -> FUTURE RISK")
    print("=" * 110)
    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    return result


# =============================================================================
# 2. Top-K prediction / weight mechanism
# =============================================================================

def prediction_weight_mechanism(
    args,
    tickers,
    loading_map,
    topk_root,
    output_dir,
):
    rows = []

    for lam in args.lambdas:
        for seed in args.seeds:
            run = load_models(
                seed,
                lam,
                tickers,
                topk_root,
            )

            for date in run["dates"]:
                date = pd.Timestamp(date).normalize()

                if date not in loading_map:
                    continue

                dfl_pred = run["dfl_pred"].loc[date].to_numpy()
                topk_pred = run["topk_pred"].loc[date].to_numpy()

                dfl_weight = run["dfl_weight"].loc[date].to_numpy()
                topk_weight = run["topk_weight"].loc[date].to_numpy()

                true_dfl = run["dfl_true"].loc[date].to_numpy()
                true_topk = run["topk_true"].loc[date].to_numpy()

                if not np.allclose(
                    true_dfl,
                    true_topk,
                    atol=1e-8,
                    rtol=0,
                ):
                    raise RuntimeError(
                        f"True-return mismatch: "
                        f"lambda={lam}, seed={seed}, date={date}"
                    )

                true_return = (
                    true_dfl + true_topk
                ) / 2.0

                loading = loading_map[date]

                pred_diff = (
                    topk_pred - dfl_pred
                )

                weight_diff = (
                    topk_weight - dfl_weight
                )

                error_dfl = np.abs(
                    dfl_pred - true_return
                )

                error_topk = np.abs(
                    topk_pred - true_return
                )

                error_improvement = (
                    error_dfl - error_topk
                )

                high = (
                    loading
                    >= np.quantile(
                        loading,
                        0.75,
                    )
                )

                low = ~high

                rows.append({
                    "lambda": lam,
                    "seed": seed,
                    "date": date,

                    "loading_pred_diff_rho":
                        safe_spearman(
                            loading,
                            pred_diff,
                        ),

                    "loading_weight_diff_rho":
                        safe_spearman(
                            loading,
                            weight_diff,
                        ),

                    "pred_diff_weight_diff_rho":
                        safe_spearman(
                            pred_diff,
                            weight_diff,
                        ),

                    "loading_true_return_rho":
                        safe_spearman(
                            loading,
                            true_return,
                        ),

                    "loading_error_improvement_rho":
                        safe_spearman(
                            loading,
                            error_improvement,
                        ),

                    "high_minus_low_pred_diff":
                        float(
                            pred_diff[high].mean()
                            - pred_diff[low].mean()
                        ),

                    "high_minus_low_weight_diff":
                        float(
                            weight_diff[high].mean()
                            - weight_diff[low].mean()
                        ),

                    "high_minus_low_true_return":
                        float(
                            true_return[high].mean()
                            - true_return[low].mean()
                        ),

                    "high_minus_low_error_improvement":
                        float(
                            error_improvement[high].mean()
                            - error_improvement[low].mean()
                        ),
                })

            print(
                f"DONE | lambda={lam:g} | seed={seed}"
            )

    daily = pd.DataFrame(rows)

    metrics = [
        "loading_pred_diff_rho",
        "loading_weight_diff_rho",
        "pred_diff_weight_diff_rho",
        "loading_true_return_rho",
        "loading_error_improvement_rho",
        "high_minus_low_pred_diff",
        "high_minus_low_weight_diff",
        "high_minus_low_true_return",
        "high_minus_low_error_improvement",
    ]

    by_seed = (
        daily.groupby(
            ["lambda", "seed"],
            as_index=False,
        )[metrics]
        .mean()
    )

    seed_invariant = {
        "loading_true_return_rho",
        "high_minus_low_true_return",
    }

    summary_rows = []

    for lam in args.lambdas:
        current = by_seed[
            by_seed["lambda"] == lam
        ]

        for metric in metrics:
            values = (
                current[metric]
                .dropna()
                .to_numpy()
            )

            p_raw = (
                np.nan
                if metric in seed_invariant
                else safe_wilcoxon(
                    values,
                    np.zeros_like(values),
                )
            )

            summary_rows.append({
                "lambda": lam,
                "metric": metric,
                "mean": np.mean(values),
                "median": np.median(values),
                "positive_seeds": int((values > 0).sum()),
                "negative_seeds": int((values < 0).sum()),
                "p_raw": p_raw,
            })

    summary = pd.DataFrame(
        summary_rows
    )

    summary["p_holm"] = np.nan

    for lam in args.lambdas:
        mask = (
            (summary["lambda"] == lam)
            & summary["p_raw"].notna()
        )

        summary.loc[
            mask,
            "p_holm",
        ] = holm_adjust(
            summary.loc[
                mask,
                "p_raw",
            ].to_numpy()
        )

    daily.to_csv(
        output_dir / "topk_prediction_weight_daily.csv",
        index=False,
    )

    by_seed.to_csv(
        output_dir / "topk_prediction_weight_by_seed.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "topk_prediction_weight_summary.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("TOP-K PREDICTION / WEIGHT MECHANISM")
    print("=" * 110)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    return summary


def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    topk_root = resolve_topk_root(
        args
    )

    print("=" * 110)
    print("TOP-K MECHANISM ANALYSIS")
    print("=" * 110)
    print(f"Top-K root : {topk_root}")
    print("K          : 1")
    print("Scale      : none")
    print(f"Seeds      : {args.seeds}")
    print(f"Lambdas    : {args.lambdas}")

    (
        tickers,
        matrix_map,
        residual_map,
        loading_map,
    ) = build_topk_maps(args)

    risk_validity(
        args,
        tickers,
        matrix_map,
        residual_map,
        output_dir,
    )

    prediction_weight_mechanism(
        args,
        tickers,
        loading_map,
        topk_root,
        output_dir,
    )


if __name__ == "__main__":
    main()