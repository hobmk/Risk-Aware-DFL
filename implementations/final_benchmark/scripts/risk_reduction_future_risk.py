from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset
from implementations.rank_tail_rcr_dfl.scripts.validate_rank_tail_exposure import (
    future_fixed_weight_risk,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 25.0])
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20])
    parser.add_argument(
        "--output-dir",
        default="implementations/final_benchmark/outputs/risk_reduction_future_risk",
    )
    return parser.parse_args()


def quadratic_exposure(weights, matrices):
    return np.einsum(
        "bi,bij,bj->b",
        weights,
        matrices,
        weights,
        optimize=True,
    )


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan

    return float(spearmanr(x, y).statistic)


def hac_slope(x, y, lags):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 5 or np.std(x) < 1e-15:
        return np.nan, np.nan

    x = (x - x.mean()) / x.std(ddof=1)

    X = sm.add_constant(x)

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lags},
    )

    return float(model.params[1]), float(model.pvalues[1])


def build_risk_maps(args):
    rt_dataset = RankTailRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        tail_q=args.tail_q,
        tail_side="down",
        dtype=torch.float64,
    )

    topk_dataset = EigenRCRDataset(
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

    if rt_dataset.tickers != topk_dataset.tickers:
        raise RuntimeError("Rank-Tail / Top-K ticker mismatch.")

    tickers = list(rt_dataset.tickers)

    rt_map = {
        pd.Timestamp(date).normalize(): (
            rt_dataset.rank_tail_risk[i]
            .detach().cpu().numpy().astype(np.float64)
        )
        for i, date in enumerate(rt_dataset.target_dates)
    }

    topk_map = {
        pd.Timestamp(date).normalize(): (
            topk_dataset.eigen_risk[i]
            .detach().cpu().numpy().astype(np.float64)
        )
        for i, date in enumerate(topk_dataset.target_dates)
    }

    return tickers, {
        "rank_tail": rt_map,
        "topk": topk_map,
    }


def load_run(model, seed, lam, tickers):
    dfl = load_dfl_asset_predictions(
        FINAL_ROOT
        / "dfl_mvo"
        / lambda_name(lam)
        / f"seed_{seed}"
        / "asset_predictions.csv",
        tickers,
    )

    model_dir = (
        FINAL_ROOT
        / model
        / lambda_name(lam)
        / f"seed_{seed}"
    )

    model_weights = load_wide(
        model_dir / "portfolio_weights.csv",
        tickers,
    )

    model_true = load_wide(
        model_dir / "true_returns.csv",
        tickers,
    )

    dates = dfl["weights"].index

    for df in [
        dfl["true_returns"],
        model_weights,
        model_true,
    ]:
        dates = dates.intersection(df.index)

    dates = dates.sort_values()

    dfl_true = dfl["true_returns"].loc[dates].to_numpy()
    candidate_true = model_true.loc[dates].to_numpy()

    if not np.allclose(
        dfl_true,
        candidate_true,
        atol=1e-8,
        rtol=0,
    ):
        raise RuntimeError(
            f"True return mismatch: "
            f"model={model}, lambda={lam}, seed={seed}"
        )

    return {
        "dates": dates,
        "returns": 0.5 * (dfl_true + candidate_true),
        "dfl_weights": dfl["weights"].loc[dates].to_numpy(),
        "model_weights": model_weights.loc[dates].to_numpy(),
    }


def run_analysis(args, tickers, risk_maps):
    daily_frames = []
    seed_rows = []

    for model in ["rank_tail", "topk"]:
        matrix_map = risk_maps[model]

        for lam in args.lambdas:
            for seed in args.seeds:
                run = load_run(
                    model,
                    seed,
                    lam,
                    tickers,
                )

                dates = run["dates"]

                matrices = np.stack(
                    [
                        matrix_map[
                            pd.Timestamp(date).normalize()
                        ]
                        for date in dates
                    ],
                    axis=0,
                )

                dfl_exposure = quadratic_exposure(
                    run["dfl_weights"],
                    matrices,
                )

                model_exposure = quadratic_exposure(
                    run["model_weights"],
                    matrices,
                )

                # 양수 = 모델이 DFL보다 targeted exposure를 줄임
                exposure_reduction = (
                    dfl_exposure
                    - model_exposure
                )

                for horizon in args.horizons:
                    (
                        dfl_variance,
                        dfl_volatility,
                        dfl_mdd,
                    ) = future_fixed_weight_risk(
                        weights=run["dfl_weights"],
                        returns=run["returns"],
                        horizon=horizon,
                        periods_per_year=252,
                    )

                    (
                        model_variance,
                        model_volatility,
                        model_mdd,
                    ) = future_fixed_weight_risk(
                        weights=run["model_weights"],
                        returns=run["returns"],
                        horizon=horizon,
                        periods_per_year=252,
                    )

                    future = {
                        "variance": (
                            dfl_variance
                            - model_variance
                        ),
                        "volatility": (
                            dfl_volatility
                            - model_volatility
                        ),
                        "mdd": (
                            dfl_mdd
                            - model_mdd
                        ),
                    }

                    daily = pd.DataFrame({
                        "model": model,
                        "lambda": lam,
                        "seed": seed,
                        "horizon": horizon,
                        "date": dates,
                        "dfl_exposure": dfl_exposure,
                        "model_exposure": model_exposure,
                        "exposure_reduction": exposure_reduction,
                        "future_variance_reduction": future["variance"],
                        "future_volatility_reduction": future["volatility"],
                        "future_mdd_reduction": future["mdd"],
                    })

                    daily_frames.append(daily)

                    for risk_name, risk_reduction in future.items():
                        valid = (
                            np.isfinite(exposure_reduction)
                            & np.isfinite(risk_reduction)
                        )

                        x = exposure_reduction[valid]
                        y = risk_reduction[valid]

                        reduced = x > 0

                        beta, hac_p = hac_slope(
                            x,
                            y,
                            horizon - 1,
                        )

                        seed_rows.append({
                            "model": model,
                            "lambda": lam,
                            "seed": seed,
                            "horizon": horizon,
                            "future_metric": risk_name,

                            "mean_exposure_reduction":
                                float(np.mean(x)),

                            "exposure_reduction_days_pct":
                                float(100 * np.mean(reduced)),

                            "mean_future_risk_reduction":
                                float(np.mean(y)),

                            "mean_future_risk_reduction_when_exposure_reduced":
                                (
                                    float(np.mean(y[reduced]))
                                    if reduced.any()
                                    else np.nan
                                ),

                            "positive_future_reduction_when_exposure_reduced_pct":
                                (
                                    float(
                                        100
                                        * np.mean(
                                            y[reduced] > 0
                                        )
                                    )
                                    if reduced.any()
                                    else np.nan
                                ),

                            "delta_exposure_future_risk_spearman":
                                safe_spearman(
                                    x,
                                    y,
                                ),

                            "hac_beta_per_1sd_exposure_reduction":
                                beta,

                            "hac_p":
                                hac_p,
                        })

                print(
                    f"DONE | {model} | "
                    f"lambda={lam:g} | seed={seed}"
                )

    return (
        pd.concat(
            daily_frames,
            ignore_index=True,
        ),
        pd.DataFrame(seed_rows),
    )


def build_summary(by_seed):
    rows = []

    group_cols = [
        "model",
        "lambda",
        "horizon",
        "future_metric",
    ]

    for keys, group in by_seed.groupby(group_cols):
        model, lam, horizon, metric = keys

        for test_metric in [
            "mean_future_risk_reduction_when_exposure_reduced",
            "delta_exposure_future_risk_spearman",
            "hac_beta_per_1sd_exposure_reduction",
        ]:
            values = (
                group[test_metric]
                .dropna()
                .to_numpy()
            )

            p_raw = safe_wilcoxon(
                values,
                np.zeros_like(values),
            )

            rows.append({
                "model": model,
                "lambda": lam,
                "horizon": horizon,
                "future_metric": metric,
                "test_metric": test_metric,

                "mean":
                    float(np.mean(values)),

                "median":
                    float(np.median(values)),

                "positive_seeds":
                    int(np.sum(values > 0)),

                "negative_seeds":
                    int(np.sum(values < 0)),

                "p_raw":
                    p_raw,
            })

    summary = pd.DataFrame(rows)
    summary["p_holm"] = np.nan

    for keys, group in summary.groupby(
        ["model", "lambda", "horizon"]
    ):
        idx = group.index

        summary.loc[
            idx,
            "p_holm",
        ] = holm_adjust(
            group["p_raw"].to_numpy()
        )

    return summary


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print("TARGETED EXPOSURE REDUCTION -> FUTURE RISK REDUCTION")
    print("=" * 110)

    tickers, risk_maps = build_risk_maps(
        args
    )

    daily, by_seed = run_analysis(
        args,
        tickers,
        risk_maps,
    )

    summary = build_summary(
        by_seed
    )

    daily.to_csv(
        output_dir / "daily.csv",
        index=False,
    )

    by_seed.to_csv(
        output_dir / "by_seed.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("Saved:")
    print(output_dir / "daily.csv")
    print(output_dir / "by_seed.csv")
    print(output_dir / "summary.csv")


if __name__ == "__main__":
    main()