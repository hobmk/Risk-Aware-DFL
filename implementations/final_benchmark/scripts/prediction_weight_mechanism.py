from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from implementations.final_benchmark.scripts.return_channel_analysis import (
    FINAL_ROOT,
    holm_adjust,
    lambda_name,
    load_dfl_asset_predictions,
    load_wide,
    safe_wilcoxon,
)
from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 25.0])
    parser.add_argument(
        "--output-dir",
        default="implementations/final_benchmark/outputs/prediction_weight_mechanism",
    )
    return parser.parse_args()


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan

    return float(spearmanr(x, y).statistic)


def build_centrality_map(args):
    dataset = RankTailRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        tail_q=args.tail_q,
        tail_side="down",
    )

    tickers = list(dataset.tickers)
    centrality_map = {}

    for i, date in enumerate(dataset.target_dates):
        dependence = (
            dataset.rank_tail_dependence[i]
            .detach().cpu().numpy().astype(np.float64)
        )

        np.fill_diagonal(dependence, 0.0)

        # 양의 downside tail dependence가 강한 종목일수록 큰 값
        centrality = np.maximum(dependence, 0.0).sum(axis=1)

        centrality_map[pd.Timestamp(date).normalize()] = centrality

    return tickers, centrality_map


def load_run(seed, lam, tickers):
    dfl_dir = FINAL_ROOT / "dfl_mvo" / lambda_name(lam) / f"seed_{seed}"
    rt_dir = FINAL_ROOT / "rank_tail" / lambda_name(lam) / f"seed_{seed}"

    dfl = load_dfl_asset_predictions(
        dfl_dir / "asset_predictions.csv",
        tickers,
    )

    rt_pred = load_wide(rt_dir / "predicted_returns.csv", tickers)
    rt_weight = load_wide(rt_dir / "portfolio_weights.csv", tickers)
    rt_true = load_wide(rt_dir / "true_returns.csv", tickers)

    dates = dfl["predicted_returns"].index

    for df in [
        dfl["weights"],
        dfl["true_returns"],
        rt_pred,
        rt_weight,
        rt_true,
    ]:
        dates = dates.intersection(df.index)

    return {
        "dates": dates.sort_values(),
        "dfl_pred": dfl["predicted_returns"],
        "dfl_weight": dfl["weights"],
        "dfl_true": dfl["true_returns"],
        "rt_pred": rt_pred,
        "rt_weight": rt_weight,
        "rt_true": rt_true,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers, centrality_map = build_centrality_map(args)

    daily_rows = []

    for lam in args.lambdas:
        for seed in args.seeds:
            run = load_run(seed, lam, tickers)

            for date in run["dates"]:
                if date not in centrality_map:
                    continue

                dfl_pred = run["dfl_pred"].loc[date].to_numpy()
                rt_pred = run["rt_pred"].loc[date].to_numpy()

                dfl_weight = run["dfl_weight"].loc[date].to_numpy()
                rt_weight = run["rt_weight"].loc[date].to_numpy()

                true_dfl = run["dfl_true"].loc[date].to_numpy()
                true_rt = run["rt_true"].loc[date].to_numpy()

                if not np.allclose(true_dfl, true_rt, atol=1e-8, rtol=0):
                    raise RuntimeError(
                        f"True-return mismatch: lambda={lam}, seed={seed}, date={date}"
                    )

                true_return = 0.5 * (true_dfl + true_rt)
                centrality = centrality_map[date]

                pred_diff = rt_pred - dfl_pred
                weight_diff = rt_weight - dfl_weight

                error_dfl = np.abs(dfl_pred - true_return)
                error_rt = np.abs(rt_pred - true_return)
                error_improvement = error_dfl - error_rt

                q75 = np.quantile(centrality, 0.75)
                high = centrality >= q75
                low = ~high

                daily_rows.append({
                    "lambda": lam,
                    "seed": seed,
                    "date": date,

                    # 고위험 종목일수록 RT prediction이 더 낮아지는가?
                    "centrality_pred_diff_rho":
                        safe_spearman(centrality, pred_diff),

                    # 고위험 종목일수록 RT weight가 더 낮아지는가?
                    "centrality_weight_diff_rho":
                        safe_spearman(centrality, weight_diff),

                    # prediction 변화가 실제 weight 변화로 연결되는가?
                    "pred_diff_weight_diff_rho":
                        safe_spearman(pred_diff, weight_diff),

                    # 고위험 종목의 실제 다음 수익률이 낮은가?
                    "centrality_true_return_rho":
                        safe_spearman(centrality, true_return),

                    # 고위험 종목에서 prediction error가 개선되는가?
                    "centrality_error_improvement_rho":
                        safe_spearman(centrality, error_improvement),

                    # 상위 25% 위험 종목 - 나머지 종목
                    "high_minus_low_pred_diff":
                        float(pred_diff[high].mean() - pred_diff[low].mean()),

                    "high_minus_low_weight_diff":
                        float(weight_diff[high].mean() - weight_diff[low].mean()),

                    "high_minus_low_true_return":
                        float(true_return[high].mean() - true_return[low].mean()),

                    "high_minus_low_error_improvement":
                        float(
                            error_improvement[high].mean()
                            - error_improvement[low].mean()
                        ),
                })

            print(f"DONE | lambda={lam:g} | seed={seed}")

    daily = pd.DataFrame(daily_rows)

    metric_columns = [
        "centrality_pred_diff_rho",
        "centrality_weight_diff_rho",
        "pred_diff_weight_diff_rho",
        "centrality_true_return_rho",
        "centrality_error_improvement_rho",
        "high_minus_low_pred_diff",
        "high_minus_low_weight_diff",
        "high_minus_low_true_return",
        "high_minus_low_error_improvement",
    ]

    by_seed = (
        daily.groupby(["lambda", "seed"], as_index=False)[metric_columns]
        .mean()
    )

    summary_rows = []

    for lam in args.lambdas:
        current = by_seed[by_seed["lambda"] == lam]

        for metric in metric_columns:
            values = current[metric].dropna().to_numpy()

            p_raw = safe_wilcoxon(
                values,
                np.zeros_like(values),
            )

            summary_rows.append({
                "lambda": lam,
                "metric": metric,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "positive_seeds": int(np.sum(values > 0)),
                "negative_seeds": int(np.sum(values < 0)),
                "p_raw": p_raw,
            })

    summary = pd.DataFrame(summary_rows)
    summary["p_holm"] = np.nan

    for lam in args.lambdas:
        mask = summary["lambda"] == lam
        summary.loc[mask, "p_holm"] = holm_adjust(
            summary.loc[mask, "p_raw"].to_numpy()
        )

    daily.to_csv(output_dir / "prediction_weight_daily.csv", index=False)
    by_seed.to_csv(output_dir / "prediction_weight_by_seed.csv", index=False)
    summary.to_csv(output_dir / "prediction_weight_summary.csv", index=False)

    print()
    print("=" * 120)
    print("PREDICTION / WEIGHT MECHANISM SUMMARY")
    print("=" * 120)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("Saved:")
    print(output_dir / "prediction_weight_daily.csv")
    print(output_dir / "prediction_weight_by_seed.csv")
    print(output_dir / "prediction_weight_summary.csv")


if __name__ == "__main__":
    main()