from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon

from implementations.dfl_mvo_lee2025.src.dataset import RollingMVODataset
from implementations.eigen_rcr_dfl.src.optimization import (
    build_markowitz_layer,
    covariance_to_risk_factor,
)
from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset


TRADING_DAYS = 252
FINAL_ROOT = Path("implementations/final_benchmark/outputs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--max-weight", type=float, default=0.2)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 25.0])
    parser.add_argument(
        "--output-dir",
        default="implementations/final_benchmark/outputs/return_channel_analysis",
    )
    return parser.parse_args()


def lambda_name(value: float) -> str:
    return f"lambda_{value:g}"


def load_wide(path: Path, tickers: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else "Date"

    if date_col not in df.columns:
        raise KeyError(f"No date column: {path}")

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    result = (
        df.set_index(date_col)
        .sort_index()
        .reindex(columns=tickers)
    )

    if result.isna().any().any():
        raise RuntimeError(f"NaN after loading: {path}")

    return result.astype(np.float64)


def load_dfl_asset_predictions(
    path: Path,
    tickers: list[str],
) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "date",
        "ticker",
        "predicted_return",
        "predicted_weight",
        "true_return",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Missing columns in {path}: {sorted(missing)}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)

    def pivot(value: str) -> pd.DataFrame:
        result = (
            df.pivot(index="date", columns="ticker", values=value)
            .sort_index()
            .reindex(columns=tickers)
        )

        if result.isna().any().any():
            raise RuntimeError(f"NaN after pivoting {value}: {path}")

        return result.astype(np.float64)

    return {
        "predicted_returns": pivot("predicted_return"),
        "weights": pivot("predicted_weight"),
        "true_returns": pivot("true_return"),
    }


def portfolio_metrics(
    weights: np.ndarray,
    returns: np.ndarray,
) -> dict[str, float]:
    p = np.sum(weights * returns, axis=1)
    wealth = np.cumprod(1.0 + p)

    annualized_return = (
        wealth[-1] ** (TRADING_DAYS / len(p)) - 1.0
        if wealth[-1] > 0
        else np.nan
    )

    volatility = np.std(p, ddof=1) * np.sqrt(TRADING_DAYS)

    sharpe = (
        np.mean(p) / np.std(p, ddof=1) * np.sqrt(TRADING_DAYS)
        if np.std(p, ddof=1) > 0
        else np.nan
    )

    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    mdd = -np.min(drawdown)

    return {
        "annualized_return": float(annualized_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
    }


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a) - np.asarray(b)

    if np.allclose(diff, 0.0):
        return 1.0

    return float(
        wilcoxon(
            a,
            b,
            alternative="two-sided",
            zero_method="wilcox",
        ).pvalue
    )


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)

    running = 0.0
    m = len(p_values)

    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)

    return adjusted


def build_risk_maps(args):
    dfl_dataset = RollingMVODataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float64,
    )

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

    if dfl_dataset.tickers != rt_dataset.tickers:
        raise RuntimeError("DFL / Rank-Tail ticker order mismatch.")

    tickers = list(dfl_dataset.tickers)
    dfl_map = {}
    rt_map = {}

    eye = np.eye(len(tickers), dtype=np.float64)

    for i, date in enumerate(dfl_dataset.target_dates):
        pos = int(dfl_dataset.target_positions[i])
        window = dfl_dataset.returns[pos - args.lookback:pos]

        cov = np.cov(window, rowvar=False, ddof=1)
        cov = 0.5 * (cov + cov.T) + 1e-6 * eye

        dfl_map[pd.Timestamp(date).normalize()] = cov

    for i, date in enumerate(rt_dataset.target_dates):
        rt_map[pd.Timestamp(date).normalize()] = (
            rt_dataset.covariances[i]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    return tickers, dfl_map, rt_map


def solve_weights(
    layer,
    predictions: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    mu = torch.tensor(predictions, dtype=torch.float64)
    covariance_tensor = torch.tensor(covariance, dtype=torch.float64)

    risk_factor = covariance_to_risk_factor(
        covariance=covariance_tensor,
        risk_aversion=risk_aversion,
        jitter=0.0,
    )

    with torch.no_grad():
        weights, = layer(mu, risk_factor)

    return (
        weights.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def aligned_run_data(
    seed: int,
    lam: float,
    tickers: list[str],
):
    dfl_dir = (
        FINAL_ROOT
        / "dfl_mvo"
        / lambda_name(lam)
        / f"seed_{seed}"
    )

    rt_dir = (
        FINAL_ROOT
        / "rank_tail"
        / lambda_name(lam)
        / f"seed_{seed}"
    )

    # DFL-MVO는 asset_predictions.csv long-format
    dfl = load_dfl_asset_predictions(
        dfl_dir / "asset_predictions.csv",
        tickers,
    )

    dfl_pred = dfl["predicted_returns"]
    dfl_weights = dfl["weights"]
    dfl_true = dfl["true_returns"]

    # Rank-Tail은 separate wide-format
    rt_pred = load_wide(
        rt_dir / "predicted_returns.csv",
        tickers,
    )

    rt_weights = load_wide(
        rt_dir / "portfolio_weights.csv",
        tickers,
    )

    rt_true = load_wide(
        rt_dir / "true_returns.csv",
        tickers,
    )

    dates = dfl_pred.index

    for df in [
        dfl_weights,
        dfl_true,
        rt_pred,
        rt_weights,
        rt_true,
    ]:
        dates = dates.intersection(df.index)

    dates = dates.sort_values()

    if len(dates) == 0:
        raise RuntimeError(
            f"No common dates: lambda={lam}, seed={seed}"
        )

    target_dfl = dfl_true.loc[dates].to_numpy()
    target_rt = rt_true.loc[dates].to_numpy()

    if not np.allclose(
        target_dfl,
        target_rt,
        atol=1e-7,
        rtol=0,
    ):
        max_diff = float(
            np.max(np.abs(target_dfl - target_rt))
        )

        raise RuntimeError(
            f"True return mismatch: "
            f"lambda={lam}, seed={seed}, "
            f"max_diff={max_diff:.3e}"
        )

    return {
        "dates": dates,
        "dfl_pred": dfl_pred.loc[dates].to_numpy(),
        "rt_pred": rt_pred.loc[dates].to_numpy(),
        "dfl_weights": dfl_weights.loc[dates].to_numpy(),
        "rt_weights": rt_weights.loc[dates].to_numpy(),
        "target": target_dfl,
    }


# =============================================================================
# Experiment 1
# Return / Volatility / MDD decomposition
# =============================================================================

def experiment_1(args, tickers):
    rows = []

    for lam in args.lambdas:
        for seed in args.seeds:
            data = aligned_run_data(seed, lam, tickers)

            for model, weights in [
                ("dfl_mvo", data["dfl_weights"]),
                ("rank_tail", data["rt_weights"]),
            ]:
                metrics = portfolio_metrics(
                    weights,
                    data["target"],
                )

                rows.append({
                    "lambda": lam,
                    "seed": seed,
                    "model": model,
                    **metrics,
                })

    by_seed = pd.DataFrame(rows)
    summary_rows = []

    for lam in args.lambdas:
        current = by_seed[
            by_seed["lambda"] == lam
        ]

        for metric in [
            "annualized_return",
            "volatility",
            "sharpe",
            "max_drawdown",
        ]:
            dfl = (
                current[
                    current["model"] == "dfl_mvo"
                ]
                .sort_values("seed")[metric]
                .to_numpy()
            )

            rt = (
                current[
                    current["model"] == "rank_tail"
                ]
                .sort_values("seed")[metric]
                .to_numpy()
            )

            diff = rt - dfl

            lower_is_better = metric in {
                "volatility",
                "max_drawdown",
            }

            better = (
                diff < 0
                if lower_is_better
                else diff > 0
            )

            summary_rows.append({
                "lambda": lam,
                "metric": metric,
                "dfl_mean": np.mean(dfl),
                "rank_tail_mean": np.mean(rt),
                "mean_difference_rt_minus_dfl": np.mean(diff),
                "median_difference_rt_minus_dfl": np.median(diff),
                "rank_tail_better_seeds": int(np.sum(better)),
                "p_raw": safe_wilcoxon(rt, dfl),
            })

    summary = pd.DataFrame(summary_rows)
    summary["p_holm"] = np.nan

    for lam in args.lambdas:
        mask = summary["lambda"] == lam

        summary.loc[mask, "p_holm"] = holm_adjust(
            summary.loc[mask, "p_raw"].to_numpy()
        )

    return by_seed, summary


# =============================================================================
# Experiment 2
# Prediction × Risk Matrix 2×2 swap
# =============================================================================

def experiment_2(
    args,
    tickers,
    dfl_map,
    rt_map,
):
    layer = build_markowitz_layer(
        n_assets=len(tickers),
        max_weight=args.max_weight,
    )

    result_rows = []
    reproduction_rows = []

    for lam in args.lambdas:
        for seed in args.seeds:
            data = aligned_run_data(
                seed,
                lam,
                tickers,
            )

            dates = data["dates"]

            if not all(
                date in dfl_map
                and date in rt_map
                for date in dates
            ):
                raise RuntimeError(
                    f"Risk matrix date mismatch: "
                    f"lambda={lam}, seed={seed}"
                )

            dfl_cov = np.stack(
                [dfl_map[d] for d in dates],
                axis=0,
            )

            rt_cov = np.stack(
                [rt_map[d] for d in dates],
                axis=0,
            )

            combinations = {
                "DFLpred_DFLrisk": (
                    data["dfl_pred"],
                    dfl_cov,
                ),
                "DFLpred_RTrisk": (
                    data["dfl_pred"],
                    rt_cov,
                ),
                "RTpred_DFLrisk": (
                    data["rt_pred"],
                    dfl_cov,
                ),
                "RTpred_RTrisk": (
                    data["rt_pred"],
                    rt_cov,
                ),
            }

            solved = {}

            for name, (
                prediction,
                covariance,
            ) in combinations.items():

                weights = solve_weights(
                    layer=layer,
                    predictions=prediction,
                    covariance=covariance,
                    risk_aversion=lam,
                )

                solved[name] = weights

                metrics = portfolio_metrics(
                    weights,
                    data["target"],
                )

                result_rows.append({
                    "lambda": lam,
                    "seed": seed,
                    "combination": name,
                    **metrics,
                })

            reproduction_rows.append({
                "lambda": lam,
                "seed": seed,

                "dfl_max_abs_weight_diff":
                    float(
                        np.max(
                            np.abs(
                                solved["DFLpred_DFLrisk"]
                                - data["dfl_weights"]
                            )
                        )
                    ),

                "dfl_mean_abs_weight_diff":
                    float(
                        np.mean(
                            np.abs(
                                solved["DFLpred_DFLrisk"]
                                - data["dfl_weights"]
                            )
                        )
                    ),

                "rt_max_abs_weight_diff":
                    float(
                        np.max(
                            np.abs(
                                solved["RTpred_RTrisk"]
                                - data["rt_weights"]
                            )
                        )
                    ),

                "rt_mean_abs_weight_diff":
                    float(
                        np.mean(
                            np.abs(
                                solved["RTpred_RTrisk"]
                                - data["rt_weights"]
                            )
                        )
                    ),
            })

            print(
                f"SWAP DONE | "
                f"lambda={lam:g} | "
                f"seed={seed}"
            )

    by_seed = pd.DataFrame(result_rows)
    reproduction = pd.DataFrame(reproduction_rows)

    contrasts = [
        (
            "prediction_effect_under_DFL_risk",
            "RTpred_DFLrisk",
            "DFLpred_DFLrisk",
        ),
        (
            "prediction_effect_under_RT_risk",
            "RTpred_RTrisk",
            "DFLpred_RTrisk",
        ),
        (
            "risk_effect_under_DFL_prediction",
            "DFLpred_RTrisk",
            "DFLpred_DFLrisk",
        ),
        (
            "risk_effect_under_RT_prediction",
            "RTpred_RTrisk",
            "RTpred_DFLrisk",
        ),
        (
            "total_RT_minus_DFL",
            "RTpred_RTrisk",
            "DFLpred_DFLrisk",
        ),
    ]

    contrast_rows = []

    for lam in args.lambdas:
        current = by_seed[
            by_seed["lambda"] == lam
        ]

        for (
            contrast_name,
            model_a,
            model_b,
        ) in contrasts:

            for metric in [
                "annualized_return",
                "volatility",
                "sharpe",
                "max_drawdown",
            ]:
                a = (
                    current[
                        current["combination"]
                        == model_a
                    ]
                    .sort_values("seed")[metric]
                    .to_numpy()
                )

                b = (
                    current[
                        current["combination"]
                        == model_b
                    ]
                    .sort_values("seed")[metric]
                    .to_numpy()
                )

                diff = a - b

                lower_is_better = (
                    metric
                    in {
                        "volatility",
                        "max_drawdown",
                    }
                )

                better = (
                    diff < 0
                    if lower_is_better
                    else diff > 0
                )

                contrast_rows.append({
                    "lambda": lam,
                    "contrast": contrast_name,
                    "metric": metric,
                    "model_a_mean": np.mean(a),
                    "model_b_mean": np.mean(b),
                    "mean_difference_a_minus_b": np.mean(diff),
                    "median_difference_a_minus_b": np.median(diff),
                    "a_better_seeds": int(np.sum(better)),
                    "p_raw": safe_wilcoxon(a, b),
                })

    contrast_summary = pd.DataFrame(
        contrast_rows
    )

    contrast_summary["p_holm"] = np.nan

    for (
        lam,
        contrast,
    ), group in contrast_summary.groupby(
        ["lambda", "contrast"]
    ):
        idx = group.index

        contrast_summary.loc[
            idx,
            "p_holm",
        ] = holm_adjust(
            group["p_raw"].to_numpy()
        )

    return (
        by_seed,
        contrast_summary,
        reproduction,
    )


# =============================================================================
# Experiment 3
# Weight-shift realized-return attribution
# =============================================================================

def experiment_3(args, tickers):
    rows = []

    for lam in args.lambdas:
        for seed in args.seeds:
            data = aligned_run_data(
                seed,
                lam,
                tickers,
            )

            dfl_w = data["dfl_weights"]
            rt_w = data["rt_weights"]
            target = data["target"]

            delta_w = rt_w - dfl_w

            daily_advantage = np.sum(
                delta_w * target,
                axis=1,
            )

            overweight_contribution = np.sum(
                np.clip(
                    delta_w,
                    0.0,
                    None,
                )
                * target,
                axis=1,
            )

            underweight_contribution = np.sum(
                np.clip(
                    delta_w,
                    None,
                    0.0,
                )
                * target,
                axis=1,
            )

            dfl_port = np.sum(
                dfl_w * target,
                axis=1,
            )

            rt_port = np.sum(
                rt_w * target,
                axis=1,
            )

            identity_error = np.max(
                np.abs(
                    daily_advantage
                    - (
                        rt_port
                        - dfl_port
                    )
                )
            )

            daily_ic = []

            for t in range(len(target)):
                if (
                    np.std(delta_w[t]) < 1e-15
                    or np.std(target[t]) < 1e-15
                ):
                    continue

                corr = spearmanr(
                    delta_w[t],
                    target[t],
                ).statistic

                if np.isfinite(corr):
                    daily_ic.append(corr)

            rows.append({
                "lambda": lam,
                "seed": seed,

                "mean_daily_return_advantage":
                    np.mean(
                        daily_advantage
                    ),

                "annualized_arithmetic_return_advantage":
                    TRADING_DAYS
                    * np.mean(
                        daily_advantage
                    ),

                "median_daily_return_advantage":
                    np.median(
                        daily_advantage
                    ),

                "positive_advantage_days_pct":
                    100.0
                    * np.mean(
                        daily_advantage > 0
                    ),

                "annualized_overweight_contribution":
                    TRADING_DAYS
                    * np.mean(
                        overweight_contribution
                    ),

                "annualized_underweight_contribution":
                    TRADING_DAYS
                    * np.mean(
                        underweight_contribution
                    ),

                "mean_weight_shift_realized_return_spearman":
                    (
                        np.mean(daily_ic)
                        if daily_ic
                        else np.nan
                    ),

                "identity_max_abs_error":
                    identity_error,
            })

    by_seed = pd.DataFrame(rows)
    summary_rows = []

    for lam in args.lambdas:
        current = by_seed[
            by_seed["lambda"] == lam
        ]

        metric = (
            "annualized_arithmetic_return_advantage"
        )

        values = current[
            metric
        ].to_numpy()

        p_raw = (
            1.0
            if np.allclose(
                values,
                0.0,
            )
            else float(
                wilcoxon(
                    values,
                    alternative="two-sided",
                    zero_method="wilcox",
                ).pvalue
            )
        )

        summary_rows.append({
            "lambda": lam,

            "mean_annualized_return_advantage":
                current[metric].mean(),

            "median_annualized_return_advantage":
                current[metric].median(),

            "positive_return_advantage_seeds":
                int(
                    (
                        current[metric] > 0
                    ).sum()
                ),

            "mean_positive_advantage_days_pct":
                current[
                    "positive_advantage_days_pct"
                ].mean(),

            "mean_annualized_overweight_contribution":
                current[
                    "annualized_overweight_contribution"
                ].mean(),

            "mean_annualized_underweight_contribution":
                current[
                    "annualized_underweight_contribution"
                ].mean(),

            "mean_weight_shift_realized_return_spearman":
                current[
                    "mean_weight_shift_realized_return_spearman"
                ].mean(),

            "p_raw":
                p_raw,
        })

    summary = pd.DataFrame(
        summary_rows
    )

    return (
        by_seed,
        summary,
    )


def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 110
    )
    print(
        "RETURN-CHANNEL ANALYSIS"
    )
    print(
        "=" * 110
    )

    tickers, dfl_map, rt_map = (
        build_risk_maps(args)
    )

    # -------------------------------------------------------------------------
    # Experiment 1
    # -------------------------------------------------------------------------
    exp1_by_seed, exp1_summary = (
        experiment_1(
            args,
            tickers,
        )
    )

    exp1_by_seed.to_csv(
        output_dir
        / "01_return_risk_decomposition_by_seed.csv",
        index=False,
    )

    exp1_summary.to_csv(
        output_dir
        / "01_return_risk_decomposition_summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "EXPERIMENT 1 — RETURN / RISK DECOMPOSITION"
    )
    print(
        "=" * 110
    )

    print(
        exp1_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Experiment 2
    # -------------------------------------------------------------------------
    (
        exp2_by_seed,
        exp2_summary,
        reproduction,
    ) = experiment_2(
        args,
        tickers,
        dfl_map,
        rt_map,
    )

    exp2_by_seed.to_csv(
        output_dir
        / "02_prediction_risk_swap_by_seed.csv",
        index=False,
    )

    exp2_summary.to_csv(
        output_dir
        / "02_prediction_risk_swap_contrasts.csv",
        index=False,
    )

    reproduction.to_csv(
        output_dir
        / "02_reproduction_check.csv",
        index=False,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "EXPERIMENT 2 — REPRODUCTION CHECK"
    )
    print(
        "=" * 110
    )

    print(
        reproduction.describe().to_string(
            float_format=lambda x: (
                f"{x:.6e}"
            )
        )
    )

    print()
    print(
        "=" * 110
    )
    print(
        "EXPERIMENT 2 — PREDICTION × RISK MATRIX 2×2"
    )
    print(
        "=" * 110
    )

    important = exp2_summary[
        exp2_summary[
            "metric"
        ].isin(
            [
                "annualized_return",
                "sharpe",
            ]
        )
    ]

    print(
        important.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Experiment 3
    # -------------------------------------------------------------------------
    exp3_by_seed, exp3_summary = (
        experiment_3(
            args,
            tickers,
        )
    )

    exp3_by_seed.to_csv(
        output_dir
        / "03_weight_shift_return_attribution_by_seed.csv",
        index=False,
    )

    exp3_summary.to_csv(
        output_dir
        / "03_weight_shift_return_attribution_summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "EXPERIMENT 3 — WEIGHT-SHIFT RETURN ATTRIBUTION"
    )
    print(
        "=" * 110
    )

    print(
        exp3_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print("Saved:")

    for path in sorted(
        output_dir.glob("*.csv")
    ):
        print(path)


if __name__ == "__main__":
    main()