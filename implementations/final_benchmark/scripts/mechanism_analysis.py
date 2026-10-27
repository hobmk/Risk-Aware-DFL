from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from implementations.eigen_rcr_dfl.src.dataset import EigenRCRDataset
from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset


FINAL_ROOT = Path(
    "implementations/final_benchmark/outputs"
)

DEFAULT_OUTPUT = (
    FINAL_ROOT / "mechanism"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final benchmark residual-risk mechanism analysis"
    )

    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--eigen-k", type=int, default=1)

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[1.0, 25.0],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(42, 62)),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
    )

    return parser.parse_args()


def lambda_dir(value: float) -> str:
    return f"lambda_{value:g}"


def load_weights(
    model: str,
    lam: float,
    seed: int,
    tickers: list[str],
) -> pd.DataFrame:

    run_dir = (
        FINAL_ROOT
        / model
        / lambda_dir(lam)
        / f"seed_{seed}"
    )

    if model == "dfl_mvo":
        path = (
            run_dir
            / "asset_predictions.csv"
        )

        df = pd.read_csv(path)

        required = {
            "date",
            "ticker",
            "predicted_weight",
        }

        missing = (
            required
            .difference(df.columns)
        )

        if missing:
            raise KeyError(
                f"{path} missing: "
                f"{sorted(missing)}"
            )

        df["date"] = (
            pd.to_datetime(
                df["date"]
            ).dt.normalize()
        )

        df["ticker"] = (
            df["ticker"]
            .astype(str)
        )

        weights = (
            df.pivot(
                index="date",
                columns="ticker",
                values="predicted_weight",
            )
            .sort_index()
            .reindex(columns=tickers)
        )

    else:
        path = (
            run_dir
            / "portfolio_weights.csv"
        )

        df = pd.read_csv(path)

        date_column = (
            "date"
            if "date" in df.columns
            else "Date"
        )

        if date_column not in df.columns:
            raise KeyError(
                f"No date column: {path}"
            )

        df[date_column] = (
            pd.to_datetime(
                df[date_column]
            ).dt.normalize()
        )

        weights = (
            df
            .set_index(date_column)
            .sort_index()
            .reindex(columns=tickers)
        )

    if weights.isna().any().any():
        raise RuntimeError(
            f"NaN after loading: {path}"
        )

    return weights.astype(
        np.float64
    )


def quadratic(
    weights: np.ndarray,
    matrices: np.ndarray,
) -> np.ndarray:

    return np.einsum(
        "bi,bij,bj->b",
        weights,
        matrices,
        weights,
        optimize=True,
    )


def hhi(
    weights: np.ndarray,
) -> np.ndarray:

    return np.sum(
        weights ** 2,
        axis=1,
    )


def build_risk_maps(
    args,
):
    print(
        "Building Rank-Tail risk matrices..."
    )

    rank_tail = RankTailRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        tail_q=args.tail_q,
        tail_side="down",
    )

    print(
        "Building Top-K risk matrices..."
    )

    topk = EigenRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        eigen_k=args.eigen_k,
        scale_mode="none",
    )

    if (
        list(rank_tail.tickers)
        != list(topk.tickers)
    ):
        raise RuntimeError(
            "Ticker mismatch between datasets"
        )

    tickers = [
        str(x)
        for x in rank_tail.tickers
    ]

    risk_map = {}

    for i, date in enumerate(
        rank_tail.target_dates
    ):
        date = pd.Timestamp(
            date
        ).normalize()

        risk_map[date] = {
            "rank_tail": (
                rank_tail
                .rank_tail_risk[i]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            ),

            "residual": (
                rank_tail
                .residual_covariances[i]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            ),
        }

    topk_map = {}

    for i, date in enumerate(
        topk.target_dates
    ):
        date = pd.Timestamp(
            date
        ).normalize()

        topk_map[date] = (
            topk
            .eigen_risk[i]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    common_dates = (
        set(risk_map)
        .intersection(topk_map)
    )

    residual_diff = []

    for i, date in enumerate(
        topk.target_dates
    ):
        date = pd.Timestamp(
            date
        ).normalize()

        if date not in common_dates:
            continue

        rt_residual = (
            risk_map[date]["residual"]
        )

        eigen_residual = (
            topk
            .residual_covariances[i]
            .detach()
            .cpu()
            .numpy()
        )

        residual_diff.append(
            np.max(
                np.abs(
                    rt_residual
                    - eigen_residual
                )
            )
        )

    print(
        "Max residual covariance difference:",
        f"{max(residual_diff):.3e}",
    )

    return (
        tickers,
        risk_map,
        topk_map,
    )


def aligned_matrices(
    dates,
    risk_map,
    topk_map,
):
    rt = np.stack(
        [
            risk_map[date]["rank_tail"]
            for date in dates
        ],
        axis=0,
    )

    residual = np.stack(
        [
            risk_map[date]["residual"]
            for date in dates
        ],
        axis=0,
    )

    topk = np.stack(
        [
            topk_map[date]
            for date in dates
        ],
        axis=0,
    )

    return (
        rt,
        residual,
        topk,
    )


def rank_tail_metrics(
    weights,
    rt_matrix,
    residual_matrix,
):
    concentration = hhi(weights)

    total = quadratic(
        weights,
        rt_matrix,
    )

    residual = quadratic(
        weights,
        residual_matrix,
    )

    diagonal = np.zeros_like(
        rt_matrix
    )

    idx = np.arange(
        rt_matrix.shape[1]
    )

    diagonal[
        :,
        idx,
        idx,
    ] = np.diagonal(
        rt_matrix,
        axis1=1,
        axis2=2,
    )

    offdiag = quadratic(
        weights,
        rt_matrix - diagonal,
    )

    positive_offdiag_matrix = (
        np.maximum(
            rt_matrix - diagonal,
            0.0,
        )
    )

    positive_offdiag = quadratic(
        weights,
        positive_offdiag_matrix,
    )

    return {
        "rt_total":
            total,

        "rt_hhi_normalized":
            total
            / np.maximum(
                concentration,
                1e-15,
            ),

        "rt_over_residual":
            total
            / np.maximum(
                residual,
                1e-15,
            ),

        "rt_offdiag":
            offdiag,

        "rt_positive_offdiag":
            positive_offdiag,

        "hhi":
            concentration,
    }


def topk_metrics(
    weights,
    topk_matrix,
    residual_matrix,
):
    concentration = hhi(weights)

    exposure = quadratic(
        weights,
        topk_matrix,
    )

    residual = quadratic(
        weights,
        residual_matrix,
    )

    return {
        "topk_total":
            exposure,

        "topk_hhi_normalized":
            exposure
            / np.maximum(
                concentration,
                1e-15,
            ),

        "topk_share":
            exposure
            / np.maximum(
                residual,
                1e-15,
            ),

        "hhi":
            concentration,
    }


def holm_adjust(
    p_values,
):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty_like(
        p_values
    )

    running = 0.0
    m = len(p_values)

    for rank, idx in enumerate(
        order
    ):
        value = (
            (m - rank)
            * p_values[idx]
        )

        running = max(
            running,
            value,
        )

        adjusted[idx] = min(
            running,
            1.0,
        )

    return adjusted


def analyze_pair(
    daily,
    lam,
    mechanism,
    baseline_model,
    candidate_model,
    primary_metrics,
):
    subset = daily[
        (
            daily["lambda"] == lam
        )
        & (
            daily["mechanism"]
            == mechanism
        )
    ]

    seed_rows = []

    for seed in sorted(
        subset["seed"].unique()
    ):
        seed_data = subset[
            subset["seed"] == seed
        ]

        baseline = seed_data[
            seed_data["model"]
            == baseline_model
        ]

        candidate = seed_data[
            seed_data["model"]
            == candidate_model
        ]

        row = {
            "lambda": lam,
            "mechanism": mechanism,
            "seed": seed,
        }

        for metric in primary_metrics:
            b = float(
                baseline[metric].mean()
            )

            c = float(
                candidate[metric].mean()
            )

            row[
                f"{metric}_dfl"
            ] = b

            row[
                f"{metric}_candidate"
            ] = c

            row[
                f"{metric}_difference"
            ] = c - b

            row[
                f"{metric}_reduction_pct"
            ] = (
                (1.0 - c / b)
                * 100.0
                if abs(b) > 1e-15
                else np.nan
            )

        seed_rows.append(row)

    seed_df = pd.DataFrame(
        seed_rows
    )

    summary_rows = []

    for metric in primary_metrics:
        b = seed_df[
            f"{metric}_dfl"
        ].to_numpy()

        c = seed_df[
            f"{metric}_candidate"
        ].to_numpy()

        diff = c - b

        statistic, p_raw = wilcoxon(
            c,
            b,
            alternative="two-sided",
            zero_method="wilcox",
        )

        reduction_pct = seed_df[
            f"{metric}_reduction_pct"
        ].to_numpy()

        summary_rows.append({
            "lambda": lam,
            "mechanism": mechanism,
            "metric": metric,

            "dfl_mean":
                float(np.mean(b)),

            "candidate_mean":
                float(np.mean(c)),

            "mean_difference":
                float(np.mean(diff)),

            "median_difference":
                float(np.median(diff)),

            "mean_reduction_pct":
                float(
                    np.nanmean(
                        reduction_pct
                    )
                ),

            "median_reduction_pct":
                float(
                    np.nanmedian(
                        reduction_pct
                    )
                ),

            "reduction_seeds":
                int(
                    np.sum(diff < 0)
                ),

            "increase_seeds":
                int(
                    np.sum(diff > 0)
                ),

            "wilcoxon_statistic":
                float(statistic),

            "p_raw":
                float(p_raw),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary["p_holm"] = holm_adjust(
        summary["p_raw"]
        .to_numpy()
    )

    summary["significant_0.05"] = (
        summary["p_holm"]
        < 0.05
    )

    return (
        seed_df,
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

    (
        tickers,
        risk_map,
        topk_map,
    ) = build_risk_maps(
        args
    )

    daily_rows = []

    for lam in args.lambdas:
        for seed in args.seeds:

            dfl = load_weights(
                "dfl_mvo",
                lam,
                seed,
                tickers,
            )

            rank_tail = load_weights(
                "rank_tail",
                lam,
                seed,
                tickers,
            )

            topk = load_weights(
                "topk",
                lam,
                seed,
                tickers,
            )

            common_dates = (
                dfl.index
                .intersection(
                    rank_tail.index
                )
                .intersection(
                    topk.index
                )
            )

            common_dates = (
                pd.DatetimeIndex(
                    [
                        date
                        for date in common_dates
                        if (
                            date in risk_map
                            and date in topk_map
                        )
                    ]
                )
                .sort_values()
            )

            if len(common_dates) == 0:
                raise RuntimeError(
                    f"No common dates: "
                    f"lambda={lam}, seed={seed}"
                )

            dfl_w = dfl.loc[
                common_dates
            ].to_numpy()

            rt_w = rank_tail.loc[
                common_dates
            ].to_numpy()

            topk_w = topk.loc[
                common_dates
            ].to_numpy()

            (
                rt_matrix,
                residual_matrix,
                topk_matrix,
            ) = aligned_matrices(
                common_dates,
                risk_map,
                topk_map,
            )

            # ---------------------------------------------
            # Rank-Tail mechanism
            # ---------------------------------------------

            for model, weights in [
                ("dfl_mvo", dfl_w),
                ("rank_tail", rt_w),
            ]:
                metrics = rank_tail_metrics(
                    weights,
                    rt_matrix,
                    residual_matrix,
                )

                for i, date in enumerate(
                    common_dates
                ):
                    daily_rows.append({
                        "lambda": lam,
                        "seed": seed,
                        "date": date,
                        "mechanism": "rank_tail",
                        "model": model,
                        **{
                            key: values[i]
                            for key, values
                            in metrics.items()
                        },
                    })

            # ---------------------------------------------
            # Top-K mechanism
            # ---------------------------------------------

            for model, weights in [
                ("dfl_mvo", dfl_w),
                ("topk", topk_w),
            ]:
                metrics = topk_metrics(
                    weights,
                    topk_matrix,
                    residual_matrix,
                )

                for i, date in enumerate(
                    common_dates
                ):
                    daily_rows.append({
                        "lambda": lam,
                        "seed": seed,
                        "date": date,
                        "mechanism": "topk",
                        "model": model,
                        **{
                            key: values[i]
                            for key, values
                            in metrics.items()
                        },
                    })

            print(
                "DONE | "
                f"lambda={lam:g} | "
                f"seed={seed} | "
                f"dates={len(common_dates)}"
            )

    daily = pd.DataFrame(
        daily_rows
    )

    seed_frames = []
    summary_frames = []

    for lam in args.lambdas:

        seed_df, summary = analyze_pair(
            daily=daily,
            lam=lam,
            mechanism="rank_tail",
            baseline_model="dfl_mvo",
            candidate_model="rank_tail",
            primary_metrics=[
                "rt_total",
                "rt_hhi_normalized",
                "rt_over_residual",
            ],
        )

        seed_frames.append(
            seed_df
        )

        summary_frames.append(
            summary
        )

        seed_df, summary = analyze_pair(
            daily=daily,
            lam=lam,
            mechanism="topk",
            baseline_model="dfl_mvo",
            candidate_model="topk",
            primary_metrics=[
                "topk_total",
                "topk_hhi_normalized",
                "topk_share",
            ],
        )

        seed_frames.append(
            seed_df
        )

        summary_frames.append(
            summary
        )

    seed_summary = pd.concat(
        seed_frames,
        ignore_index=True,
    )

    summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    daily.to_csv(
        output_dir
        / "mechanism_daily.csv",
        index=False,
    )

    seed_summary.to_csv(
        output_dir
        / "mechanism_by_seed.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "mechanism_summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 120
    )
    print(
        "MECHANISM ANALYSIS SUMMARY"
    )
    print(
        "=" * 120
    )

    print(
        summary[
            [
                "lambda",
                "mechanism",
                "metric",
                "dfl_mean",
                "candidate_mean",
                "mean_difference",
                "mean_reduction_pct",
                "reduction_seeds",
                "increase_seeds",
                "p_raw",
                "p_holm",
                "significant_0.05",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print("Saved:")
    print(
        output_dir
        / "mechanism_daily.csv"
    )
    print(
        output_dir
        / "mechanism_by_seed.csv"
    )
    print(
        output_dir
        / "mechanism_summary.csv"
    )


if __name__ == "__main__":
    main()