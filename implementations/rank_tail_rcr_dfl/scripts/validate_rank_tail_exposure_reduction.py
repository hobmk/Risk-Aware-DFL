from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from implementations.rank_tail_rcr_dfl.src.dataset import (
    RankTailRCRDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "DFL-MVO vs Rank-Tail DFL "
            "Rank-Tail exposure comparison"
        )
    )

    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--hac-lags", type=int, default=19)

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
    )

    parser.add_argument(
        "--baseline-root",
        default=(
            "implementations/"
            "dfl_mvo_lee2025/"
            "outputs/combined/dow30/"
            "final_full_grid_h64_d0_s30_std/"
            "alpha_0.50/"
            "lambda_1.00/"
            "maxw_0.20"
        ),
    )

    parser.add_argument(
        "--rank-tail-root",
        default=(
            "implementations/"
            "rank_tail_rcr_dfl/"
            "outputs/q_grid"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    return parser.parse_args()


def load_weights(
    path: Path,
    tickers: list[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Result not found: {path}"
        )

    df = pd.read_csv(path)

    # -----------------------------------------------------
    # Long format:
    # date | ticker | predicted_weight
    # -----------------------------------------------------
    if {
        "date",
        "ticker",
        "predicted_weight",
    }.issubset(df.columns):

        df["date"] = pd.to_datetime(
            df["date"]
        ).dt.normalize()

        df["ticker"] = (
            df["ticker"].astype(str)
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

    # -----------------------------------------------------
    # Wide format:
    # date | AAPL | AMGN | AMZN | ...
    # -----------------------------------------------------
    elif "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"]
        ).dt.normalize()

        weights = (
            df
            .set_index("date")
            .sort_index()
            .reindex(columns=tickers)
        )

    else:
        raise KeyError(
            f"Unsupported weight format: {path}"
        )

    if weights.isna().any().any():
        missing_columns = [
            ticker
            for ticker in tickers
            if ticker not in weights.columns
        ]

        raise RuntimeError(
            f"NaN after loading weights: {path}\n"
            f"Missing tickers: {missing_columns}"
        )

    return weights.astype(
        np.float64
    )

def build_matrix_map(
    dataset: RankTailRCRDataset,
):
    result = {}

    for index in range(
        len(dataset)
    ):
        item = dataset[index]

        date = pd.Timestamp(
            item["target_date"]
        ).normalize()

        matrix = (
            dataset.rank_tail_risk[index]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=False,
            )
        )

        result[date] = matrix

    return result


def quadratic_exposure(
    weights: np.ndarray,
    matrices: np.ndarray,
):
    return np.einsum(
        "bi,bij,bj->b",
        weights,
        matrices,
        weights,
        optimize=True,
    )


def split_matrices(
    matrices: np.ndarray,
):
    n_assets = matrices.shape[1]

    diagonal = np.zeros_like(
        matrices
    )

    diag_values = np.diagonal(
        matrices,
        axis1=1,
        axis2=2,
    )

    idx = np.arange(
        n_assets
    )

    diagonal[
        :,
        idx,
        idx,
    ] = diag_values

    offdiag = (
        matrices - diagonal
    )

    # Diagnostic exposure only.
    # This matrix is NOT used for optimization.
    positive_offdiag = np.maximum(
        offdiag,
        0.0,
    )

    return (
        diagonal,
        offdiag,
        positive_offdiag,
    )


def compute_exposures(
    weights: np.ndarray,
    matrices: np.ndarray,
):
    (
        diagonal,
        offdiag,
        positive_offdiag,
    ) = split_matrices(
        matrices
    )

    return {
        "rt_total_exposure":
            quadratic_exposure(
                weights,
                matrices,
            ),

        "residual_diag_exposure":
            quadratic_exposure(
                weights,
                diagonal,
            ),

        "rt_offdiag_exposure":
            quadratic_exposure(
                weights,
                offdiag,
            ),

        "rt_positive_offdiag_exposure":
            quadratic_exposure(
                weights,
                positive_offdiag,
            ),
    }


def hac_mean_test(
    values: np.ndarray,
    hac_lags: int,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    X = np.ones(
        (len(values), 1),
        dtype=np.float64,
    )

    model = sm.OLS(
        values,
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": hac_lags,
        },
    )

    return {
        "mean":
            float(model.params[0]),

        "hac_se":
            float(model.bse[0]),

        "hac_t":
            float(model.tvalues[0]),

        "hac_p":
            float(model.pvalues[0]),
    }


def high_exposure_interaction(
    delta: np.ndarray,
    high_q4: np.ndarray,
    hac_lags: int,
):
    work = pd.DataFrame(
        {
            "delta": delta,
            "high_q4": high_q4,
        }
    ).dropna()

    X = sm.add_constant(
        work["high_q4"]
    )

    model = sm.OLS(
        work["delta"],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": hac_lags,
        },
    )

    return {
        "q4_interaction_beta":
            float(
                model.params["high_q4"]
            ),

        "q4_interaction_se":
            float(
                model.bse["high_q4"]
            ),

        "q4_interaction_t":
            float(
                model.tvalues["high_q4"]
            ),

        "q4_interaction_p":
            float(
                model.pvalues["high_q4"]
            ),
    }


def relative_change(
    baseline: float,
    rank_tail: float,
):
    if abs(baseline) < 1e-15:
        return np.nan

    return (
        (
            rank_tail
            / baseline
            - 1.0
        )
        * 100.0
    )


def main():
    args = parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            Path(
                "implementations/"
                "rank_tail_rcr_dfl/"
                "outputs/mechanism"
            )
            / f"q_{args.tail_q:.2f}"
            / "step2_exposure_reduction"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 105
    )
    print(
        "RANK-TAIL STEP 2: "
        "DFL-MVO vs RANK-TAIL DFL EXPOSURE"
    )
    print(
        "=" * 105
    )

    print(
        f"Tail q   : "
        f"{args.tail_q:.2f}"
    )

    print(
        f"HAC lags : "
        f"{args.hac_lags}"
    )

    print(
        f"Seeds    : "
        f"{args.seeds}"
    )

    # ---------------------------------------------------------
    # Same Rank-Tail matrices as Step 1
    # ---------------------------------------------------------

    dataset = RankTailRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,

        return_type="simple",
        covariance_jitter=1e-6,

        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,

        tail_q=args.tail_q,
    )

    tickers = [
        str(ticker)
        for ticker in dataset.tickers
    ]

    matrix_map = build_matrix_map(
        dataset
    )

    baseline_root = Path(
        args.baseline_root
    )

    rank_tail_root = Path(
        args.rank_tail_root
    )

    exposure_metrics = [
        "rt_total_exposure",
        "residual_diag_exposure",
        "rt_offdiag_exposure",
        "rt_positive_offdiag_exposure",
    ]

    daily_frames = []
    summary_rows = []

    # ---------------------------------------------------------
    # Seed loop
    # ---------------------------------------------------------

    for seed in args.seeds:

        baseline_path = (
            baseline_root
            / f"seed_{seed}"
            / "asset_predictions.csv"
        )

        rank_tail_path = (
            rank_tail_root
            / f"q_{args.tail_q:.2f}"
            / f"seed_{seed}"
            / "portfolio_weights.csv"
        )

        baseline_weights = load_weights(
            baseline_path,
            tickers,
        )

        rank_tail_weights = load_weights(
            rank_tail_path,
            tickers,
        )

        common_dates = (
            baseline_weights.index
            .intersection(
                rank_tail_weights.index
            )
            .sort_values()
        )

        if len(common_dates) == 0:
            raise RuntimeError(
                f"No common dates for seed {seed}"
            )

        baseline_weights = (
            baseline_weights
            .loc[common_dates]
        )

        rank_tail_weights = (
            rank_tail_weights
            .loc[common_dates]
        )

        missing_dates = [
            date
            for date in common_dates
            if date not in matrix_map
        ]

        if missing_dates:
            raise KeyError(
                "Rank-Tail matrix missing: "
                f"{missing_dates[0]}"
            )

        matrices = np.stack(
            [
                matrix_map[date]
                for date in common_dates
            ],
            axis=0,
        )

        baseline_exposures = (
            compute_exposures(
                baseline_weights.to_numpy(
                    dtype=np.float64
                ),
                matrices,
            )
        )

        rank_tail_exposures = (
            compute_exposures(
                rank_tail_weights.to_numpy(
                    dtype=np.float64
                ),
                matrices,
            )
        )

        daily = pd.DataFrame(
            {
                "seed": seed,
                "date": common_dates,
            }
        )

        for metric in exposure_metrics:

            baseline_values = (
                baseline_exposures[
                    metric
                ]
            )

            rank_tail_values = (
                rank_tail_exposures[
                    metric
                ]
            )

            delta = (
                rank_tail_values
                - baseline_values
            )

            daily[
                f"dfl_{metric}"
            ] = baseline_values

            daily[
                f"rank_tail_{metric}"
            ] = rank_tail_values

            daily[
                f"delta_{metric}"
            ] = delta

            # ---------------------------------------------
            # Entire test period
            # ---------------------------------------------

            dfl_mean = float(
                baseline_values.mean()
            )

            rt_mean = float(
                rank_tail_values.mean()
            )

            hac = hac_mean_test(
                delta,
                args.hac_lags,
            )

            # ---------------------------------------------
            # High-exposure Q4 is defined ONLY
            # from the baseline DFL-MVO exposure.
            # ---------------------------------------------

            q4_threshold = float(
                np.quantile(
                    baseline_values,
                    0.75,
                )
            )

            high_q4 = (
                baseline_values
                >= q4_threshold
            )

            daily[
                f"high_q4_{metric}"
            ] = high_q4.astype(int)

            dfl_q4_mean = float(
                baseline_values[
                    high_q4
                ].mean()
            )

            rt_q4_mean = float(
                rank_tail_values[
                    high_q4
                ].mean()
            )

            interaction = (
                high_exposure_interaction(
                    delta=delta,
                    high_q4=(
                        high_q4.astype(float)
                    ),
                    hac_lags=(
                        args.hac_lags
                    ),
                )
            )

            summary_rows.append(
                {
                    "seed": seed,

                    "tail_q":
                        args.tail_q,

                    "exposure_metric":
                        metric,

                    "n_dates":
                        len(common_dates),

                    "dfl_mean":
                        dfl_mean,

                    "rank_tail_mean":
                        rt_mean,

                    "rank_tail_minus_dfl":
                        rt_mean - dfl_mean,

                    "relative_change_pct":
                        relative_change(
                            dfl_mean,
                            rt_mean,
                        ),

                    "hac_se":
                        hac["hac_se"],

                    "hac_t":
                        hac["hac_t"],

                    "hac_p":
                        hac["hac_p"],

                    "q4_threshold":
                        q4_threshold,

                    "q4_n":
                        int(
                            high_q4.sum()
                        ),

                    "dfl_q4_mean":
                        dfl_q4_mean,

                    "rank_tail_q4_mean":
                        rt_q4_mean,

                    "q4_rank_tail_minus_dfl":
                        (
                            rt_q4_mean
                            - dfl_q4_mean
                        ),

                    "q4_relative_change_pct":
                        relative_change(
                            dfl_q4_mean,
                            rt_q4_mean,
                        ),

                    **interaction,
                }
            )

        daily_frames.append(
            daily
        )

        print(
            f"seed {seed} | "
            f"common dates={len(common_dates)}"
        )

    # ---------------------------------------------------------
    # Save seed-level results
    # ---------------------------------------------------------

    daily_all = pd.concat(
        daily_frames,
        ignore_index=True,
    )

    by_seed = pd.DataFrame(
        summary_rows
    )

    by_seed[
        "significant_reduction"
    ] = (
        (
            by_seed[
                "rank_tail_minus_dfl"
            ] < 0
        )
        &
        (
            by_seed["hac_p"] < 0.05
        )
    ).astype(int)

    by_seed[
        "q4_stronger_reduction"
    ] = (
        by_seed[
            "q4_interaction_beta"
        ] < 0
    ).astype(int)

    by_seed[
        "q4_significant_stronger_reduction"
    ] = (
        (
            by_seed[
                "q4_interaction_beta"
            ] < 0
        )
        &
        (
            by_seed[
                "q4_interaction_p"
            ] < 0.05
        )
    ).astype(int)

    # ---------------------------------------------------------
    # Multi-seed summary
    # ---------------------------------------------------------

    summary = (
        by_seed
        .groupby(
            [
                "tail_q",
                "exposure_metric",
            ],
            as_index=False,
        )
        .agg(
            seeds=(
                "seed",
                "nunique",
            ),

            mean_dfl_exposure=(
                "dfl_mean",
                "mean",
            ),

            mean_rank_tail_exposure=(
                "rank_tail_mean",
                "mean",
            ),

            mean_relative_change_pct=(
                "relative_change_pct",
                "mean",
            ),

            std_relative_change_pct=(
                "relative_change_pct",
                "std",
            ),

            reduction_direction_seeds=(
                "relative_change_pct",
                lambda x:
                    int(
                        (x < 0).sum()
                    ),
            ),

            significant_reduction_seeds=(
                "significant_reduction",
                "sum",
            ),

            mean_q4_dfl_exposure=(
                "dfl_q4_mean",
                "mean",
            ),

            mean_q4_rank_tail_exposure=(
                "rank_tail_q4_mean",
                "mean",
            ),

            mean_q4_relative_change_pct=(
                "q4_relative_change_pct",
                "mean",
            ),

            std_q4_relative_change_pct=(
                "q4_relative_change_pct",
                "std",
            ),

            q4_stronger_reduction_seeds=(
                "q4_stronger_reduction",
                "sum",
            ),

            q4_significant_stronger_reduction_seeds=(
                "q4_significant_stronger_reduction",
                "sum",
            ),

            mean_q4_interaction_beta=(
                "q4_interaction_beta",
                "mean",
            ),
        )
    )

    daily_path = (
        output_dir
        / "rank_tail_exposure_reduction_daily.csv"
    )

    by_seed_path = (
        output_dir
        / "rank_tail_exposure_reduction_by_seed.csv"
    )

    summary_path = (
        output_dir
        / "rank_tail_exposure_reduction_summary.csv"
    )

    daily_all.to_csv(
        daily_path,
        index=False,
    )

    by_seed.to_csv(
        by_seed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Primary outputs
    # ---------------------------------------------------------

    primary = summary[
        summary[
            "exposure_metric"
        ].isin(
            [
                "rt_offdiag_exposure",
                "rt_positive_offdiag_exposure",
            ]
        )
    ].copy()

    print()
    print(
        "=" * 105
    )
    print(
        "PRIMARY RANK-TAIL EXPOSURE REDUCTION RESULTS"
    )
    print(
        "=" * 105
    )

    print(
        primary[
            [
                "exposure_metric",
                "mean_dfl_exposure",
                "mean_rank_tail_exposure",
                "mean_relative_change_pct",
                "std_relative_change_pct",
                "reduction_direction_seeds",
                "significant_reduction_seeds",
                "mean_q4_relative_change_pct",
                "q4_stronger_reduction_seeds",
                "q4_significant_stronger_reduction_seeds",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(daily_path)
    print(by_seed_path)
    print(summary_path)


if __name__ == "__main__":
    main()