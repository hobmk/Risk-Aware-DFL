from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from implementations.common.portfolio_evaluation import evaluate_portfolio
from implementations.eigen_rcr_dfl.src.optimization import (
    build_markowitz_layer,
    covariance_to_risk_factor,
)
from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset


FINAL_ROOT = Path(
    "implementations/final_benchmark/outputs"
)

OUTPUT_DIR = (
    FINAL_ROOT / "placebo_rank_tail_shuffle"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank-Tail shuffled-risk structural placebo"
    )

    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.2)

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(42, 62)),
    )

    parser.add_argument(
        "--placebo-seeds",
        nargs="+",
        type=int,
        default=list(range(1000, 1010)),
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
    )

    return parser.parse_args()


def load_wide(
    path: Path,
    tickers: list[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    date_col = (
        "date"
        if "date" in df.columns
        else "Date"
    )

    if date_col not in df.columns:
        raise KeyError(
            f"No date column: {path}"
        )

    df[date_col] = (
        pd.to_datetime(
            df[date_col]
        ).dt.normalize()
    )

    result = (
        df
        .set_index(date_col)
        .sort_index()
        .reindex(columns=tickers)
    )

    if result.isna().any().any():
        raise RuntimeError(
            f"NaN after loading: {path}"
        )

    return result.astype(
        np.float64
    )


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


def solve_weights(
    layer,
    predicted_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
):
    mu = torch.tensor(
        predicted_returns,
        dtype=torch.float64,
    )

    covariance_tensor = torch.tensor(
        covariance,
        dtype=torch.float64,
    )

    risk_factor = covariance_to_risk_factor(
        covariance=covariance_tensor,
        risk_aversion=risk_aversion,
        jitter=0.0,
    )

    with torch.no_grad():
        weights, = layer(
            mu,
            risk_factor,
        )

    return (
        weights
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def evaluate(
    weights: np.ndarray,
    true_returns: np.ndarray,
):
    portfolio_returns = np.sum(
        weights * true_returns,
        axis=1,
    )

    return evaluate_portfolio(
        portfolio_returns=portfolio_returns,
        weights=weights,
        asset_returns=true_returns,
        cost_bps=(0.0, 5.0, 10.0),
        active_threshold=1e-3,
    )


def main():
    args = parse_args()

    if args.risk_aversion != 1.0:
        print(
            "WARNING: final placebo was designed "
            "for lambda=1."
        )

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
        "RANK-TAIL STRUCTURAL PLACEBO"
    )
    print(
        "=" * 110
    )

    print(
        f"lambda          : "
        f"{args.risk_aversion:g}"
    )

    print(
        f"tail q          : "
        f"{args.tail_q:.2f}"
    )

    print(
        f"model seeds     : "
        f"{len(args.seeds)}"
    )

    print(
        f"placebo shuffles: "
        f"{len(args.placebo_seeds)}"
    )

    # ---------------------------------------------------------
    # Build the exact final Rank-Tail risk matrices.
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
        tail_side="down",
    )

    tickers = [
        str(x)
        for x in dataset.tickers
    ]

    n_assets = len(
        tickers
    )

    matrix_map = {}

    for i, date in enumerate(
        dataset.target_dates
    ):
        date = pd.Timestamp(
            date
        ).normalize()

        matrix_map[date] = {
            "market": (
                dataset
                .market_covariances[i]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            ),

            "rank_tail": (
                dataset
                .rank_tail_risk[i]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            ),

            "effective": (
                dataset
                .covariances[i]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            ),
        }

    layer = build_markowitz_layer(
        n_assets=n_assets,
        max_weight=args.max_weight,
    )

    identity = np.eye(
        n_assets,
        dtype=np.float64,
    )

    original_rows = []
    placebo_rows = []
    reproduction_rows = []

    # Fixed permutation per placebo replication.
    # The same relabeling is used across all dates and seeds.
    permutations = {}

    for placebo_seed in (
        args.placebo_seeds
    ):
        rng = np.random.default_rng(
            placebo_seed
        )

        permutations[
            placebo_seed
        ] = rng.permutation(
            n_assets
        )

    # ---------------------------------------------------------
    # Model seed loop
    # ---------------------------------------------------------

    for seed in args.seeds:
        run_dir = (
            FINAL_ROOT
            / "rank_tail"
            / "lambda_1"
            / f"seed_{seed}"
        )

        predictions = load_wide(
            run_dir
            / "predicted_returns.csv",
            tickers,
        )

        true_returns = load_wide(
            run_dir
            / "true_returns.csv",
            tickers,
        )

        saved_weights = load_wide(
            run_dir
            / "portfolio_weights.csv",
            tickers,
        )

        dates = (
            predictions.index
            .intersection(
                true_returns.index
            )
            .intersection(
                saved_weights.index
            )
        )

        dates = pd.DatetimeIndex(
            [
                date
                for date in dates
                if date in matrix_map
            ]
        ).sort_values()

        if len(dates) == 0:
            raise RuntimeError(
                f"No aligned dates: seed={seed}"
            )

        pred = (
            predictions
            .loc[dates]
            .to_numpy()
        )

        target = (
            true_returns
            .loc[dates]
            .to_numpy()
        )

        saved_w = (
            saved_weights
            .loc[dates]
            .to_numpy()
        )

        market = np.stack(
            [
                matrix_map[
                    date
                ]["market"]
                for date in dates
            ],
            axis=0,
        )

        rank_tail = np.stack(
            [
                matrix_map[
                    date
                ]["rank_tail"]
                for date in dates
            ],
            axis=0,
        )

        original_covariance = np.stack(
            [
                matrix_map[
                    date
                ]["effective"]
                for date in dates
            ],
            axis=0,
        )

        # -----------------------------------------------------
        # Sanity check:
        # same predictions + same covariance should reproduce
        # the saved portfolio weights.
        # -----------------------------------------------------

        reproduced_w = solve_weights(
            layer=layer,
            predicted_returns=pred,
            covariance=original_covariance,
            risk_aversion=(
                args.risk_aversion
            ),
        )

        max_abs_diff = float(
            np.max(
                np.abs(
                    reproduced_w
                    - saved_w
                )
            )
        )

        mean_abs_diff = float(
            np.mean(
                np.abs(
                    reproduced_w
                    - saved_w
                )
            )
        )

        reproduction_rows.append({
            "seed": seed,
            "max_abs_weight_difference":
                max_abs_diff,
            "mean_abs_weight_difference":
                mean_abs_diff,
        })

        # Use actual saved final weights for original metrics.
        original_metrics = evaluate(
            weights=saved_w,
            true_returns=target,
        )

        original_rows.append({
            "seed": seed,
            **original_metrics,
        })

        # -----------------------------------------------------
        # Shuffled Rank-Tail placebo
        # -----------------------------------------------------

        for placebo_seed in (
            args.placebo_seeds
        ):
            permutation = (
                permutations[
                    placebo_seed
                ]
            )

            shuffled_rank_tail = (
                rank_tail[
                    :,
                    permutation,
                    :
                ][
                    :,
                    :,
                    permutation
                ]
            )

            placebo_covariance = (
                market
                + shuffled_rank_tail
                + 1e-6 * identity[
                    None,
                    :,
                    :
                ]
            )

            placebo_covariance = (
                0.5
                * (
                    placebo_covariance
                    + placebo_covariance.transpose(
                        0,
                        2,
                        1,
                    )
                )
            )

            placebo_w = solve_weights(
                layer=layer,
                predicted_returns=pred,
                covariance=(
                    placebo_covariance
                ),
                risk_aversion=(
                    args.risk_aversion
                ),
            )

            placebo_metrics = evaluate(
                weights=placebo_w,
                true_returns=target,
            )

            placebo_rows.append({
                "seed": seed,
                "placebo_seed":
                    placebo_seed,
                **placebo_metrics,
            })

        print(
            f"DONE | seed={seed} | "
            f"dates={len(dates)} | "
            f"reproduction max diff="
            f"{max_abs_diff:.3e}"
        )

    # ---------------------------------------------------------
    # Aggregate placebo replications inside each model seed.
    # ---------------------------------------------------------

    original = pd.DataFrame(
        original_rows
    )

    placebo = pd.DataFrame(
        placebo_rows
    )

    reproduction = pd.DataFrame(
        reproduction_rows
    )

    metrics = [
        "gross_sharpe",
        "net_5bp_sharpe",
        "net_10bp_sharpe",
    ]

    placebo_by_seed = (
        placebo
        .groupby(
            "seed",
            as_index=False,
        )[metrics]
        .mean()
    )

    comparison = (
        original[
            ["seed"] + metrics
        ]
        .merge(
            placebo_by_seed,
            on="seed",
            suffixes=(
                "_original",
                "_placebo",
            ),
        )
    )

    test_rows = []

    for metric in metrics:
        original_values = comparison[
            f"{metric}_original"
        ].to_numpy()

        placebo_values = comparison[
            f"{metric}_placebo"
        ].to_numpy()

        difference = (
            original_values
            - placebo_values
        )

        statistic, p_raw = wilcoxon(
            original_values,
            placebo_values,
            alternative="two-sided",
            zero_method="wilcox",
        )

        test_rows.append({
            "metric": metric,

            "original_mean":
                float(
                    np.mean(
                        original_values
                    )
                ),

            "placebo_mean":
                float(
                    np.mean(
                        placebo_values
                    )
                ),

            "mean_difference_original_minus_placebo":
                float(
                    np.mean(
                        difference
                    )
                ),

            "median_difference_original_minus_placebo":
                float(
                    np.median(
                        difference
                    )
                ),

            "original_wins":
                int(
                    np.sum(
                        difference > 0
                    )
                ),

            "placebo_wins":
                int(
                    np.sum(
                        difference < 0
                    )
                ),

            "wilcoxon_statistic":
                float(statistic),

            "p_raw":
                float(p_raw),
        })

    tests = pd.DataFrame(
        test_rows
    )

    tests["p_holm"] = holm_adjust(
        tests[
            "p_raw"
        ].to_numpy()
    )

    tests["significant_0.05"] = (
        tests["p_holm"] < 0.05
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    original.to_csv(
        output_dir
        / "original_metrics.csv",
        index=False,
    )

    placebo.to_csv(
        output_dir
        / "placebo_all_shuffles.csv",
        index=False,
    )

    placebo_by_seed.to_csv(
        output_dir
        / "placebo_mean_by_seed.csv",
        index=False,
    )

    comparison.to_csv(
        output_dir
        / "original_vs_placebo_by_seed.csv",
        index=False,
    )

    reproduction.to_csv(
        output_dir
        / "reproduction_check.csv",
        index=False,
    )

    tests.to_csv(
        output_dir
        / "placebo_tests.csv",
        index=False,
    )

    print()
    print(
        "=" * 110
    )

    print(
        "REPRODUCTION CHECK"
    )

    print(
        "=" * 110
    )

    print(
        reproduction.describe().to_string()
    )

    print()
    print(
        "=" * 110
    )

    print(
        "ORIGINAL vs SHUFFLED RANK-TAIL PLACEBO"
    )

    print(
        "=" * 110
    )

    print(
        tests.to_string(
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
        / "original_metrics.csv"
    )
    print(
        output_dir
        / "placebo_all_shuffles.csv"
    )
    print(
        output_dir
        / "placebo_mean_by_seed.csv"
    )
    print(
        output_dir
        / "original_vs_placebo_by_seed.csv"
    )
    print(
        output_dir
        / "reproduction_check.csv"
    )
    print(
        output_dir
        / "placebo_tests.csv"
    )


if __name__ == "__main__":
    main()