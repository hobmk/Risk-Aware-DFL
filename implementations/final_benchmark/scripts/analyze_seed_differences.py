from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "implementations/"
    "final_benchmark/"
    "outputs/"
    "common_evaluation/"
    "all_runs_common_metrics.csv"
)

OUTPUT_DIR = Path(
    "implementations/"
    "final_benchmark/"
    "outputs/"
    "seed_analysis"
)

BASELINE = "dfl_mvo"

MODELS = [
    "ridge",
    "ledoit_wolf",
    "rank_tail",
    "topk",
]

LAMBDAS = [
    1.0,
    25.0,
]

METRIC = "gross_sharpe"


def main():
    data = pd.read_csv(INPUT)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_rows = []
    summary_rows = []

    for lam in LAMBDAS:
        subset = data[
            data["lambda"] == lam
        ]

        pivot = subset.pivot(
            index="seed",
            columns="model",
            values=METRIC,
        ).sort_index()

        print(
            "\n"
            + "=" * 100
        )
        print(
            f"LAMBDA = {lam:g}"
        )
        print(
            "=" * 100
        )

        for model in MODELS:
            baseline = pivot[
                BASELINE
            ]

            candidate = pivot[
                model
            ]

            difference = (
                candidate
                - baseline
            )

            comparison = pd.DataFrame({
                "seed": pivot.index,
                "dfl_mvo": baseline.values,
                model: candidate.values,
                "difference": difference.values,
            })

            wins = int(
                np.sum(
                    difference > 0
                )
            )

            losses = int(
                np.sum(
                    difference < 0
                )
            )

            ties = int(
                np.sum(
                    difference == 0
                )
            )

            mean_diff = float(
                difference.mean()
            )

            median_diff = float(
                difference.median()
            )

            std_diff = float(
                difference.std(
                    ddof=1
                )
            )

            min_diff = float(
                difference.min()
            )

            max_diff = float(
                difference.max()
            )

            print(
                f"\n{model.upper()} - DFL_MVO"
            )
            print("-" * 70)

            print(
                comparison.to_string(
                    index=False,
                    float_format=lambda x: (
                        f"{x:+.6f}"
                    ),
                )
            )

            print(
                "\nSummary"
            )

            print(
                f"Mean diff   : "
                f"{mean_diff:+.6f}"
            )

            print(
                f"Median diff : "
                f"{median_diff:+.6f}"
            )

            print(
                f"Std diff    : "
                f"{std_diff:.6f}"
            )

            print(
                f"Min diff    : "
                f"{min_diff:+.6f}"
            )

            print(
                f"Max diff    : "
                f"{max_diff:+.6f}"
            )

            print(
                f"W / L / T   : "
                f"{wins} / "
                f"{losses} / "
                f"{ties}"
            )

            for seed, base, cand, diff in zip(
                pivot.index,
                baseline,
                candidate,
                difference,
            ):
                detail_rows.append({
                    "lambda": lam,
                    "model": model,
                    "seed": int(seed),
                    "dfl_mvo_sharpe": float(
                        base
                    ),
                    "model_sharpe": float(
                        cand
                    ),
                    "difference": float(
                        diff
                    ),
                })

            summary_rows.append({
                "lambda": lam,
                "model": model,
                "mean_difference": (
                    mean_diff
                ),
                "median_difference": (
                    median_diff
                ),
                "std_difference": (
                    std_diff
                ),
                "min_difference": (
                    min_diff
                ),
                "max_difference": (
                    max_diff
                ),
                "wins": wins,
                "losses": losses,
                "ties": ties,
            })

    detail = pd.DataFrame(
        detail_rows
    )

    summary = pd.DataFrame(
        summary_rows
    )

    detail.to_csv(
        OUTPUT_DIR
        / "seed_level_sharpe_differences.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "seed_difference_summary.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "OVERALL SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "seed_level_sharpe_differences.csv"
    )

    print(
        OUTPUT_DIR
        / "seed_difference_summary.csv"
    )


if __name__ == "__main__":
    main()