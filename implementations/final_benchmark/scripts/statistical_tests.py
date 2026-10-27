from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


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
    "statistics"
)

MODELS = [
    "dfl_mvo",
    "ridge",
    "ledoit_wolf",
    "rank_tail",
    "topk",
]

LAMBDAS = [
    1.0,
    25.0,
]

METRICS = [
    "gross_sharpe",
    "net_5bp_sharpe",
    "net_10bp_sharpe",
]


def holm_adjust(p_values):
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

    running_max = 0.0
    m = len(
        p_values
    )

    for rank, index in enumerate(
        order
    ):
        adjusted_p = (
            (m - rank)
            * p_values[index]
        )

        running_max = max(
            running_max,
            adjusted_p,
        )

        adjusted[index] = min(
            running_max,
            1.0,
        )

    return adjusted


def main():
    data = pd.read_csv(
        INPUT
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    friedman_rows = []
    pairwise_rows = []

    for lam in LAMBDAS:
        for metric in METRICS:
            subset = data[
                data["lambda"]
                == lam
            ]

            pivot = subset.pivot(
                index="seed",
                columns="model",
                values=metric,
            )

            pivot = pivot[
                MODELS
            ]

            if pivot.isna().any().any():
                raise RuntimeError(
                    f"Missing data: "
                    f"lambda={lam}, "
                    f"metric={metric}"
                )

            statistic, p_value = (
                friedmanchisquare(
                    *[
                        pivot[
                            model
                        ].to_numpy()
                        for model
                        in MODELS
                    ]
                )
            )

            friedman_rows.append({
                "lambda": lam,
                "metric": metric,
                "statistic": statistic,
                "p_value": p_value,
            })

            pairs = []

            for i in range(
                len(MODELS)
            ):
                for j in range(
                    i + 1,
                    len(MODELS),
                ):
                    model_a = MODELS[i]
                    model_b = MODELS[j]

                    a = pivot[
                        model_a
                    ].to_numpy()

                    b = pivot[
                        model_b
                    ].to_numpy()

                    diff = (
                        a - b
                    )

                    statistic_w, p_raw = (
                        wilcoxon(
                            a,
                            b,
                            alternative="two-sided",
                            zero_method="wilcox",
                        )
                    )

                    pairs.append({
                        "lambda": lam,
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_a": float(
                            np.mean(a)
                        ),
                        "mean_b": float(
                            np.mean(b)
                        ),
                        "mean_difference": float(
                            np.mean(diff)
                        ),
                        "wins_a": int(
                            np.sum(
                                diff > 0
                            )
                        ),
                        "wins_b": int(
                            np.sum(
                                diff < 0
                            )
                        ),
                        "ties": int(
                            np.sum(
                                diff == 0
                            )
                        ),
                        "wilcoxon_statistic": (
                            statistic_w
                        ),
                        "p_raw": (
                            p_raw
                        ),
                    })

            adjusted = holm_adjust(
                [
                    row["p_raw"]
                    for row
                    in pairs
                ]
            )

            for row, p_holm in zip(
                pairs,
                adjusted,
            ):
                row[
                    "p_holm"
                ] = float(
                    p_holm
                )

                row[
                    "significant_0.05"
                ] = bool(
                    p_holm < 0.05
                )

                pairwise_rows.append(
                    row
                )

    friedman_df = pd.DataFrame(
        friedman_rows
    )

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    friedman_df.to_csv(
        OUTPUT_DIR
        / "friedman_tests.csv",
        index=False,
    )

    pairwise_df.to_csv(
        OUTPUT_DIR
        / "wilcoxon_holm_tests.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FRIEDMAN TESTS"
    )

    print(
        "=" * 100
    )

    print(
        friedman_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PAIRWISE WILCOXON + HOLM"
    )

    print(
        "=" * 100
    )

    important = pairwise_df[
        (
            pairwise_df[
                "model_a"
            ]
            == "dfl_mvo"
        )
        |
        (
            pairwise_df[
                "model_b"
            ]
            == "dfl_mvo"
        )
    ]

    print(
        important[
            [
                "lambda",
                "metric",
                "model_a",
                "model_b",
                "mean_difference",
                "wins_a",
                "wins_b",
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

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "friedman_tests.csv"
    )

    print(
        OUTPUT_DIR
        / "wilcoxon_holm_tests.csv"
    )


if __name__ == "__main__":
    main()