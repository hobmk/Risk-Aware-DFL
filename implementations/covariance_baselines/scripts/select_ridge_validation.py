from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RIDGE_VALUES = {0.05, 0.10, 0.20, 0.30, 0.40}
SEEDS = {42, 43, 44, 45, 46}

ROOT = Path(
    "implementations/"
    "covariance_baselines/"
    "outputs/"
    "ridge_validation"
)

OUTPUT_DIR = Path(
    "implementations/"
    "covariance_baselines/"
    "outputs/"
    "validation_selection"
)


def main():
    rows = []

    for rho in sorted(RIDGE_VALUES):
        for seed in sorted(SEEDS):
            run_dir = (
                ROOT
                / f"rho_{rho:.2f}"
                / f"seed_{seed}"
            )

            summary_path = (
                run_dir
                / "validation_portfolio_summary.csv"
            )

            config_path = (
                run_dir
                / "config.json"
            )

            if not summary_path.exists():
                print(
                    f"[MISSING] "
                    f"rho={rho:.2f} | "
                    f"seed={seed}"
                )
                continue

            summary = pd.read_csv(
                summary_path
            ).iloc[0]

            row = {
                "rho": rho,
                "seed": seed,
                "run_dir": str(run_dir),
                "val_return": float(
                    summary[
                        "gross_annualized_return"
                    ]
                ),
                "val_vol": float(
                    summary[
                        "gross_volatility"
                    ]
                ),
                "val_sharpe": float(
                    summary[
                        "gross_sharpe"
                    ]
                ),
                "val_sortino": float(
                    summary[
                        "gross_sortino"
                    ]
                ),
                "val_mdd": float(
                    summary[
                        "gross_max_drawdown"
                    ]
                ),
                "turnover": float(
                    summary[
                        "annualized_turnover"
                    ]
                ),
                "hhi": float(
                    summary[
                        "hhi"
                    ]
                ),
                "effective_n": float(
                    summary[
                        "effective_n"
                    ]
                ),
            }

            if config_path.exists():
                with open(
                    config_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    config = json.load(file)

                row["lambda"] = float(
                    config["risk_aversion"]
                )
                row["alpha"] = float(
                    config["alpha"]
                )
                row["standardized"] = bool(
                    config[
                        "standardize_inputs"
                    ]
                )

            rows.append(
                row
            )

    runs = pd.DataFrame(
        rows
    )

    expected = (
        len(RIDGE_VALUES)
        * len(SEEDS)
    )

    print(
        f"Candidate runs found: "
        f"{len(runs)}"
    )

    print(
        f"Expected runs       : "
        f"{expected}"
    )

    if len(runs) != expected:
        raise RuntimeError(
            "Ridge validation run 수가 "
            "예상과 다릅니다."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = runs.sort_values(
        [
            "rho",
            "seed",
        ]
    ).reset_index(
        drop=True
    )

    runs.to_csv(
        OUTPUT_DIR
        / "ridge_validation_runs.csv",
        index=False,
    )

    summary = (
        runs.groupby(
            "rho",
            as_index=False,
        )
        .agg(
            n_seeds=(
                "seed",
                "nunique",
            ),
            val_return_mean=(
                "val_return",
                "mean",
            ),
            val_return_std=(
                "val_return",
                "std",
            ),
            val_vol_mean=(
                "val_vol",
                "mean",
            ),
            val_vol_std=(
                "val_vol",
                "std",
            ),
            val_sharpe_mean=(
                "val_sharpe",
                "mean",
            ),
            val_sharpe_std=(
                "val_sharpe",
                "std",
            ),
            val_sortino_mean=(
                "val_sortino",
                "mean",
            ),
            val_sortino_std=(
                "val_sortino",
                "std",
            ),
            val_mdd_mean=(
                "val_mdd",
                "mean",
            ),
            val_mdd_std=(
                "val_mdd",
                "std",
            ),
            turnover_mean=(
                "turnover",
                "mean",
            ),
            hhi_mean=(
                "hhi",
                "mean",
            ),
            effective_n_mean=(
                "effective_n",
                "mean",
            ),
        )
    )

    summary = summary.sort_values(
        [
            "val_sharpe_mean",
            "turnover_mean",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        OUTPUT_DIR
        / "ridge_validation_summary.csv",
        index=False,
    )

    best = summary.iloc[0]

    selected = {
        "model": "ridge_covariance_dfl",
        "selection_rule": (
            "highest mean validation gross "
            "Sharpe across seeds 42-46"
        ),
        "validation_only": True,
        "selection_lambda": 1.0,
        "selection_alpha": 0.5,
        "candidate_rho_values": (
            sorted(
                RIDGE_VALUES
            )
        ),
        "selection_seeds": (
            sorted(
                SEEDS
            )
        ),
        "reuse_for_final_lambdas": [
            1.0,
            25.0,
        ],
        "selected": {
            "rho": float(
                best["rho"]
            ),
            "validation_sharpe_mean": float(
                best[
                    "val_sharpe_mean"
                ]
            ),
            "validation_sharpe_std": float(
                best[
                    "val_sharpe_std"
                ]
            ),
            "validation_return_mean": float(
                best[
                    "val_return_mean"
                ]
            ),
            "validation_volatility_mean": float(
                best[
                    "val_vol_mean"
                ]
            ),
            "validation_sortino_mean": float(
                best[
                    "val_sortino_mean"
                ]
            ),
            "validation_mdd_mean": float(
                best[
                    "val_mdd_mean"
                ]
            ),
            "validation_turnover_mean": float(
                best[
                    "turnover_mean"
                ]
            ),
            "validation_hhi_mean": float(
                best[
                    "hhi_mean"
                ]
            ),
            "validation_effective_n_mean": float(
                best[
                    "effective_n_mean"
                ]
            ),
        },
    }

    with open(
        OUTPUT_DIR
        / "selected_ridge_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RIDGE VALIDATION SELECTION"
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
        "\nSelected:"
    )

    print(
        json.dumps(
            selected,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()