from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from implementations.common.portfolio_evaluation import (
    equal_weight_portfolio,
    evaluate_portfolio,
)


MODELS = [
    "dfl_mvo",
    "ridge",
    "ledoit_wolf",
    "rank_tail",
    "topk",
]

LAMBDAS = [1.0, 25.0]
SEEDS = list(range(42, 62))

ROOT = Path(
    "implementations/"
    "final_benchmark/"
    "outputs"
)

OUTPUT_DIR = ROOT / "common_evaluation"


def load_dfl_mvo(run_dir: Path):
    path = run_dir / "asset_predictions.csv"
    df = pd.read_csv(path)

    required = {
        "date",
        "ticker",
        "true_return",
        "predicted_weight",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}"
        )

    weights = df.pivot(
        index="date",
        columns="ticker",
        values="predicted_weight",
    )

    returns = df.pivot(
        index="date",
        columns="ticker",
        values="true_return",
    )

    common_columns = sorted(
        set(weights.columns)
        & set(returns.columns)
    )

    weights = weights[
        common_columns
    ].sort_index()

    returns = returns[
        common_columns
    ].sort_index()

    if not weights.index.equals(
        returns.index
    ):
        raise RuntimeError(
            f"Date mismatch: {run_dir}"
        )

    return (
        weights.index.to_numpy(),
        weights.to_numpy(
            dtype=np.float64
        ),
        returns.to_numpy(
            dtype=np.float64
        ),
        common_columns,
    )


def load_standard_run(run_dir: Path):
    weights_path = (
        run_dir
        / "portfolio_weights.csv"
    )

    returns_path = (
        run_dir
        / "true_returns.csv"
    )

    weights_df = pd.read_csv(
        weights_path
    )

    returns_df = pd.read_csv(
        returns_path
    )

    if "date" not in weights_df:
        raise ValueError(
            f"date column 없음: {weights_path}"
        )

    if "date" not in returns_df:
        raise ValueError(
            f"date column 없음: {returns_path}"
        )

    weights_df = weights_df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    returns_df = returns_df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    weight_columns = [
        column
        for column in weights_df.columns
        if column != "date"
    ]

    return_columns = [
        column
        for column in returns_df.columns
        if column != "date"
    ]

    if weight_columns != return_columns:
        raise RuntimeError(
            f"Ticker mismatch: {run_dir}"
        )

    if not np.array_equal(
        weights_df["date"].astype(str).to_numpy(),
        returns_df["date"].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            f"Date mismatch: {run_dir}"
        )

    return (
        weights_df["date"]
        .astype(str)
        .to_numpy(),
        weights_df[
            weight_columns
        ].to_numpy(
            dtype=np.float64
        ),
        returns_df[
            return_columns
        ].to_numpy(
            dtype=np.float64
        ),
        weight_columns,
    )


def load_run(
    model: str,
    run_dir: Path,
):
    if model == "dfl_mvo":
        return load_dfl_mvo(
            run_dir
        )

    return load_standard_run(
        run_dir
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    canonical_dates = None
    canonical_returns = None
    canonical_tickers = None

    for model in MODELS:
        for risk_aversion in LAMBDAS:
            for seed in SEEDS:
                run_dir = (
                    ROOT
                    / model
                    / f"lambda_{risk_aversion:g}"
                    / f"seed_{seed}"
                )

                (
                    dates,
                    weights,
                    asset_returns,
                    tickers,
                ) = load_run(
                    model,
                    run_dir,
                )

                if canonical_dates is None:
                    canonical_dates = (
                        dates.copy()
                    )

                    canonical_returns = (
                        asset_returns.copy()
                    )

                    canonical_tickers = list(
                        tickers
                    )

                else:
                    if list(
                        tickers
                    ) != canonical_tickers:
                        raise RuntimeError(
                            "Ticker mismatch | "
                            f"{model} | "
                            f"lambda={risk_aversion:g} | "
                            f"seed={seed}"
                        )

                    if not np.array_equal(
                        dates,
                        canonical_dates,
                    ):
                        raise RuntimeError(
                            "Date mismatch | "
                            f"{model} | "
                            f"lambda={risk_aversion:g} | "
                            f"seed={seed}"
                        )

                    max_return_diff = (
                        np.max(
                            np.abs(
                                asset_returns
                                - canonical_returns
                            )
                        )
                    )

                    if max_return_diff > 1e-6:
                        raise RuntimeError(
                            "True return mismatch | "
                            f"{model} | "
                            f"lambda={risk_aversion:g} | "
                            f"seed={seed} | "
                            f"max diff="
                            f"{max_return_diff:.3e}"
                        )

                portfolio_returns = (
                    np.sum(
                        weights
                        * asset_returns,
                        axis=1,
                    )
                )

                metrics = evaluate_portfolio(
                    portfolio_returns=(
                        portfolio_returns
                    ),
                    weights=weights,
                    asset_returns=(
                        asset_returns
                    ),
                    cost_bps=(
                        0.0,
                        5.0,
                        10.0,
                    ),
                    active_threshold=1e-3,
                )

                row = {
                    "model": model,
                    "lambda": risk_aversion,
                    "seed": seed,
                    **metrics,
                }

                rows.append(
                    row
                )

                print(
                    f"DONE | "
                    f"{model:12s} | "
                    f"lambda="
                    f"{risk_aversion:g} | "
                    f"seed={seed} | "
                    f"Sharpe="
                    f"{metrics['gross_sharpe']:.4f}"
                )

    runs = pd.DataFrame(
        rows
    )

    runs.to_csv(
        OUTPUT_DIR
        / "all_runs_common_metrics.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 10-seed mean / std
    # ---------------------------------------------------------

    metadata = {
        "model",
        "lambda",
        "seed",
    }

    numeric_metrics = [
        column
        for column in runs.columns
        if column not in metadata
        and pd.api.types.is_numeric_dtype(
            runs[column]
        )
    ]

    grouped = (
        runs.groupby(
            [
                "model",
                "lambda",
            ]
        )[numeric_metrics]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    grouped.columns = [
        f"{metric}_{stat}"
        for metric, stat
        in grouped.columns
    ]

    grouped = (
        grouped.reset_index()
    )

    grouped.to_csv(
        OUTPUT_DIR
        / "model_summary_mean_std.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Equal Weight
    # ---------------------------------------------------------

    n_assets = canonical_returns.shape[1]

    ew_weights = np.full(
        canonical_returns.shape,
        1.0 / n_assets,
        dtype=np.float64,
    )

    ew_returns = np.sum(
        ew_weights * canonical_returns,
        axis=1,
    )

    ew_metrics = evaluate_portfolio(
        portfolio_returns=ew_returns,
        weights=ew_weights,
        asset_returns=canonical_returns,
        cost_bps=(0.0, 5.0, 10.0),
        active_threshold=1e-3,
    )

    pd.DataFrame([
        ew_metrics
    ]).to_csv(
        OUTPUT_DIR
        / "equal_weight_metrics.csv",
        index=False,
    )

    ew_weights_df = pd.DataFrame(
        ew_weights,
        columns=canonical_tickers,
    )

    ew_weights_df.insert(
        0,
        "date",
        canonical_dates,
    )

    ew_weights_df.to_csv(
        OUTPUT_DIR
        / "equal_weight_weights.csv",
        index=False,
    )

    pd.DataFrame({
        "date": canonical_dates,
        "portfolio_return": (
            ew_returns
        ),
    }).to_csv(
        OUTPUT_DIR
        / "equal_weight_daily.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Compact final tables
    # ---------------------------------------------------------

    performance_columns = [
        "model",
        "lambda",

        "gross_annualized_return_mean",
        "gross_annualized_return_std",

        "gross_volatility_mean",
        "gross_volatility_std",

        "gross_sharpe_mean",
        "gross_sharpe_std",

        "gross_sortino_mean",
        "gross_sortino_std",

        "gross_max_drawdown_mean",
        "gross_max_drawdown_std",

        "gross_calmar_mean",
        "gross_calmar_std",

        "gross_cvar_95_daily_mean",
        "gross_cvar_95_daily_std",
    ]

    performance_columns = [
        column
        for column in performance_columns
        if column in grouped.columns
    ]

    grouped[
        performance_columns
    ].to_csv(
        OUTPUT_DIR
        / "performance_table.csv",
        index=False,
    )

    cost_columns = [
        "model",
        "lambda",

        "gross_annualized_return_mean",
        "gross_sharpe_mean",

        "net_5bp_annualized_return_mean",
        "net_5bp_sharpe_mean",

        "net_10bp_annualized_return_mean",
        "net_10bp_sharpe_mean",
    ]

    cost_columns = [
        column
        for column in cost_columns
        if column in grouped.columns
    ]

    grouped[
        cost_columns
    ].to_csv(
        OUTPUT_DIR
        / "transaction_cost_table.csv",
        index=False,
    )

    structure_columns = [
        "model",
        "lambda",

        "active_assets_mean",
        "active_assets_std",

        "hhi_mean",
        "hhi_std",

        "effective_n_mean",
        "effective_n_std",

        "annualized_turnover_mean",
        "annualized_turnover_std",
    ]

    structure_columns = [
        column
        for column in structure_columns
        if column in grouped.columns
    ]

    grouped[
        structure_columns
    ].to_csv(
        OUTPUT_DIR
        / "portfolio_structure_table.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "COMMON EVALUATION COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Learned runs : "
        f"{len(runs)}"
    )

    print(
        f"Test periods : "
        f"{len(canonical_dates)}"
    )

    print(
        f"N assets     : "
        f"{len(canonical_tickers)}"
    )

    print(
        "\nGross Sharpe"
    )

    sharpe_table = (
        grouped[
            [
                "model",
                "lambda",
                "gross_sharpe_mean",
                "gross_sharpe_std",
            ]
        ]
        .sort_values(
            [
                "lambda",
                "gross_sharpe_mean",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print(
        sharpe_table.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print(
        "\nEqual Weight"
    )

    for key in [
        "gross_annualized_return",
        "gross_volatility",
        "gross_sharpe",
        "gross_sortino",
        "gross_max_drawdown",
        "gross_calmar",
        "gross_cvar_95_daily",
        "annualized_turnover",
        "hhi",
        "effective_n",
    ]:
        if key in ew_metrics:
            print(
                f"{key:28s}: "
                f"{ew_metrics[key]:.6f}"
            )

    print(
        "\nSaved:"
    )

    for filename in [
        "all_runs_common_metrics.csv",
        "model_summary_mean_std.csv",
        "performance_table.csv",
        "transaction_cost_table.csv",
        "portfolio_structure_table.csv",
        "equal_weight_metrics.csv",
    ]:
        print(
            f"  {OUTPUT_DIR / filename}"
        )


if __name__ == "__main__":
    main()