from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

METRICS = [
    "test_mse",
    "test_regret",
    "annualized_return_mean",
    "annualized_return_cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "final_wealth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="final_full_grid_h64_d0_s30")
    parser.add_argument("--dataset-name", default="dow30")
    parser.add_argument("--output-name", default="full_grid_tables")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-weights", nargs="+", type=float, default=[0.1, 0.2, 1.0])
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_folder_value(name: str, prefix: str) -> float:
    if not name.startswith(prefix):
        raise ValueError(f"잘못된 폴더명입니다: {name}")

    return float(name.removeprefix(prefix))


def parse_seed(name: str) -> int:
    if not name.startswith("seed_"):
        raise ValueError(f"잘못된 seed 폴더명입니다: {name}")

    return int(name.removeprefix("seed_"))


def summary_value(
    summary: dict,
    key: str,
    fallback: float,
) -> float:
    value = summary.get(key)

    if value is None:
        return float(fallback)

    return float(value)


def calculate_daily_metrics(
    daily: pd.DataFrame,
) -> dict[str, float]:
    returns = pd.to_numeric(
        daily["portfolio_return"],
        errors="raise",
    ).astype(float)

    wealth = (1.0 + returns).cumprod()

    n_days = len(returns)
    mean_daily = float(returns.mean())
    std_daily = float(returns.std(ddof=1))

    annualized_return_mean = mean_daily * 252.0
    annualized_volatility = std_daily * np.sqrt(252.0)

    if annualized_volatility > 0:
        sharpe_ratio = (
            annualized_return_mean
            / annualized_volatility
        )
    else:
        sharpe_ratio = np.nan

    final_wealth = float(wealth.iloc[-1])

    if n_days > 0 and final_wealth > 0:
        annualized_return_cagr = (
            final_wealth ** (252.0 / n_days)
            - 1.0
        )
    else:
        annualized_return_cagr = np.nan

    drawdown = wealth / wealth.cummax() - 1.0
    maximum_drawdown = float(-drawdown.min())

    return {
        "annualized_return_mean":
            annualized_return_mean,
        "annualized_return_cagr":
            annualized_return_cagr,
        "annualized_volatility":
            annualized_volatility,
        "sharpe_ratio":
            sharpe_ratio,
        "maximum_drawdown":
            maximum_drawdown,
        "final_wealth":
            final_wealth,
    }


def load_run(summary_path: Path) -> dict:
    run_dir = summary_path.parent

    seed = parse_seed(
        run_dir.name
    )

    max_weight = parse_folder_value(
        run_dir.parent.name,
        "maxw_",
    )

    risk_aversion = parse_folder_value(
        run_dir.parent.parent.name,
        "lambda_",
    )

    alpha = parse_folder_value(
        run_dir.parent.parent.parent.name,
        "alpha_",
    )

    daily_path = (
        run_dir
        / "daily_portfolio.csv"
    )

    if not daily_path.exists():
        raise FileNotFoundError(
            f"daily_portfolio.csv 없음: "
            f"{daily_path}"
        )

    summary = load_json(
        summary_path
    )

    daily = pd.read_csv(
        daily_path
    )

    required_columns = {
        "date",
        "portfolio_return",
    }

    missing_columns = (
        required_columns
        - set(daily.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{daily_path}에 필요한 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily = (
        daily
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    calculated = calculate_daily_metrics(
        daily
    )

    return {
        "alpha": alpha,
        "risk_aversion": risk_aversion,
        "max_weight": max_weight,
        "seed": seed,
        "start_date":
            daily["date"].min().date(),
        "end_date":
            daily["date"].max().date(),
        "n_test_days":
            len(daily),
        "test_mse":
            float(summary["test_mse"]),
        "test_regret":
            float(summary["test_regret"]),
        "annualized_return_mean":
            summary_value(
                summary,
                "annualized_return_mean",
                calculated[
                    "annualized_return_mean"
                ],
            ),
        "annualized_return_cagr":
            summary_value(
                summary,
                "annualized_return_cagr",
                calculated[
                    "annualized_return_cagr"
                ],
            ),
        "annualized_volatility":
            summary_value(
                summary,
                "annualized_volatility",
                calculated[
                    "annualized_volatility"
                ],
            ),
        "sharpe_ratio":
            summary_value(
                summary,
                "sharpe_ratio",
                calculated[
                    "sharpe_ratio"
                ],
            ),
        "maximum_drawdown":
            summary_value(
                summary,
                "maximum_drawdown",
                calculated[
                    "maximum_drawdown"
                ],
            ),
        "final_wealth":
            summary_value(
                summary,
                "final_wealth",
                calculated[
                    "final_wealth"
                ],
            ),
    }


def aggregate_results(
    by_seed: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "alpha",
        "risk_aversion",
        "max_weight",
    ]

    grouped = by_seed.groupby(
        group_columns,
        sort=True,
    )

    rows: list[dict] = []

    for keys, frame in grouped:
        alpha, risk_aversion, max_weight = keys

        row = {
            "alpha": alpha,
            "risk_aversion":
                risk_aversion,
            "max_weight":
                max_weight,
            "n_seeds":
                len(frame),
        }

        for metric in METRICS:
            row[
                f"{metric}_mean"
            ] = frame[metric].mean()

            row[
                f"{metric}_std"
            ] = frame[metric].std(
                ddof=1
            )

            row[
                f"{metric}_min"
            ] = frame[metric].min()

            row[
                f"{metric}_max"
            ] = frame[metric].max()

        rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    rank_groups = summary.groupby(
        [
            "risk_aversion",
            "max_weight",
        ]
    )

    summary[
        "final_wealth_rank"
    ] = rank_groups[
        "final_wealth_mean"
    ].rank(
        ascending=False,
        method="min",
    )

    summary[
        "sharpe_ratio_rank"
    ] = rank_groups[
        "sharpe_ratio_mean"
    ].rank(
        ascending=False,
        method="min",
    )

    summary[
        "test_regret_rank"
    ] = rank_groups[
        "test_regret_mean"
    ].rank(
        ascending=True,
        method="min",
    )

    return (
        summary
        .sort_values(
            [
                "risk_aversion",
                "max_weight",
                "alpha",
            ]
        )
        .reset_index(drop=True)
    )


def make_pivot_table(
    summary: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    return (
        summary.pivot_table(
            index=[
                "risk_aversion",
                "max_weight",
            ],
            columns="alpha",
            values=f"{metric}_mean",
            aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )


def make_best_by_constraint(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    criteria = {
        "final_wealth": (
            "final_wealth_mean",
            False,
        ),
        "sharpe_ratio": (
            "sharpe_ratio_mean",
            False,
        ),
        "annualized_return_cagr": (
            "annualized_return_cagr_mean",
            False,
        ),
        "maximum_drawdown": (
            "maximum_drawdown_mean",
            True,
        ),
        "test_regret": (
            "test_regret_mean",
            True,
        ),
        "test_mse": (
            "test_mse_mean",
            True,
        ),
    }

    rows: list[dict] = []

    grouped = summary.groupby(
        [
            "risk_aversion",
            "max_weight",
        ],
        sort=True,
    )

    for keys, frame in grouped:
        risk_aversion, max_weight = keys

        for criterion, (
            column,
            lower_is_better,
        ) in criteria.items():
            if lower_is_better:
                best_index = frame[
                    column
                ].idxmin()
            else:
                best_index = frame[
                    column
                ].idxmax()

            best = summary.loc[
                best_index
            ]

            rows.append(
                {
                    "risk_aversion":
                        risk_aversion,
                    "max_weight":
                        max_weight,
                    "criterion":
                        criterion,
                    "best_alpha":
                        best["alpha"],
                    "best_value":
                        best[column],
                    "direction":
                        (
                            "min"
                            if lower_is_better
                            else "max"
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def make_missing_runs(
    by_seed: pd.DataFrame,
) -> pd.DataFrame:
    alphas = sorted(
        by_seed["alpha"].unique()
    )

    risk_aversions = sorted(
        by_seed[
            "risk_aversion"
        ].unique()
    )

    max_weights = sorted(
        by_seed[
            "max_weight"
        ].unique()
    )

    seeds = sorted(
        by_seed["seed"].unique()
    )

    existing = {
        (
            row.alpha,
            row.risk_aversion,
            row.max_weight,
            row.seed,
        )
        for row in by_seed.itertuples()
    }

    rows = []

    for combination in product(
        alphas,
        risk_aversions,
        max_weights,
        seeds,
    ):
        if combination in existing:
            continue

        alpha, risk_aversion, max_weight, seed = (
            combination
        )

        rows.append(
            {
                "alpha": alpha,
                "risk_aversion":
                    risk_aversion,
                "max_weight":
                    max_weight,
                "seed": seed,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "alpha",
            "risk_aversion",
            "max_weight",
            "seed",
        ],
    )


def main() -> None:
    args = parse_args()

    experiment_root = (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "combined"
        / args.dataset_name
        / args.experiment_name
    )

    if not experiment_root.exists():
        raise FileNotFoundError(
            f"실험 폴더가 없습니다: "
            f"{experiment_root}"
        )

    summary_paths = sorted(
        experiment_root.glob(
            "alpha_*/lambda_*/"
            "maxw_*/seed_*/summary.json"
        )
    )

    if not summary_paths:
        raise RuntimeError(
            f"summary.json을 찾지 못했습니다: "
            f"{experiment_root}"
        )

    rows = []

    for summary_path in summary_paths:
        rows.append(
            load_run(summary_path)
        )

    by_seed = pd.DataFrame(rows)

    by_seed = by_seed[
        by_seed["seed"].isin(args.seeds)
    ].copy()

    max_weight_mask = np.zeros(
        len(by_seed),
        dtype=bool,
    )

    for max_weight in args.max_weights:
        max_weight_mask |= np.isclose(
            by_seed["max_weight"].to_numpy(),
            max_weight,
        )

    by_seed = (
        by_seed[max_weight_mask]
        .sort_values(
            [
                "risk_aversion",
                "max_weight",
                "alpha",
                "seed",
            ]
        )
        .reset_index(drop=True)
    )

    if by_seed.empty:
        raise RuntimeError(
            "지정한 seed와 max_weight에 해당하는 결과가 없습니다."
        )

    summary = aggregate_results(
        by_seed
    )

    best_by_constraint = (
        make_best_by_constraint(
            summary
        )
    )

    missing_runs = make_missing_runs(
        by_seed
    )

    output_dir = (
        experiment_root
        / "_paper_report"
        / args.output_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_seed.to_csv(
        output_dir
        / "full_grid_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        output_dir
        / "full_grid_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_by_constraint.to_csv(
        output_dir
        / "best_by_constraint.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_runs.to_csv(
        output_dir
        / "missing_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pivot_metrics = [
        "final_wealth",
        "sharpe_ratio",
        "annualized_return_mean",
        "annualized_return_cagr",
        "maximum_drawdown",
        "test_regret",
        "test_mse",
    ]

    for metric in pivot_metrics:
        pivot = make_pivot_table(
            summary,
            metric,
        )

        pivot.to_csv(
            output_dir
            / f"{metric}_by_alpha.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(
        f"Loaded runs: {len(by_seed)}"
    )

    print(
        f"Aggregated settings: "
        f"{len(summary)}"
    )

    print(
        f"Missing runs: "
        f"{len(missing_runs)}"
    )

    print(
        f"Saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()