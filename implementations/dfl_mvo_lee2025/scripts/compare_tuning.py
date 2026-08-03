from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_LAMBDAS = [0.1, 0.5, 1.0, 5.0]

METRICS = [
    "test_mse",
    "test_regret",
    "annualized_return_mean",
    "annualized_return_cagr",
    "sharpe_ratio",
    "maximum_drawdown",
    "final_wealth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    parser.add_argument("--max-weight", type=float, default=1.0)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])

    parser.add_argument(
        "--comparison-type",
        choices=["tuning", "standardization"],
        default="tuning",
    )

    parser.add_argument("--dpi", type=int, default=300)

    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_directory(
    root: Path,
    alpha: float,
    risk_aversion: float,
    max_weight: float,
    seed: int,
) -> Path:
    return (
        root
        / f"alpha_{alpha:.2f}"
        / f"lambda_{risk_aversion:.2f}"
        / f"maxw_{max_weight:.2f}"
        / f"seed_{seed}"
    )


def load_seed_result(
    root: Path,
    condition: str,
    alpha: float,
    risk_aversion: float,
    max_weight: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    directory = run_directory(
        root=root,
        alpha=alpha,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        seed=seed,
    )

    summary_path = directory / "summary.json"
    daily_path = directory / "daily_portfolio.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing summary.json: {summary_path}"
        )

    if not daily_path.exists():
        raise FileNotFoundError(
            f"Missing daily_portfolio.csv: {daily_path}"
        )

    summary = load_json(summary_path)

    daily = pd.read_csv(
        daily_path,
        usecols=[
            "date",
            "portfolio_return",
        ],
    )

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily["portfolio_return"] = pd.to_numeric(
        daily["portfolio_return"],
        errors="raise",
    )

    daily = (
        daily.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    daily["wealth"] = (
        1.0 + daily["portfolio_return"]
    ).cumprod()

    metrics = {
        "condition": condition,
        "alpha": alpha,
        "risk_aversion": risk_aversion,
        "max_weight": max_weight,
        "seed": seed,
        "test_mse": float(
            summary["test_mse"]
        ),
        "test_regret": float(
            summary["test_regret"]
        ),
        "annualized_return_mean": float(
            daily["portfolio_return"].mean()
            * 252.0
        ),
        "annualized_return_cagr": float(
            summary["annualized_return_cagr"]
        ),
        "sharpe_ratio": float(
            summary["sharpe_ratio"]
        ),
        "maximum_drawdown": float(
            summary["maximum_drawdown"]
        ),
        "final_wealth": float(
            summary["final_wealth"]
        ),
    }

    return (
        daily[
            [
                "date",
                "wealth",
            ]
        ],
        metrics,
    )


def aggregate_condition(
    root: Path,
    condition: str,
    alpha: float,
    risk_aversion: float,
    max_weight: float,
    seeds: list[int],
) -> tuple[pd.DataFrame, list[dict]]:
    curves: list[pd.DataFrame] = []
    metrics: list[dict] = []

    for seed in seeds:
        curve, row = load_seed_result(
            root=root,
            condition=condition,
            alpha=alpha,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            seed=seed,
        )

        curves.append(
            curve.rename(
                columns={
                    "wealth": f"seed_{seed}"
                }
            )
        )

        metrics.append(row)

    merged = curves[0]

    for curve in curves[1:]:
        merged = merged.merge(
            curve,
            on="date",
            how="inner",
            validate="one_to_one",
        )

    seed_columns = [
        column
        for column in merged.columns
        if column.startswith("seed_")
    ]

    merged["wealth_mean"] = merged[
        seed_columns
    ].mean(axis=1)

    merged["wealth_std"] = merged[
        seed_columns
    ].std(axis=1, ddof=1)

    return merged, metrics


def summarize_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    grouped = metrics.groupby(
        [
            "condition",
            "alpha",
            "risk_aversion",
            "max_weight",
        ],
        sort=True,
    )

    for keys, frame in grouped:
        (
            condition,
            alpha,
            risk_aversion,
            max_weight,
        ) = keys

        row = {
            "condition": condition,
            "alpha": alpha,
            "risk_aversion": risk_aversion,
            "max_weight": max_weight,
            "n_seeds": len(frame),
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

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "risk_aversion",
                "alpha",
                "condition",
            ]
        )
        .reset_index(drop=True)
    )


def plot_comparison(
    before: pd.DataFrame,
    after: pd.DataFrame,
    before_condition: str,
    after_condition: str,
    before_label: str,
    after_label: str,
    title_prefix: str,
    alpha: float,
    risk_aversion: float,
    max_weight: float,
    output_path: Path,
    dpi: int,
) -> pd.DataFrame:
    comparison = (
        before[
            [
                "date",
                "wealth_mean",
            ]
        ]
        .rename(
            columns={
                "wealth_mean":
                    before_condition
            }
        )
        .merge(
            after[
                [
                    "date",
                    "wealth_mean",
                ]
            ].rename(
                columns={
                    "wealth_mean":
                        after_condition
                }
            ),
            on="date",
            how="inner",
            validate="one_to_one",
        )
    )

    figure, axis = plt.subplots(
        figsize=(10.5, 5.4)
    )

    axis.plot(
        comparison["date"],
        comparison[before_condition],
        linewidth=1.8,
        label=before_label,
    )

    axis.plot(
        comparison["date"],
        comparison[after_condition],
        linewidth=1.8,
        label=after_label,
    )

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
    )

    axis.set_title(
        rf"{title_prefix}  "
        rf"($\alpha={alpha:g}$, "
        rf"$\lambda={risk_aversion:g}$, "
        rf"$w_{{max}}={max_weight:g}$)"
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative Wealth")

    axis.grid(
        True,
        alpha=0.25,
        linewidth=0.6,
    )

    axis.legend()

    axis.xaxis.set_major_locator(
        mdates.MonthLocator(
            interval=3
        )
    )

    axis.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y-%m"
        )
    )

    axis.tick_params(
        axis="x",
        rotation=35,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    return comparison


def main() -> None:
    args = parse_args()

    before_root = resolve_path(
        args.before_root
    )

    after_root = resolve_path(
        args.after_root
    )

    if not before_root.exists():
        raise FileNotFoundError(
            "Before root does not exist: "
            f"{before_root}"
        )

    if not after_root.exists():
        raise FileNotFoundError(
            "After root does not exist: "
            f"{after_root}"
        )

    if args.comparison_type == "tuning":
        before_condition = "before_tuning"
        after_condition = "after_tuning"

        before_label = "Before tuning"
        after_label = "After tuning"

        title_prefix = (
            "Cumulative Wealth: "
            "Before vs After Tuning"
        )

        output_name = (
            "tuning_comparison"
        )

    else:
        before_condition = (
            "without_standardization"
        )

        after_condition = (
            "with_standardization"
        )

        before_label = (
            "Without standardization"
        )

        after_label = (
            "With standardization"
        )

        title_prefix = (
            "Cumulative Wealth: "
            "Standardization Effect"
        )

        output_name = (
            "standardization_comparison"
        )

    output_dir = (
        after_root
        / "_paper_report"
        / output_name
        / f"maxw_{args.max_weight:.2f}"
    )

    curve_dir = (
        output_dir
        / "curves"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    curve_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics: list[dict] = []

    for risk_aversion in args.lambdas:
        for alpha in args.alphas:
            (
                before_curve,
                before_metrics,
            ) = aggregate_condition(
                root=before_root,
                condition=before_condition,
                alpha=alpha,
                risk_aversion=
                    risk_aversion,
                max_weight=
                    args.max_weight,
                seeds=args.seeds,
            )

            (
                after_curve,
                after_metrics,
            ) = aggregate_condition(
                root=after_root,
                condition=after_condition,
                alpha=alpha,
                risk_aversion=
                    risk_aversion,
                max_weight=
                    args.max_weight,
                seeds=args.seeds,
            )

            all_metrics.extend(
                before_metrics
            )

            all_metrics.extend(
                after_metrics
            )

            stem = (
                f"alpha_{alpha:.2f}_"
                f"lambda_{risk_aversion:.2f}_"
                f"maxw_{args.max_weight:.2f}"
            )

            comparison = plot_comparison(
                before=before_curve,
                after=after_curve,
                before_condition=
                    before_condition,
                after_condition=
                    after_condition,
                before_label=
                    before_label,
                after_label=
                    after_label,
                title_prefix=
                    title_prefix,
                alpha=alpha,
                risk_aversion=
                    risk_aversion,
                max_weight=
                    args.max_weight,
                output_path=
                    output_dir
                    / f"{stem}.png",
                dpi=args.dpi,
            )

            comparison.to_csv(
                curve_dir
                / f"{stem}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(
                f"Created: {stem}"
            )

    metrics = pd.DataFrame(
        all_metrics
    )

    metrics.to_csv(
        output_dir
        / f"{output_name}_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_metrics(
        metrics
    )

    summary.to_csv(
        output_dir
        / f"{output_name}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()