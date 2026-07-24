from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="paper_main_v1")
    parser.add_argument("--dataset-name", default="dow30")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_results(experiment_root: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for summary_path in experiment_root.glob(
        "alpha_*/lambda_*/maxw_*/seed_*/summary.json"
    ):
        summary = read_json(summary_path)
        config_path = summary_path.parent / "config.json"
        config = read_json(config_path) if config_path.exists() else {}
        arguments = config.get("arguments", {})

        rows.append(
            {
                "alpha": float(summary["alpha"]),
                "risk_aversion": float(summary["lambda"]),
                "max_weight": float(summary["max_weight"]),
                "seed": int(summary["seed"]),
                "best_epoch": int(summary["best_epoch"]),
                "validation_combined": float(summary["validation_combined"]),
                "validation_regret": float(summary["validation_regret"]),
                "validation_mse": float(summary["validation_mse"]),
                "test_combined": float(summary["test_combined"]),
                "test_regret": float(summary["test_regret"]),
                "test_mse": float(summary["test_mse"]),
                "annualized_return_cagr": float(summary["annualized_return_cagr"]),
                "sharpe_ratio": float(summary["sharpe_ratio"]),
                "maximum_drawdown": float(summary["maximum_drawdown"]),
                "final_wealth": float(summary["final_wealth"]),
                "average_active_assets": float(summary["average_active_assets"]),
                "learning_rate": arguments.get("learning_rate"),
                "weight_decay": arguments.get("weight_decay"),
                "hidden_dim": arguments.get("hidden_dim"),
                "dropout": arguments.get("dropout"),
                "batch_size": arguments.get("batch_size"),
                "epochs": arguments.get("epochs"),
                "patience": arguments.get("patience"),
                "summary_path": str(summary_path),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"summary.json을 찾지 못했습니다: {experiment_root}"
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["risk_aversion", "alpha", "max_weight", "seed"])
        .reset_index(drop=True)
    )


def build_group_summary(results: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["alpha", "risk_aversion", "max_weight"]
    metric_columns = [
        "test_regret",
        "test_mse",
        "annualized_return_cagr",
        "sharpe_ratio",
        "maximum_drawdown",
        "final_wealth",
        "average_active_assets",
        "best_epoch",
    ]

    grouped = (
        results.groupby(group_columns)[metric_columns]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    grouped.columns = [
        column[0]
        if isinstance(column, tuple) and column[1] == ""
        else "_".join(column)
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]

    return grouped.sort_values(
        ["risk_aversion", "alpha", "max_weight"]
    ).reset_index(drop=True)


def build_best_by_lambda(grouped: pd.DataFrame) -> pd.DataFrame:
    best_rows = []

    for _, subset in grouped.groupby("risk_aversion"):
        best_rows.append(
            subset.loc[subset["test_regret_mean"].idxmin()]
        )

    return pd.DataFrame(best_rows).reset_index(drop=True)


def check_grid_status(
    experiment_root: Path,
    results: pd.DataFrame,
) -> pd.DataFrame:
    status_path = experiment_root / "grid_status.csv"
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()

    grid_config_path = experiment_root / "grid_config.json"
    expected_runs = None

    if grid_config_path.exists():
        expected_runs = read_json(grid_config_path).get("total_runs")

    return pd.DataFrame(
        [
            {
                "expected_runs": expected_runs,
                "actual_summary_files": len(results),
                "duplicate_runs": int(
                    results.duplicated(
                        ["alpha", "risk_aversion", "max_weight", "seed"]
                    ).sum()
                ),
                "status_rows": len(status),
                "completed": (
                    int((status["status"] == "completed").sum())
                    if not status.empty
                    else None
                ),
                "failed": (
                    int((status["status"] == "failed").sum())
                    if not status.empty
                    else None
                ),
                "skipped_existing": (
                    int((status["status"] == "skipped_existing").sum())
                    if not status.empty
                    else None
                ),
            }
        ]
    )


def save_line_plot(
    grouped: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))

    for risk_aversion, subset in grouped.groupby("risk_aversion"):
        subset = subset.sort_values("alpha")
        axis.errorbar(
            subset["alpha"],
            subset[metric_mean],
            yerr=subset[metric_std].fillna(0),
            marker="o",
            capsize=3,
            label=f"lambda={risk_aversion:g}",
        )

    axis.set_xlabel("Alpha")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


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
            f"실험 폴더가 없습니다: {experiment_root}"
        )

    analysis_dir = experiment_root / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(experiment_root)
    grouped = build_group_summary(results)
    best_by_lambda = build_best_by_lambda(grouped)
    completion = check_grid_status(experiment_root, results)

    results.to_csv(
        analysis_dir / "results_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grouped.to_csv(
        analysis_dir / "results_grouped.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_by_lambda.to_csv(
        analysis_dir / "best_by_lambda.csv",
        index=False,
        encoding="utf-8-sig",
    )
    completion.to_csv(
        analysis_dir / "completion_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_line_plot(
        grouped,
        "test_regret_mean",
        "test_regret_std",
        "Test regret",
        "Test regret by alpha and risk aversion",
        analysis_dir / "test_regret.png",
    )
    save_line_plot(
        grouped,
        "test_mse_mean",
        "test_mse_std",
        "Test MSE",
        "Test MSE by alpha and risk aversion",
        analysis_dir / "test_mse.png",
    )
    save_line_plot(
        grouped,
        "sharpe_ratio_mean",
        "sharpe_ratio_std",
        "Sharpe ratio",
        "Sharpe ratio by alpha and risk aversion",
        analysis_dir / "sharpe_ratio.png",
    )
    save_line_plot(
        grouped,
        "average_active_assets_mean",
        "average_active_assets_std",
        "Average active assets",
        "Average active assets by alpha and risk aversion",
        analysis_dir / "active_assets.png",
    )

    print("=" * 80)
    print("Completion check")
    print("=" * 80)
    print(completion.to_string(index=False))
    print()
    print("=" * 80)
    print("Best alpha by lambda based on mean test regret")
    print("=" * 80)
    print(
        best_by_lambda[
            [
                "risk_aversion",
                "alpha",
                "max_weight",
                "test_regret_mean",
                "test_regret_std",
                "test_mse_mean",
                "sharpe_ratio_mean",
                "maximum_drawdown_mean",
                "average_active_assets_mean",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Saved analysis directory: {analysis_dir}")


if __name__ == "__main__":
    main()
