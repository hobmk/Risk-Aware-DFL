from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cumulative returns by alpha and/or max-weight."
    )

    parser.add_argument("--experiment-name", default="final_full_grid_h64_d0_s30")
    parser.add_argument("--dataset-name", default="dow30")

    parser.add_argument(
        "--comparison",
        choices=["alpha", "max_weight", "both"],
        default="both",
        help="alpha 비교, max-weight 비교, 또는 둘 다 생성",
    )

    parser.add_argument(
        "--risk-aversions",
        type=float,
        nargs="+",
        default=[0.5],
        help="alpha 비교에 사용할 lambda 목록",
    )

    parser.add_argument(
        "--alpha-max-weights",
        type=float,
        nargs="+",
        default=[0.1],
        help="alpha 비교에 사용할 max-weight 목록",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="max-weight 비교 시 고정할 alpha",
    )

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="alpha 비교에 포함할 alpha 목록",
    )

    parser.add_argument(
        "--max-weights",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 1.0],
        help="max-weight 비교에 포함할 max-weight 목록",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="사용할 seed 목록",
    )

    parser.add_argument("--dpi", type=int, default=300)

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_experiment_root(
    dataset_name: str,
    experiment_name: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "combined"
        / dataset_name
        / experiment_name
    )


def approximately_equal(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def collect_matching_run_dirs(
    experiment_root: Path,
    alpha: float,
    risk_aversion: float,
    max_weight: float,
    seeds: list[int],
) -> list[Path]:
    matched: list[Path] = []

    run_dirs = sorted(
        path
        for path in experiment_root.glob("alpha_*/lambda_*/maxw_*/seed_*")
        if path.is_dir()
    )

    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        daily_path = run_dir / "daily_portfolio.csv"

        if not summary_path.exists() or not daily_path.exists():
            continue

        summary = load_json(summary_path)

        run_alpha = float(summary["alpha"])
        run_lambda = float(summary["lambda"])
        run_max_weight = float(summary["max_weight"])
        run_seed = int(summary["seed"])

        if not approximately_equal(run_alpha, alpha):
            continue
        if not approximately_equal(run_lambda, risk_aversion):
            continue
        if not approximately_equal(run_max_weight, max_weight):
            continue
        if run_seed not in seeds:
            continue

        matched.append(run_dir)

    matched = sorted(
        matched,
        key=lambda path: int(load_json(path / "summary.json")["seed"]),
    )

    return matched


def load_seed_wealth(run_dir: Path) -> pd.DataFrame:
    daily_path = run_dir / "daily_portfolio.csv"
    summary = load_json(run_dir / "summary.json")
    seed = int(summary["seed"])

    daily = pd.read_csv(daily_path)

    if "date" not in daily.columns:
        raise KeyError(f"'date' column not found: {daily_path}")
    if "portfolio_return" not in daily.columns:
        raise KeyError(f"'portfolio_return' column not found: {daily_path}")

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    daily["wealth"] = (1.0 + daily["portfolio_return"]).cumprod()

    return daily[["date", "wealth"]].rename(
        columns={"wealth": f"seed_{seed}"}
    )


def aggregate_wealth_across_seeds(run_dirs: list[Path]) -> pd.DataFrame:
    if not run_dirs:
        raise RuntimeError("No matching run directories found.")

    merged: pd.DataFrame | None = None

    for run_dir in run_dirs:
        seed_wealth = load_seed_wealth(run_dir)

        if merged is None:
            merged = seed_wealth
        else:
            merged = merged.merge(seed_wealth, on="date", how="inner")

    assert merged is not None

    seed_columns = [column for column in merged.columns if column.startswith("seed_")]
    if not seed_columns:
        raise RuntimeError("No seed columns found after merge.")

    merged["wealth_mean"] = merged[seed_columns].mean(axis=1)
    merged["wealth_std"] = merged[seed_columns].std(axis=1).fillna(0.0)

    return merged


def save_alpha_comparison(
    experiment_root: Path,
    output_dir: Path,
    risk_aversion: float,
    max_weight: float,
    alphas: list[float],
    seeds: list[int],
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))

    summary_rows: list[dict] = []

    for alpha in alphas:
        run_dirs = collect_matching_run_dirs(
            experiment_root=experiment_root,
            alpha=alpha,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            seeds=seeds,
        )

        if len(run_dirs) != len(seeds):
            raise RuntimeError(
                "alpha comparison에서 필요한 seed 수가 맞지 않습니다. "
                f"alpha={alpha}, lambda={risk_aversion}, max_weight={max_weight}, "
                f"expected={len(seeds)}, found={len(run_dirs)}"
            )

        aggregated = aggregate_wealth_across_seeds(run_dirs)

        label = f"alpha={alpha:g}"
        axis.plot(
            aggregated["date"],
            aggregated["wealth_mean"],
            label=label,
            linewidth=2.5,
        )

        aggregated.to_csv(
            output_dir / f"alpha_{alpha:g}_lambda_{risk_aversion:g}_maxw_{max_weight:g}_wealth.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary_rows.append(
            {
                "alpha": alpha,
                "risk_aversion": risk_aversion,
                "max_weight": max_weight,
                "final_wealth_mean": aggregated["wealth_mean"].iloc[-1],
                "final_wealth_std": aggregated["wealth_std"].iloc[-1],
                "n_seeds": len(run_dirs),
            }
        )

    axis.set_title(
        f"Cumulative Wealth by Alpha (lambda={risk_aversion:g}, max_weight={max_weight:g})"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative Wealth")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_dir / f"cumulative_return_alpha_lambda_{risk_aversion:g}_maxw_{max_weight:g}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / f"cumulative_return_alpha_lambda_{risk_aversion:g}_maxw_{max_weight:g}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_max_weight_comparison(
    experiment_root: Path,
    output_dir: Path,
    risk_aversion: float,
    alpha: float,
    max_weights: list[float],
    seeds: list[int],
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))

    summary_rows: list[dict] = []

    for max_weight in max_weights:
        run_dirs = collect_matching_run_dirs(
            experiment_root=experiment_root,
            alpha=alpha,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            seeds=seeds,
        )

        if len(run_dirs) != len(seeds):
            raise RuntimeError(
                "max-weight comparison에서 필요한 seed 수가 맞지 않습니다. "
                f"alpha={alpha}, lambda={risk_aversion}, max_weight={max_weight}, "
                f"expected={len(seeds)}, found={len(run_dirs)}"
            )

        aggregated = aggregate_wealth_across_seeds(run_dirs)

        label = f"max_weight={max_weight:g}"
        axis.plot(
            aggregated["date"],
            aggregated["wealth_mean"],
            label=label,
            linewidth=2.5,
        )

        aggregated.to_csv(
            output_dir / f"alpha_{alpha:g}_lambda_{risk_aversion:g}_maxw_{max_weight:g}_wealth.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary_rows.append(
            {
                "alpha": alpha,
                "risk_aversion": risk_aversion,
                "max_weight": max_weight,
                "final_wealth_mean": aggregated["wealth_mean"].iloc[-1],
                "final_wealth_std": aggregated["wealth_std"].iloc[-1],
                "n_seeds": len(run_dirs),
            }
        )

    axis.set_title(
        f"Cumulative Wealth by Max Weight (alpha={alpha:g}, lambda={risk_aversion:g})"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative Wealth")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_dir / f"cumulative_return_maxw_alpha_{alpha:g}_lambda_{risk_aversion:g}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / f"cumulative_return_maxw_alpha_{alpha:g}_lambda_{risk_aversion:g}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()

    experiment_root = resolve_experiment_root(
        dataset_name=args.dataset_name,
        experiment_name=args.experiment_name,
    )

    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {experiment_root}")

    output_dir = (
        experiment_root
        / "_paper_report"
        / "cumulative_returns"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Compare cumulative returns")
    print("=" * 80)
    print(f"Experiment root: {experiment_root}")
    print(f"Output dir: {output_dir}")
    print(f"Seeds: {args.seeds}")

    if args.comparison in {"alpha", "both"}:
        print()

        for risk_aversion in args.risk_aversions:
            for max_weight in args.alpha_max_weights:
                print(
                    f"[Alpha comparison] "
                    f"lambda={risk_aversion:g}, "
                    f"max_weight={max_weight:g}"
                )

                save_alpha_comparison(
                    experiment_root=experiment_root,
                    output_dir=output_dir,
                    risk_aversion=risk_aversion,
                    max_weight=max_weight,
                    alphas=args.alphas,
                    seeds=args.seeds,
                    dpi=args.dpi,
                )

        print("Alpha comparison saved.")

    if args.comparison in {"max_weight", "both"}:
        print()
        print(
            f"[Max-weight comparison] alpha={args.alpha:g}, lambda={args.risk_aversion:g}"
        )
        save_max_weight_comparison(
            experiment_root=experiment_root,
            output_dir=output_dir,
            risk_aversion=args.risk_aversion,
            alpha=args.alpha,
            max_weights=args.max_weights,
            seeds=args.seeds,
            dpi=args.dpi,
        )
        print("Max-weight comparison saved.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()