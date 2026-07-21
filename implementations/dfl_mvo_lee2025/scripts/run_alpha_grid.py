from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--mse-scale",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
    "--hidden-dim",
    type=int,
    default=128,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    return parser.parse_args()


def get_dataset_name(price_csv: str) -> str:
    return Path(price_csv).stem.replace(
        "_adjusted_close",
        "",
    )


def get_run_directory(
    dataset_name: str,
    alpha: float,
    risk_aversion: float,
    max_weight: float,
    seed: int,
) -> Path:
    return (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "combined"
        / dataset_name
        / f"alpha_{alpha:.2f}"
        / f"lambda_{risk_aversion:.2f}"
        / f"maxw_{max_weight:.2f}"
        / f"seed_{seed}"
    )


def run_experiment(
    args: argparse.Namespace,
    alpha: float,
    dataset_name: str,
) -> bool:
    run_directory = get_run_directory(
        dataset_name=dataset_name,
        alpha=alpha,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        seed=args.seed,
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = run_directory / "run.log"

    command = [
        sys.executable,
        "-u",
        "-m",
        "implementations.dfl_mvo_lee2025.scripts."
        "train_mlp_markowitz_combined",
        "--price-csv",
        args.price_csv,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--alpha",
        str(alpha),
        "--mse-scale",
        str(args.mse_scale),
        "--risk-aversion",
        str(args.risk_aversion),
        "--max-weight",
        str(args.max_weight),
        "--seed",
        str(args.seed),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--active-threshold",
        str(args.active_threshold),
    ]

    print("=" * 80)
    print(
        f"Starting alpha={alpha:.2f}, "
        f"lambda={args.risk_aversion:.2f}, "
        f"seed={args.seed}"
    )
    print("Log:", log_path)
    print("=" * 80)

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError(
                "학습 프로세스의 출력을 읽을 수 없습니다."
            )

        for line in process.stdout:
            print(
                line,
                end="",
            )
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        print(
            f"[FAILED] alpha={alpha:.2f}, "
            f"return code={return_code}"
        )
        return False

    summary_path = run_directory / "summary.json"

    if not summary_path.exists():
        print(
            f"[FAILED] summary.json이 생성되지 않았습니다: "
            f"{summary_path}"
        )
        return False

    print(
        f"[DONE] alpha={alpha:.2f}"
    )
    return True


def create_grid_summary(
    args: argparse.Namespace,
    dataset_name: str,
) -> Path:
    rows: list[dict] = []

    for alpha in args.alphas:
        run_directory = get_run_directory(
            dataset_name=dataset_name,
            alpha=alpha,
            risk_aversion=args.risk_aversion,
            max_weight=args.max_weight,
            seed=args.seed,
        )

        summary_path = run_directory / "summary.json"

        if not summary_path.exists():
            continue

        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            summary = json.load(file)

        rows.append(
            {
                "dataset": summary["dataset"],
                "alpha": summary["alpha"],
                "lambda": summary["lambda"],
                "seed": summary["seed"],
                "best_epoch": summary["best_epoch"],
                "validation_combined":
                    summary["validation_combined"],
                "validation_regret":
                    summary["validation_regret"],
                "validation_mse":
                    summary["validation_mse"],
                "test_combined":
                    summary["test_combined"],
                "test_regret":
                    summary["test_regret"],
                "test_mse":
                    summary["test_mse"],
                "annualized_return_cagr":
                    summary["annualized_return_cagr"],
                "annualized_return_mean":
                    summary["annualized_return_mean"],
                "annualized_volatility":
                    summary["annualized_volatility"],
                "sharpe_ratio":
                    summary["sharpe_ratio"],
                "maximum_drawdown":
                    summary["maximum_drawdown"],
                "final_wealth":
                    summary["final_wealth"],
                "average_active_assets":
                    summary["average_active_assets"],
            }
        )

    if not rows:
        raise RuntimeError(
            "요약할 summary.json이 없습니다."
        )

    rows.sort(
        key=lambda row: row["alpha"]
    )

    summary_directory = (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "combined"
        / dataset_name
    )

    summary_path = (
        summary_directory
        / (
            f"alpha_grid_"
            f"lambda_{args.risk_aversion:.2f}_"
            f"seed_{args.seed}.csv"
        )
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    return summary_path


def main() -> None:
    args = parse_args()

    for alpha in args.alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"alpha는 0 이상 1 이하여야 합니다: {alpha}"
            )

    dataset_name = get_dataset_name(
        args.price_csv
    )

    failed_alphas: list[float] = []

    for alpha in args.alphas:
        success = run_experiment(
            args=args,
            alpha=alpha,
            dataset_name=dataset_name,
        )

        if not success:
            failed_alphas.append(alpha)

            if not args.continue_on_error:
                raise RuntimeError(
                    f"alpha={alpha:.2f} 실험에 실패했습니다."
                )

    summary_path = create_grid_summary(
        args=args,
        dataset_name=dataset_name,
    )

    print("=" * 80)
    print("Alpha grid completed")
    print("Summary:", summary_path)

    if failed_alphas:
        print(
            "Failed alphas:",
            failed_alphas,
        )
    else:
        print("All experiments completed successfully.")

    print("=" * 80)


if __name__ == "__main__":
    main()
