from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
    )
    parser.add_argument(
        "--dropouts",
        type=float,
        nargs="+",
        default=[0.0],
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-4],
    )
    parser.add_argument(
        "--weight-decays",
        type=float,
        nargs="+",
        default=[1e-2],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
    )
    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    return parser.parse_args()


def dataset_name_from_path(
    price_csv: str,
) -> str:
    return Path(price_csv).stem.replace(
        "_adjusted_close",
        "",
    )


def run_directory(
    dataset_name: str,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Path:
    return (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "tuning_mse"
        / dataset_name
        / f"hidden_{hidden_dim}"
        / f"dropout_{dropout:.2f}"
        / f"lr_{learning_rate:g}"
        / f"wd_{weight_decay:g}"
        / f"seed_{seed}"
    )


def run_one(
    args: argparse.Namespace,
    dataset_name: str,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> bool:
    output_dir = run_directory(
        dataset_name=dataset_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )

    summary_path = output_dir / "summary.json"

    if (
        args.skip_existing
        and summary_path.exists()
    ):
        print(
            "[SKIP] "
            f"hidden={hidden_dim}, "
            f"dropout={dropout}, "
            f"lr={learning_rate:g}, "
            f"wd={weight_decay:g}, "
            f"seed={seed}"
        )
        return True

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = output_dir / "run.log"

    command = [
        sys.executable,
        "-u",
        "-m",
        "implementations.dfl_mvo_lee2025.scripts."
        "train_mse",
        "--price-csv",
        args.price_csv,
        "--hidden-dim",
        str(hidden_dim),
        "--dropout",
        str(dropout),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
    ]

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"

    print("=" * 80)
    print(
        f"START hidden={hidden_dim}, "
        f"dropout={dropout}, "
        f"lr={learning_rate:g}, "
        f"wd={weight_decay:g}, "
        f"seed={seed}"
    )
    print("Log:", log_path)
    print("=" * 80)

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )

        if process.stdout is None:
            raise RuntimeError(
                "학습 출력을 읽을 수 없습니다."
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
            "[FAILED] "
            f"hidden={hidden_dim}, "
            f"dropout={dropout}, "
            f"lr={learning_rate:g}, "
            f"wd={weight_decay:g}, "
            f"seed={seed}"
        )
        return False

    return summary_path.exists()


def collect_results(
    args: argparse.Namespace,
    dataset_name: str,
) -> Path:
    rows = []

    combinations = itertools.product(
        args.hidden_dims,
        args.dropouts,
        args.learning_rates,
        args.weight_decays,
        args.seeds,
    )

    for (
        hidden_dim,
        dropout,
        learning_rate,
        weight_decay,
        seed,
    ) in combinations:
        summary_path = (
            run_directory(
                dataset_name=dataset_name,
                hidden_dim=hidden_dim,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=seed,
            )
            / "summary.json"
        )

        if not summary_path.exists():
            continue

        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            rows.append(
                json.load(file)
            )

    if not rows:
        raise RuntimeError(
            "수집할 tuning 결과가 없습니다."
        )

    rows.sort(
        key=lambda row: (
            row["best_validation_mse"],
            row["generalization_gap"],
        )
    )

    result_dir = (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "tuning_mse"
        / dataset_name
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    result_path = (
        result_dir
        / f"tuning_summary_{timestamp}.csv"
    )

    fieldnames = [
        "dataset",
        "hidden_dim",
        "dropout",
        "learning_rate",
        "weight_decay",
        "seed",
        "parameter_count",
        "best_epoch",
        "best_train_mse",
        "best_validation_mse",
        "generalization_gap",
        "test_mse",
    ]

    with result_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return result_path


def main() -> None:
    args = parse_args()

    dataset_name = dataset_name_from_path(
        args.price_csv
    )

    combinations = list(
        itertools.product(
            args.hidden_dims,
            args.dropouts,
            args.learning_rates,
            args.weight_decays,
            args.seeds,
        )
    )

    print(
        f"Total experiments: "
        f"{len(combinations)}"
    )

    failures = []

    for combination in combinations:
        (
            hidden_dim,
            dropout,
            learning_rate,
            weight_decay,
            seed,
        ) = combination

        success = run_one(
            args=args,
            dataset_name=dataset_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )

        if not success:
            failures.append(combination)

            if not args.continue_on_error:
                raise RuntimeError(
                    f"실험 실패: {combination}"
                )

    result_path = collect_results(
        args=args,
        dataset_name=dataset_name,
    )

    print("=" * 80)
    print("Tuning grid completed")
    print("Summary:", result_path)

    if failures:
        print("Failures:", failures)
    else:
        print("All experiments completed.")

    print("=" * 80)


if __name__ == "__main__":
    main()
