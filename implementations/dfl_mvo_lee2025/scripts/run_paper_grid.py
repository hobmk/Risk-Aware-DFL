from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="paper_main")
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--risk-aversions", nargs="+", type=float, default=[0.1, 0.5, 1, 5])
    parser.add_argument("--max-weights", nargs="+", type=float, default=[1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mse-scale", type=float, default=15.0)
    parser.add_argument("--active-threshold", type=float, default=1e-3)
    parser.add_argument("--cap-tolerance", type=float, default=1e-4)
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--validation-end", default="2022-12-31")
    parser.add_argument("--diagnostics-every", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in name.strip()
    )
    return cleaned or "experiment"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=True)


def write_status(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiment_name = sanitize_name(args.experiment_name)
    price_path = Path(args.price_csv)
    dataset_name = price_path.stem.replace("_adjusted_close", "")

    experiment_root = (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "combined"
        / dataset_name
        / experiment_name
    )
    logs_dir = experiment_root / "_grid_logs"
    status_path = experiment_root / "grid_status.csv"

    combinations = list(
        itertools.product(
            args.alphas,
            args.risk_aversions,
            args.max_weights,
            args.seeds,
        )
    )

    grid_config = {
        "experiment_name": experiment_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(combinations),
        "arguments": vars(args),
    }
    save_json(experiment_root / "grid_config.json", grid_config)

    print(f"Experiment: {experiment_name}")
    print(f"Total runs: {len(combinations)}")
    print(f"Output root: {experiment_root}")

    status_rows: list[dict] = []

    for run_index, (alpha, risk_aversion, max_weight, seed) in enumerate(
        combinations,
        start=1,
    ):
        output_dir = (
            experiment_root
            / f"alpha_{alpha:.2f}"
            / f"lambda_{risk_aversion:.2f}"
            / f"maxw_{max_weight:.2f}"
            / f"seed_{seed}"
        )
        summary_path = output_dir / "summary.json"
        run_name = (
            f"a{alpha:.2f}_l{risk_aversion:.2f}_"
            f"w{max_weight:.2f}_s{seed}"
        )
        log_path = logs_dir / f"{run_name}.log"

        if summary_path.exists() and not args.overwrite:
            print(f"[{run_index}/{len(combinations)}] SKIP {run_name}")
            status_rows.append(
                {
                    "run": run_name,
                    "alpha": alpha,
                    "risk_aversion": risk_aversion,
                    "max_weight": max_weight,
                    "seed": seed,
                    "status": "skipped_existing",
                    "return_code": 0,
                    "summary_path": str(summary_path),
                    "log_path": str(log_path),
                }
            )
            write_status(status_path, status_rows)
            continue

        command = [
            sys.executable,
            "-m",
            "implementations.dfl_mvo_lee2025.scripts.train_mlp_markowitz_combined",
            "--experiment-name",
            experiment_name,
            "--price-csv",
            args.price_csv,
            "--alpha",
            str(alpha),
            "--risk-aversion",
            str(risk_aversion),
            "--max-weight",
            str(max_weight),
            "--seed",
            str(seed),
            "--lookback",
            str(args.lookback),
            "--hidden-dim",
            str(args.hidden_dim),
            "--dropout",
            str(args.dropout),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--mse-scale",
            str(args.mse_scale),
            "--active-threshold",
            str(args.active_threshold),
            "--cap-tolerance",
            str(args.cap_tolerance),
            "--train-end",
            args.train_end,
            "--validation-end",
            args.validation_end,
            "--diagnostics-every",
            str(args.diagnostics_every),
        ]

        if args.overwrite:
            command.append("--overwrite")

        print(f"[{run_index}/{len(combinations)}] RUN  {run_name}")

        if args.dry_run:
            print(" ".join(command))
            status = "dry_run"
            return_code = 0
        else:
            logs_dir.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            return_code = process.returncode
            status = "completed" if return_code == 0 else "failed"

        status_rows.append(
            {
                "run": run_name,
                "alpha": alpha,
                "risk_aversion": risk_aversion,
                "max_weight": max_weight,
                "seed": seed,
                "status": status,
                "return_code": return_code,
                "summary_path": str(summary_path),
                "log_path": str(log_path),
            }
        )
        write_status(status_path, status_rows)

        if return_code != 0:
            print(f"FAILED: {run_name} | log={log_path}")
            if args.stop_on_error:
                raise SystemExit(return_code)

    completed = sum(row["status"] == "completed" for row in status_rows)
    failed = sum(row["status"] == "failed" for row in status_rows)
    skipped = sum(row["status"] == "skipped_existing" for row in status_rows)

    print("=" * 80)
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Status: {status_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
