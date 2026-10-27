from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


RIDGE_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40]
SEEDS = [42, 43, 44, 45, 46]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_one(
    rho: float,
    seed: int,
    device: str,
    overwrite: bool,
):
    output_dir = Path(
        "implementations/covariance_baselines/outputs/"
        f"ridge_validation/rho_{rho:.2f}/seed_{seed}"
    )

    completed = output_dir / "portfolio_summary.csv"

    if completed.exists() and not overwrite:
        return (
            rho,
            seed,
            "SKIP",
        )

    command = [
        sys.executable,
        "-m",
        "implementations.covariance_baselines.scripts.train_covariance_baseline",
        "--method",
        "ridge",
        "--ridge-shrinkage",
        str(rho),
        "--risk-aversion",
        "1",
        "--seed",
        str(seed),
        "--device",
        device,
        "--epochs",
        "200",
        "--patience",
        "20",
        "--output-dir",
        str(output_dir),
    ]

    if overwrite:
        command.append(
            "--overwrite"
        )

    print(
        f"START | rho={rho:.2f} | seed={seed}"
    )

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:
        return (
            rho,
            seed,
            f"ERROR({result.returncode})",
        )

    return (
        rho,
        seed,
        "DONE",
    )


def main():
    args = parse_args()

    jobs = [
        (rho, seed)
        for rho in RIDGE_VALUES
        for seed in SEEDS
    ]

    print(
        f"Total jobs: {len(jobs)}"
    )

    print(
        f"Workers   : {args.workers}"
    )

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = {
            executor.submit(
                run_one,
                rho,
                seed,
                args.device,
                args.overwrite,
            ): (
                rho,
                seed,
            )
            for rho, seed in jobs
        }

        for future in as_completed(
            futures
        ):
            rho, seed, status = (
                future.result()
            )

            print(
                f"{status:10s} | "
                f"rho={rho:.2f} | "
                f"seed={seed}"
            )


if __name__ == "__main__":
    main()