from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SEEDS = [42, 43, 44, 45, 46]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_one(
    seed: int,
    device: str,
    overwrite: bool,
):
    output_dir = Path(
        "implementations/covariance_baselines/outputs/"
        f"ledoit_wolf/lambda_1/seed_{seed}"
    )

    completed = (
        output_dir
        / "portfolio_summary.csv"
    )

    if completed.exists() and not overwrite:
        return (
            seed,
            "SKIP",
        )

    command = [
        sys.executable,
        "-m",
        "implementations.covariance_baselines.scripts."
        "train_covariance_baseline",

        "--method",
        "ledoit_wolf",

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
        f"START | seed={seed}"
    )

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:
        return (
            seed,
            f"ERROR({result.returncode})",
        )

    return (
        seed,
        "DONE",
    )


def main():
    args = parse_args()

    print(
        f"Total jobs: {len(SEEDS)}"
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
                seed,
                args.device,
                args.overwrite,
            ): seed
            for seed in SEEDS
        }

        for future in as_completed(
            futures
        ):
            seed, status = (
                future.result()
            )

            print(
                f"{status:10s} | "
                f"seed={seed}"
            )


if __name__ == "__main__":
    main()