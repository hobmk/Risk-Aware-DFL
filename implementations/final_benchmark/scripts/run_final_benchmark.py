from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MODELS = [
    "dfl_mvo",
    "ridge",
    "ledoit_wolf",
    "rank_tail",
    "topk",
]

LAMBDAS = [1.0, 25.0]

SEEDS = list(
    range(42, 52)
)

OUTPUT_ROOT = Path(
    "implementations/"
    "final_benchmark/"
    "outputs"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
    )

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=LAMBDAS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def output_dir(
    model: str,
    risk_aversion: float,
    seed: int,
) -> Path:

    return (
        OUTPUT_ROOT
        / model
        / f"lambda_{risk_aversion:g}"
        / f"seed_{seed}"
    )


def completed_file(
    model: str,
    run_dir: Path,
) -> Path:

    if model == "dfl_mvo":
        return (
            run_dir
            / "summary.json"
        )

    return (
        run_dir
        / "portfolio_summary.csv"
    )


def build_command(
    model: str,
    risk_aversion: float,
    seed: int,
    device: str,
    run_dir: Path,
    overwrite: bool,
) -> list[str]:

    common = [
        "--risk-aversion",
        str(risk_aversion),

        "--seed",
        str(seed),

        "--device",
        device,

        "--epochs",
        "200",

        "--patience",
        "20",

        "--output-dir",
        str(run_dir),
    ]

    if model == "dfl_mvo":
        command = [
            sys.executable,
            "-m",
            (
                "implementations."
                "dfl_mvo_lee2025."
                "scripts."
                "train_mlp_markowitz_combined"
            ),

            "--hidden-dim",
            "64",

            "--dropout",
            "0",

            "--max-weight",
            "0.2",

            "--alpha",
            "0.5",

            "--mse-scale",
            "30",

            "--learning-rate",
            "0.0001",

            "--weight-decay",
            "0.01",

            "--batch-size",
            "16",

            "--standardize-inputs",

            "--diagnostics-every",
            "0",

            *common,
        ]

    elif model == "ridge":
        command = [
            sys.executable,
            "-m",
            (
                "implementations."
                "covariance_baselines."
                "scripts."
                "train_covariance_baseline"
            ),

            "--method",
            "ridge",

            "--ridge-shrinkage",
            "0.10",

            *common,
        ]

    elif model == "ledoit_wolf":
        command = [
            sys.executable,
            "-m",
            (
                "implementations."
                "covariance_baselines."
                "scripts."
                "train_covariance_baseline"
            ),

            "--method",
            "ledoit_wolf",

            *common,
        ]

    elif model == "rank_tail":
        command = [
            sys.executable,
            "-m",
            (
                "implementations."
                "rank_tail_rcr_dfl."
                "scripts."
                "train_rank_tail_rcr"
            ),

            "--tail-q",
            "0.35",

            "--tail-side",
            "down",

            *common,
        ]

    elif model == "topk":
        command = [
            sys.executable,
            "-m",
            (
                "implementations."
                "eigen_rcr_dfl."
                "scripts."
                "train_eigen_rcr"
            ),

            "--eigen-k",
            "1",

            "--eta",
            "0.10",

            "--scale-mode",
            "none",

            *common,
        ]

    else:
        raise ValueError(
            f"Unknown model: {model}"
        )

    if overwrite:
        command.append(
            "--overwrite"
        )

    return command


def run_one(
    model: str,
    risk_aversion: float,
    seed: int,
    device: str,
    overwrite: bool,
):

    run_dir = output_dir(
        model,
        risk_aversion,
        seed,
    )

    completed = completed_file(
        model,
        run_dir,
    )

    if (
        completed.exists()
        and not overwrite
    ):
        return (
            model,
            risk_aversion,
            seed,
            "SKIP",
        )

    command = build_command(
        model=model,
        risk_aversion=(
            risk_aversion
        ),
        seed=seed,
        device=device,
        run_dir=run_dir,
        overwrite=overwrite,
    )

    print(
        f"START | "
        f"{model:12s} | "
        f"lambda={risk_aversion:g} | "
        f"seed={seed}"
    )

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:
        return (
            model,
            risk_aversion,
            seed,
            f"ERROR({result.returncode})",
        )

    return (
        model,
        risk_aversion,
        seed,
        "DONE",
    )


def main():
    args = parse_args()

    jobs = [
        (
            model,
            risk_aversion,
            seed,
        )
        for model in args.models
        for risk_aversion in args.lambdas
        for seed in args.seeds
    ]

    print("=" * 80)
    print("FINAL BENCHMARK")
    print("=" * 80)

    print(
        "Models : "
        + ", ".join(
            args.models
        )
    )

    print(
        "Lambda : "
        + ", ".join(
            f"{x:g}"
            for x in args.lambdas
        )
    )

    print(
        "Seeds  : "
        + ", ".join(
            str(x)
            for x in args.seeds
        )
    )

    print(
        f"Jobs   : {len(jobs)}"
    )

    print(
        f"Workers: {args.workers}"
    )

    print("=" * 80)

    results = []

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = {
            executor.submit(
                run_one,
                model,
                risk_aversion,
                seed,
                args.device,
                args.overwrite,
            ): (
                model,
                risk_aversion,
                seed,
            )
            for (
                model,
                risk_aversion,
                seed,
            ) in jobs
        }

        for future in as_completed(
            futures
        ):
            (
                model,
                risk_aversion,
                seed,
                status,
            ) = future.result()

            results.append(
                (
                    model,
                    risk_aversion,
                    seed,
                    status,
                )
            )

            print(
                f"{status:10s} | "
                f"{model:12s} | "
                f"lambda={risk_aversion:g} | "
                f"seed={seed}"
            )

    errors = [
        row
        for row in results
        if row[3].startswith(
            "ERROR"
        )
    ]

    print("\n" + "=" * 80)

    print(
        f"Finished: "
        f"{len(results)} / "
        f"{len(jobs)}"
    )

    print(
        f"Errors  : "
        f"{len(errors)}"
    )

    if errors:
        print("\nFailed runs:")

        for row in errors:
            print(
                f"  {row}"
            )

    print("=" * 80)


if __name__ == "__main__":
    main()