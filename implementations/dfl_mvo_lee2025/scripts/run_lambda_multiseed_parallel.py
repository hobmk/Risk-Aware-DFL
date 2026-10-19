from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel DFL-MVO lambda x seed runner"
    )

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[25, 50, 100, 200],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(42, 52)),
    )

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mse-scale", type=float, default=30.0)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--max-weight", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument(
        "--output-root",
        default=(
            "implementations/"
            "dfl_mvo_lee2025/"
            "outputs/lambda_multiseed_standardized"
        ),
    )

    parser.add_argument(
        "--risk-cache-dir",
        default=(
            "implementations/"
            "dfl_mvo_lee2025/"
            "cache/risk_matrix"
        ),
    )

    parser.add_argument(
        "--oracle-cache-dir",
        default=(
            "implementations/"
            "dfl_mvo_lee2025/"
            "cache/oracle"
        ),
    )

    return parser.parse_args()


def lambda_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))

    return str(value).replace(".", "p")


def run_one(
    *,
    risk_aversion: float,
    seed: int,
    args,
):
    lam_tag = lambda_tag(risk_aversion)

    output_dir = (
        Path(args.output_root)
        / f"lambda_{lam_tag}"
        / f"seed_{seed}"
    )

    completed_file = output_dir / "summary.json"

    # 완료된 결과는 보존하고 건너뜀.
    if completed_file.exists():
        return {
            "status": "SKIP",
            "lambda": risk_aversion,
            "seed": seed,
            "seconds": 0.0,
            "output_dir": str(output_dir),
            "log": None,
        }

    log_dir = Path(args.output_root) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        log_dir
        / f"lambda_{lam_tag}_seed_{seed}.log"
    )

    command = [
        sys.executable,
        "-m",
        (
            "implementations."
            "dfl_mvo_lee2025."
            "scripts."
            "train_mlp_markowitz_combined"
        ),

        "--risk-aversion",
        str(risk_aversion),

        "--seed",
        str(seed),

        "--device",
        args.device,

        "--epochs",
        str(args.epochs),

        "--patience",
        str(args.patience),

        "--alpha",
        str(args.alpha),

        "--mse-scale",
        str(args.mse_scale),

        "--hidden-dim",
        str(args.hidden_dim),

        "--dropout",
        str(args.dropout),

        "--max-weight",
        str(args.max_weight),

        "--batch-size",
        str(args.batch_size),

        "--learning-rate",
        str(args.learning_rate),

        "--weight-decay",
        str(args.weight_decay),

        "--standardize-inputs",

        "--diagnostics-every",
        "0",

        "--experiment-name",
        "lambda_multiseed_standardized",

        "--risk-cache-dir",
        args.risk_cache_dir,

        "--oracle-cache-dir",
        args.oracle_cache_dir,

        "--output-dir",
        str(output_dir),
    ]

    # 미완료 폴더가 남아 있는 경우
    # train script의 --overwrite로 정리 후 다시 수행.
    if output_dir.exists():
        command.append(
            "--overwrite"
        )

    # 여러 subprocess가 동시에 BLAS thread를 생성하면
    # oversubscription이 생길 수 있으므로
    # worker 내부 thread는 1개로 제한.
    env = os.environ.copy()

    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"

    start = time.perf_counter()

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log_file:

        log_file.write("COMMAND\n")
        log_file.write(
            " ".join(command) + "\n\n"
        )
        log_file.flush()

        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    elapsed = time.perf_counter() - start

    status = (
        "DONE"
        if process.returncode == 0
        else "FAILED"
    )

    return {
        "status": status,
        "lambda": risk_aversion,
        "seed": seed,
        "seconds": elapsed,
        "output_dir": str(output_dir),
        "log": str(log_path),
    }


def main():
    args = parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers는 1 이상이어야 합니다."
        )

    jobs = [
        (risk_aversion, seed)
        for risk_aversion in args.lambdas
        for seed in args.seeds
    ]

    print("=" * 72)
    print("DFL-MVO PARALLEL RUNNER")
    print("=" * 72)

    print(f"Lambdas : {args.lambdas}")
    print(f"Seeds   : {args.seeds}")
    print(f"Workers : {args.workers}")
    print(f"Device  : {args.device}")
    print(f"Jobs    : {len(jobs)}")
    print(f"Output  : {args.output_root}")

    print("=" * 72)

    results = []

    start_all = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        future_map = {}

        for risk_aversion, seed in jobs:

            print(
                f"QUEUE | "
                f"lambda={risk_aversion:g} "
                f"| seed={seed}"
            )

            future = executor.submit(
                run_one,
                risk_aversion=risk_aversion,
                seed=seed,
                args=args,
            )

            future_map[future] = (
                risk_aversion,
                seed,
            )

        for future in as_completed(
            future_map
        ):
            risk_aversion, seed = (
                future_map[future]
            )

            try:
                result = future.result()

            except Exception as exc:
                print(
                    f"FAILED | "
                    f"lambda={risk_aversion:g} "
                    f"| seed={seed} "
                    f"| {exc}"
                )

                results.append(
                    {
                        "status": "FAILED",
                        "lambda": risk_aversion,
                        "seed": seed,
                        "seconds": 0.0,
                        "output_dir": None,
                        "log": None,
                    }
                )

                continue

            results.append(result)

            print(
                f"{result['status']} | "
                f"lambda={result['lambda']:g} "
                f"| seed={result['seed']} "
                f"| {result['seconds']:.1f}s"
            )

            if result["status"] == "FAILED":
                print(
                    f"  log: "
                    f"{result['log']}"
                )

    elapsed_all = (
        time.perf_counter()
        - start_all
    )

    n_done = sum(
        result["status"] == "DONE"
        for result in results
    )

    n_skip = sum(
        result["status"] == "SKIP"
        for result in results
    )

    n_failed = sum(
        result["status"] == "FAILED"
        for result in results
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(f"Expected : {len(jobs)}")
    print(f"Executed : {n_done}")
    print(f"Skipped  : {n_skip}")
    print(f"Failed   : {n_failed}")
    print(f"Elapsed  : {elapsed_all:.1f} sec")

    print("=" * 72)

    if n_failed > 0:
        raise RuntimeError(
            f"{n_failed} job(s) failed. "
            f"_logs 폴더를 확인하세요."
        )


if __name__ == "__main__":
    main()