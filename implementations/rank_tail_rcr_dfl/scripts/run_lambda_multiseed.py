from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel Rank-Tail lambda x seed runner"
        )
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

    parser.add_argument(
        "--tail-q",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--tail-side",
        choices=["down", "up"],
        default="down",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
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
        "--alpha",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--mse-scale",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--output-root",
        default=(
            "implementations/"
            "rank_tail_rcr_dfl/"
            "outputs/lambda_multiseed"
        ),
    )

    parser.add_argument(
        "--risk-cache-dir",
        default=(
            "implementations/"
            "rank_tail_rcr_dfl/"
            "cache/risk_matrix"
        ),
    )

    parser.add_argument(
        "--oracle-cache-dir",
        default=(
            "implementations/"
            "rank_tail_rcr_dfl/"
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
    lam_tag = lambda_tag(
        risk_aversion
    )

    output_dir = (
        Path(args.output_root)
        / f"lambda_{lam_tag}"
        / f"seed_{seed}"
    )

    completed_file = (
        output_dir
        / "portfolio_summary.csv"
    )

    if completed_file.exists():
        return {
            "status": "SKIP",
            "lambda": risk_aversion,
            "seed": seed,
            "seconds": 0.0,
            "output_dir": str(output_dir),
            "log": None,
        }

    # 이전 실행이 중간에 끊겨서
    # 결과 폴더만 남은 경우에만 제거한다.
    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    log_dir = (
        Path(args.output_root)
        / "_logs"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / (
            f"lambda_{lam_tag}"
            f"_seed_{seed}.log"
        )
    )

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
        str(args.tail_q),

        "--tail-side",
        args.tail_side,

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

        "--risk-cache-dir",
        args.risk_cache_dir,

        "--oracle-cache-dir",
        args.oracle_cache_dir,

        "--output-dir",
        str(output_dir),
    ]

    # 각 seed subprocess 안에서 BLAS/OpenMP가
    # 다시 여러 CPU thread를 생성하지 않도록 제한.
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

        log_file.write(
            "COMMAND\n"
        )

        log_file.write(
            " ".join(command)
            + "\n\n"
        )

        log_file.flush()

        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    if process.returncode == 0:
        status = "DONE"
    else:
        status = "FAILED"

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

    jobs = [
        (
            risk_aversion,
            seed,
        )
        for risk_aversion
        in args.lambdas
        for seed
        in args.seeds
    ]

    print("=" * 72)
    print("RANK-TAIL PARALLEL RUNNER")
    print("=" * 72)

    print(
        f"Lambdas : "
        f"{args.lambdas}"
    )

    print(
        f"Seeds   : "
        f"{args.seeds}"
    )

    print(
        f"Workers : "
        f"{args.workers}"
    )

    print(
        f"Device  : "
        f"{args.device}"
    )

    print(
        f"Jobs    : "
        f"{len(jobs)}"
    )

    print(
        f"Output  : "
        f"{args.output_root}"
    )

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
                risk_aversion=(
                    risk_aversion
                ),
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

                results.append({
                    "status": "FAILED",
                    "lambda": (
                        risk_aversion
                    ),
                    "seed": seed,
                    "seconds": 0.0,
                    "output_dir": None,
                    "log": None,
                })

                continue

            results.append(
                result
            )

            print(
                f"{result['status']} | "
                f"lambda="
                f"{result['lambda']:g} "
                f"| seed="
                f"{result['seed']} "
                f"| "
                f"{result['seconds']:.1f}s"
            )

            if (
                result["status"]
                == "FAILED"
            ):
                print(
                    f"  log: "
                    f"{result['log']}"
                )

    elapsed_all = (
        time.perf_counter()
        - start_all
    )

    n_done = sum(
        r["status"] == "DONE"
        for r in results
    )

    n_skip = sum(
        r["status"] == "SKIP"
        for r in results
    )

    n_failed = sum(
        r["status"] == "FAILED"
        for r in results
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"Expected : "
        f"{len(jobs)}"
    )

    print(
        f"Executed : "
        f"{n_done}"
    )

    print(
        f"Skipped  : "
        f"{n_skip}"
    )

    print(
        f"Failed   : "
        f"{n_failed}"
    )

    print(
        f"Elapsed  : "
        f"{elapsed_all:.1f} sec"
    )

    print("=" * 72)

    if n_failed > 0:
        raise RuntimeError(
            f"{n_failed} job(s) failed. "
            f"_logs 폴더를 확인하세요."
        )


if __name__ == "__main__":
    main()