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
        description="Parallel Top-K Eigen RCR lambda x seed runner"
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

    parser.add_argument("--eigen-k", type=int, default=5)

    parser.add_argument(
        "--scale-mode",
        choices=[
            "none",
            "preserve_residual_trace",
            "match_base_trace",
        ],
        default="preserve_residual_trace",
    )

    parser.add_argument("--eta", type=float, default=0.75)

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mse-scale", type=float, default=30.0)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--max-weight", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument(
        "--output-root",
        default=(
            "implementations/"
            "eigen_rcr_dfl/"
            "outputs/topk_high_lambda"
        ),
    )

    parser.add_argument(
        "--oracle-cache-dir",
        default=(
            "implementations/"
            "eigen_rcr_dfl/"
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
            "log": None,
        }

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
        / f"lambda_{lam_tag}_seed_{seed}.log"
    )

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
        str(args.eigen_k),

        "--scale-mode",
        args.scale_mode,

        "--eta",
        str(args.eta),

        "--risk-aversion",
        str(risk_aversion),

        "--seed",
        str(seed),

        "--device",
        args.device,

        "--hidden-dim",
        str(args.hidden_dim),

        "--dropout",
        str(args.dropout),

        "--alpha",
        str(args.alpha),

        "--mse-scale",
        str(args.mse_scale),

        "--max-weight",
        str(args.max_weight),

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

        "--oracle-cache-dir",
        args.oracle_cache_dir,

        "--output-dir",
        str(output_dir),
    ]

    # 미완료 폴더가 남았을 경우만 재실행
    if output_dir.exists():
        command.append(
            "--overwrite"
        )

    env = os.environ.copy()

    # 4개 subprocess가 각각 다시 여러 CPU thread를
    # 생성하지 않도록 제한
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
        "log": str(log_path),
    }


def run_jobs(
    jobs,
    args,
):
    results = []

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
                result = {
                    "status": "FAILED",
                    "lambda": risk_aversion,
                    "seed": seed,
                    "seconds": 0.0,
                    "log": None,
                }

                print(
                    f"FAILED | "
                    f"lambda={risk_aversion:g} "
                    f"| seed={seed} "
                    f"| {exc}"
                )

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

    return results


def main():
    args = parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers는 1 이상이어야 합니다."
        )

    if not args.seeds:
        raise ValueError(
            "--seeds가 비어 있습니다."
        )

    print("=" * 72)
    print("TOP-K EIGEN RCR PARALLEL RUNNER")
    print("=" * 72)

    print(f"K       : {args.eigen_k}")
    print(f"Eta     : {args.eta}")
    print(f"Scale   : {args.scale_mode}")

    print(f"Lambdas : {args.lambdas}")
    print(f"Seeds   : {args.seeds}")

    print(f"Workers : {args.workers}")
    print(f"Device  : {args.device}")

    print(
        f"Jobs    : "
        f"{len(args.lambdas) * len(args.seeds)}"
    )

    print(f"Output  : {args.output_root}")

    print("=" * 72)

    start_all = time.perf_counter()

    # ========================================================
    # Phase 1
    #
    # 첫 번째 seed를 λ별로 먼저 실행한다.
    #
    # λ=25 / 50 / 100 / 200은 서로 다른 Oracle cache이므로
    # 4개를 동시에 실행해도 cache 충돌이 없다.
    #
    # 이 단계가 끝난 후에는 각 λ의 Oracle cache가 생성되어
    # 나머지 seed가 모두 HIT하게 된다.
    # ========================================================

    first_seed = args.seeds[0]

    warmup_jobs = [
        (
            risk_aversion,
            first_seed,
        )
        for risk_aversion
        in args.lambdas
    ]

    print()
    print("=" * 72)
    print(
        f"PHASE 1 | "
        f"Oracle warm-up + seed {first_seed}"
    )
    print("=" * 72)

    results = run_jobs(
        warmup_jobs,
        args,
    )

    failed_warmup = [
        result
        for result in results
        if result["status"] == "FAILED"
    ]

    if failed_warmup:
        raise RuntimeError(
            "Phase 1에서 실패했습니다. "
            "Oracle cache가 완성되지 않았으므로 "
            "나머지 seed는 실행하지 않습니다."
        )

    # ========================================================
    # Phase 2
    # 나머지 seed 병렬 실행
    # ========================================================

    remaining_seeds = (
        args.seeds[1:]
    )

    remaining_jobs = [
        (
            risk_aversion,
            seed,
        )
        for seed
        in remaining_seeds
        for risk_aversion
        in args.lambdas
    ]

    if remaining_jobs:

        print()
        print("=" * 72)
        print(
            "PHASE 2 | "
            "Remaining seeds"
        )
        print("=" * 72)

        results.extend(
            run_jobs(
                remaining_jobs,
                args,
            )
        )

    elapsed_all = (
        time.perf_counter()
        - start_all
    )

    n_expected = (
        len(args.lambdas)
        * len(args.seeds)
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

    print(
        f"Expected : "
        f"{n_expected}"
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