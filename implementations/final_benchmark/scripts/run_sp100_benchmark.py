
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from implementations.final_benchmark.scripts import run_final_benchmark as baseline_runner


MODELS = ["dfl_mvo", "ridge", "ledoit_wolf", "rank_tail", "topk"]
LAMBDAS = [1.0, 25.0]
SEEDS = list(range(42, 52))

PRICE_CSV = Path("data/raw/sp100_adjusted_close.csv")
OUTPUT_ROOT = Path("implementations/final_benchmark/outputs/sp100")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "train", "evaluate", "stats"], default="all")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def validate_data():
    if not PRICE_CSV.is_file():
        raise FileNotFoundError(f"Data not found: {PRICE_CSV}")

    df = pd.read_csv(PRICE_CSV)
    if "Date" not in df.columns:
        raise ValueError("Missing Date column")

    dates = pd.to_datetime(df.pop("Date"))
    prices = df.to_numpy(dtype=np.float64)

    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Invalid or unsorted dates")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate ticker columns")

    if prices.shape[1] != 92:
        raise ValueError(f"Expected 92 assets, got {prices.shape[1]}")

    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Invalid price data")

    digest = hashlib.sha256(PRICE_CSV.read_bytes()).hexdigest()

    print(f"[DATA] {len(df)} days | {prices.shape[1]} assets")
    print(f"[DATE] {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}")
    return digest


def get_jobs():
    return [
        (model, lam, seed)
        for model in MODELS
        for lam in LAMBDAS
        for seed in SEEDS
    ]


def get_run_dir(model, lam, seed):
    return OUTPUT_ROOT / model / f"lambda_{lam:g}" / f"seed_{seed}"


def run_one(model, lam, seed):
    run_dir = get_run_dir(model, lam, seed)
    completed = baseline_runner.completed_file(model, run_dir)

    if completed.is_file():
        return model, lam, seed, "SKIP"

    if run_dir.exists():
        return model, lam, seed, "INCOMPLETE_DIR"

    command = baseline_runner.build_command(
        model=model,
        risk_aversion=lam,
        seed=seed,
        device=ARGS.device,
        run_dir=run_dir,
        overwrite=False,
    )

    command += ["--price-csv", str(PRICE_CSV)]
    command += [
        "--oracle-cache-dir",
        str(OUTPUT_ROOT / "cache" / model / "oracle"),
    ]

    if model != "topk":
        command += [
            "--risk-cache-dir",
            str(OUTPUT_ROOT / "cache" / model / "risk"),
        ]

    print(f"START | {model} | lambda={lam:g} | seed={seed}", flush=True)

    result = subprocess.run(command, check=False)

    status = (
        "DONE"
        if result.returncode == 0 and completed.is_file()
        else f"ERROR({result.returncode})"
    )

    return model, lam, seed, status


def train():
    jobs = get_jobs()
    results = []

    print(f"[TRAIN] Total jobs: {len(jobs)}")
    print(f"[TRAIN] Workers: {ARGS.workers}")

    with ThreadPoolExecutor(max_workers=ARGS.workers) as pool:
        futures = {
            pool.submit(run_one, *job): job
            for job in jobs
        }

        for future in as_completed(futures):
            model, lam, seed, status = future.result()
            results.append((model, lam, seed, status))

            print(
                f"{status:14s} | {model:12s} | "
                f"lambda={lam:g} | seed={seed}",
                flush=True,
            )

    df = pd.DataFrame(
        results,
        columns=["model", "lambda", "seed", "status"],
    )

    df.to_csv(OUTPUT_ROOT / "execution_status.csv", index=False)

    failed = df[~df["status"].isin(["DONE", "SKIP"])]

    if not failed.empty:
        raise RuntimeError(
            f"{len(failed)} experiments failed or incomplete. "
            "Check execution_status.csv"
        )


def evaluate():
    from implementations.final_benchmark.scripts import (
        evaluate_final_benchmark as evaluator,
    )

    evaluator.ROOT = OUTPUT_ROOT
    evaluator.OUTPUT_DIR = OUTPUT_ROOT / "common_evaluation"
    evaluator.MODELS = MODELS
    evaluator.LAMBDAS = LAMBDAS
    evaluator.SEEDS = SEEDS

    print("\n[EVALUATION] Starting common evaluation")
    evaluator.main()


def statistical_tests():
    from implementations.final_benchmark.scripts import (
        statistical_tests as stats,
    )

    stats.INPUT = (
        OUTPUT_ROOT / "common_evaluation" / "all_runs_common_metrics.csv"
    )
    stats.OUTPUT_DIR = OUTPUT_ROOT / "statistics"
    stats.MODELS = MODELS
    stats.LAMBDAS = LAMBDAS

    print("\n[STATISTICS] Friedman + Wilcoxon-Holm")
    stats.main()


def main():
    global ARGS
    ARGS = parse_args()

    if ARGS.workers < 1:
        raise ValueError("--workers must be >= 1")

    if OUTPUT_ROOT.resolve() == baseline_runner.OUTPUT_ROOT.resolve():
        raise RuntimeError("Refusing to overwrite Dow30 results")

    print("=" * 70)
    print("S&P100 BENCHMARK")
    print("=" * 70)
    print(f"Models : {MODELS}")
    print(f"Lambdas: {LAMBDAS}")
    print(f"Seeds  : {SEEDS}")
    print(f"Jobs   : {len(get_jobs())}")
    print(f"Output : {OUTPUT_ROOT}")

    if ARGS.stage in {"all", "train"}:
        digest = validate_data()

        if ARGS.dry_run:
            print("\n[DRY-RUN] Configuration OK. No training started.")
            return

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        manifest = OUTPUT_ROOT / "dataset_manifest.json"

        if manifest.exists():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old["sha256"] != digest:
                raise RuntimeError(
                    "Dataset changed since the previous run. "
                    "Do not mix results from different datasets."
                )
        else:
            manifest.write_text(
                json.dumps(
                    {
                        "price_csv": str(PRICE_CSV),
                        "sha256": digest,
                        "n_assets": 92,
                        "models": MODELS,
                        "lambdas": LAMBDAS,
                        "seeds": SEEDS,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        train()

    if ARGS.stage in {"all", "evaluate"}:
        evaluate()

    if ARGS.stage in {"all", "stats"}:
        statistical_tests()

    print("\n[DONE] S&P100 Benchmark completed")


if __name__ == "__main__":
    main()
