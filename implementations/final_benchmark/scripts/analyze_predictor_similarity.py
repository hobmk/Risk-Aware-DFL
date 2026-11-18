
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["dfl_mvo", "ridge", "ledoit_wolf", "rank_tail", "topk"]
ROOT = Path("implementations/final_benchmark/outputs")
SEEDS = list(range(42, 62))
LAMBDAS = [1.0, 25.0]
METRICS = ["spearman", "top5_overlap", "top5_jaccard", "prediction_rmse"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=ROOT / "predictor_similarity")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def load_prediction(root, model, lam, seed):
    run = root / model / f"lambda_{lam:g}" / f"seed_{seed}"

    if model == "dfl_mvo":
        path = run / "asset_predictions.csv"
        df = pd.read_csv(path)
        required = {"date", "ticker", "predicted_return"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path}: missing {required - set(df.columns)}")
        if df.duplicated(["date", "ticker"]).any():
            raise ValueError(f"{path}: duplicate date/ticker")
        frame = df.pivot(index="date", columns="ticker", values="predicted_return")
    else:
        path = run / "predicted_returns.csv"
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError(f"{path}: date column missing")
        frame = df.set_index("date")

    frame.index = pd.to_datetime(frame.index).normalize()
    frame.columns = frame.columns.astype(str)

    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"{path}: duplicate dates or tickers")

    frame = frame.sort_index().sort_index(axis=1).astype(np.float64)

    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: empty or nonfinite prediction values")

    return frame


def align_strict(a, b, name):
    if not a.index.equals(b.index):
        raise ValueError(f"{name}: date mismatch")
    if not a.columns.equals(b.columns):
        raise ValueError(f"{name}: ticker mismatch")
    return a, b


def topk_indices(values, tickers, k):
    # Tie-break by ticker name for reproducibility.
    return set(np.lexsort((tickers, -values))[:k].tolist())


def compare_pair(a, b, lam, model_a, seed_a, model_b, seed_b, k):
    name = f"{model_a}:{seed_a} vs {model_b}:{seed_b} lambda={lam:g}"
    a, b = align_strict(a, b, name)

    tickers = a.columns.to_numpy()
    if len(tickers) < k:
        raise ValueError(f"{name}: only {len(tickers)} tickers")

    av = a.to_numpy()
    bv = b.to_numpy()
    rows = []

    for i, date in enumerate(a.index):
        x, y = av[i], bv[i]
        xrank = pd.Series(x).rank().to_numpy()
        yrank = pd.Series(y).rank().to_numpy()

        sx = np.std(xrank)
        sy = np.std(yrank)
        corr = (
            np.corrcoef(xrank, yrank)[0, 1]
            if sx > 0 and sy > 0 else np.nan
        )

        top_a = topk_indices(x, tickers, k)
        top_b = topk_indices(y, tickers, k)
        overlap = len(top_a & top_b)

        rows.append({
            "lambda": lam,
            "model_a": model_a,
            "seed_a": seed_a,
            "model_b": model_b,
            "seed_b": seed_b,
            "date": date,
            "n_assets": len(tickers),
            "spearman": corr,
            "top5_overlap": overlap / k,
            "top5_overlap_count": overlap,
            "top5_jaccard": overlap / (2 * k - overlap),
            "prediction_rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        })

    return pd.DataFrame(rows)


def summarize_by_pair(daily):
    keys = ["lambda", "model_a", "seed_a", "model_b", "seed_b"]
    return daily.groupby(keys, as_index=False).agg(
        n_dates=("date", "nunique"),
        n_assets=("n_assets", "min"),
        spearman=("spearman", "mean"),
        top5_overlap=("top5_overlap", "mean"),
        top5_jaccard=("top5_jaccard", "mean"),
        prediction_rmse=("prediction_rmse", "mean"),
    )


def group_name(row):
    if row.model_a == row.model_b == "dfl_mvo":
        return "DFL_seed_baseline"
    return f"{row.model_a}_vs_{row.model_b}"


def build_summary(pairs):
    rows = []

    for (lam, group), df in pairs.groupby(["lambda", "comparison"]):
        row = {"lambda": lam, "comparison": group, "n_pairs": len(df)}

        for metric in METRICS:
            values = df[metric].dropna()
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q05"] = values.quantile(0.05)
            row[f"{metric}_q95"] = values.quantile(0.95)

        rows.append(row)

    return pd.DataFrame(rows)


def baseline_comparison(pairs):
    rows = []

    for lam, block in pairs.groupby("lambda"):
        baseline = block[block.comparison == "DFL_seed_baseline"]

        if len(baseline) != 190 and len(SEEDS) == 20:
            raise ValueError(f"lambda={lam:g}: DFL baseline incomplete")

        for group, df in block.groupby("comparison"):
            if group == "DFL_seed_baseline":
                continue

            for metric in METRICS:
                base = baseline[metric].dropna().to_numpy()
                values = df[metric].dropna().to_numpy()
                if not len(base) or not len(values):
                    continue

                # Positive shift denotes greater dissimilarity.
                if metric == "prediction_rmse":
                    base_shift, model_shift = base, values
                else:
                    base_shift, model_shift = 1 - base, 1 - values

                rows.append({
                    "lambda": lam,
                    "comparison": group,
                    "metric": metric,
                    "baseline_dissimilarity_mean": base_shift.mean(),
                    "model_dissimilarity_mean": model_shift.mean(),
                    "excess_dissimilarity": model_shift.mean() - base_shift.mean(),
                    "baseline_q95": np.quantile(base_shift, 0.95),
                    "model_pairs_above_baseline_q95": int(
                        np.sum(model_shift > np.quantile(base_shift, 0.95))
                    ),
                    "n_model_pairs": len(model_shift),
                })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if args.top_k < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Invalid top-k or duplicate seeds")

    outputs = {}
    print("=" * 90)
    print("STEP 2: PREDICTOR SIMILARITY")
    print("=" * 90)

    for lam in args.lambdas:
        for model in MODELS:
            for seed in args.seeds:
                frame = load_prediction(args.root, model, lam, seed)
                outputs[(lam, model, seed)] = frame

        reference = outputs[(lam, "dfl_mvo", args.seeds[0])]
        for model in MODELS:
            for seed in args.seeds:
                align_strict(
                    reference, outputs[(lam, model, seed)],
                    f"{model} seed={seed}, lambda={lam:g}"
                )

        print(f"lambda={lam:g}: {len(MODELS) * len(args.seeds)} prediction files OK")
        print(f"  dates={len(reference)}, assets={len(reference.columns)}")
        print(f"  period={reference.index.min().date()} ~ {reference.index.max().date()}")

    daily_blocks = []

    for lam in args.lambdas:
        print(f"\nAnalyzing lambda={lam:g}...")

        # DFL initialization variation: all unordered seed pairs
        for sa, sb in combinations(args.seeds, 2):
            daily_blocks.append(compare_pair(
                outputs[(lam, "dfl_mvo", sa)],
                outputs[(lam, "dfl_mvo", sb)],
                lam, "dfl_mvo", sa, "dfl_mvo", sb, args.top_k
            ))

        # Paired model comparisons under identical seeds
        for model in MODELS[1:]:
            for seed in args.seeds:
                daily_blocks.append(compare_pair(
                    outputs[(lam, "dfl_mvo", seed)],
                    outputs[(lam, model, seed)],
                    lam, "dfl_mvo", seed, model, seed, args.top_k
                ))

        # Additional residual-risk model comparison
        for seed in args.seeds:
            daily_blocks.append(compare_pair(
                outputs[(lam, "rank_tail", seed)],
                outputs[(lam, "topk", seed)],
                lam, "rank_tail", seed, "topk", seed, args.top_k
            ))

    daily = pd.concat(daily_blocks, ignore_index=True)
    pairs = summarize_by_pair(daily)
    pairs["comparison"] = pairs.apply(group_name, axis=1)

    summary = build_summary(pairs)
    baseline = baseline_comparison(pairs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_dir / "daily_similarity.csv", index=False)
    pairs.to_csv(args.output_dir / "pair_similarity.csv", index=False)
    summary.to_csv(args.output_dir / "similarity_summary.csv", index=False)
    baseline.to_csv(args.output_dir / "baseline_comparison.csv", index=False)

    for lam in args.lambdas:
        print(f"\n{'=' * 90}")
        print(f"LAMBDA = {lam:g}")
        print("=" * 90)
        view = summary[summary["lambda"] == lam][[
            "comparison", "n_pairs",
            "spearman_mean", "top5_overlap_mean", "prediction_rmse_mean"
        ]]
        print(view.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        print("\nDIFFERENCE RELATIVE TO DFL SEED BASELINE")
        view = baseline[baseline["lambda"] == lam][[
            "comparison", "metric", "excess_dissimilarity",
            "model_pairs_above_baseline_q95", "n_model_pairs"
        ]]
        print(view.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print(f"\nSaved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
