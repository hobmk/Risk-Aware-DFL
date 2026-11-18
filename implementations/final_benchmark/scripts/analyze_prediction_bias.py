
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["dfl_mvo", "ridge", "ledoit_wolf", "rank_tail", "topk"]
ROOT = Path("implementations/final_benchmark/outputs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=ROOT / "prediction_bias")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 62)))
    p.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 25.0])
    p.add_argument("--active-threshold", type=float, default=1e-3)
    return p.parse_args()


def read_wide(path, value_name):
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Missing date: {path}")
    if df["date"].duplicated().any():
        raise ValueError(f"Duplicate dates: {path}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.melt(id_vars="date", var_name="ticker", value_name=value_name)


def load_run(root, model, lam, seed):
    run = root / model / f"lambda_{lam:g}" / f"seed_{seed}"

    if model == "dfl_mvo":
        df = pd.read_csv(run / "asset_predictions.csv")
    else:
        files = {
            "predicted_return": "predicted_returns.csv",
            "true_return": "true_returns.csv",
            "predicted_weight": "portfolio_weights.csv",
        }

        frames = [read_wide(run / name, key) for key, name in files.items()]
        df = frames[0]

        for frame in frames[1:]:
            df = df.merge(frame, on=["date", "ticker"], how="outer", validate="one_to_one")

    required = ["date", "ticker", "predicted_return", "true_return", "predicted_weight"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing columns in {run}: {set(required) - set(df.columns)}")

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(f"Duplicate date/ticker in {run}")
    if not np.isfinite(df[required[2:]].to_numpy(dtype=float)).all():
        raise ValueError(f"Missing or nonfinite values in {run}")

    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def summarize_groups(df, lam, model, seed, threshold):
    df = df.copy()
    df["bias"] = df["predicted_return"] - df["true_return"]
    df["direction"] = np.where(df["predicted_return"] > 0, "Up", "Down")
    df["membership"] = np.where(df["predicted_weight"] > threshold, "In", "Out")

    specs = [
        ("All", pd.Series("All", index=df.index)),
        ("UpDown", df["direction"]),
        ("InOut", df["membership"]),
        ("Joint", df["direction"] + "_" + df["membership"]),
    ]

    rows = []
    for grouping, labels in specs:
        temp = df.assign(group=labels)
        for group, part in temp.groupby("group"):
            bias = part["bias"]
            rows.append({
                "lambda": lam,
                "model": model,
                "seed": seed,
                "grouping": grouping,
                "group": group,
                "n_assets_dates": len(part),
                "mean_bias": bias.mean(),
                "mae": bias.abs().mean(),
                "rmse": np.sqrt(np.mean(bias.to_numpy() ** 2)),
                "mean_predicted_return": part["predicted_return"].mean(),
                "mean_true_return": part["true_return"].mean(),
                "mean_predicted_weight": part["predicted_weight"].mean(),
            })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    results = []

    for lam in args.lambdas:
        reference_index = None
        reference_true = None

        for model in MODELS:
            for seed in args.seeds:
                df = load_run(args.root, model, lam, seed)
                index = pd.MultiIndex.from_frame(df[["date", "ticker"]])
                truth = df["true_return"].to_numpy(dtype=float)

                if reference_index is None:
                    reference_index = index
                    reference_true = truth
                else:
                    if not reference_index.equals(index):
                        raise ValueError(f"Date/ticker mismatch: {model}, lambda={lam}, seed={seed}")
                    if not np.allclose(reference_true, truth, rtol=1e-6, atol=1e-8):
                        raise ValueError(
                            f"True return mismatch: {model}, lambda={lam}, seed={seed}, "
                            f"max_diff={np.max(np.abs(reference_true - truth)):.3e}"
                        )
                results.append(summarize_groups(
                    df, lam, model, seed, args.active_threshold
                ))

            print(f"OK | lambda={lam:g} | {model} | {len(args.seeds)} seeds")

    runs = pd.concat(results, ignore_index=True)

    keys = ["lambda", "model", "grouping", "group"]
    summary = runs.groupby(keys, as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean_bias=("mean_bias", "mean"),
        std_bias=("mean_bias", "std"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        mean_group_size=("n_assets_dates", "mean"),
        mean_true_return=("mean_true_return", "mean"),
        mean_predicted_return=("mean_predicted_return", "mean"),
    )

    # Same-seed differences from DFL-MVO, avoiding independent-group comparisons.
    baseline = runs[runs["model"] == "dfl_mvo"][
        ["lambda", "seed", "grouping", "group", "mean_bias", "mae"]
    ].rename(columns={"mean_bias": "dfl_bias", "mae": "dfl_mae"})

    paired = runs[runs["model"] != "dfl_mvo"].merge(
        baseline, on=["lambda", "seed", "grouping", "group"], validate="many_to_one"
    )
    paired["delta_bias"] = paired["mean_bias"] - paired["dfl_bias"]
    paired["delta_mae"] = paired["mae"] - paired["dfl_mae"]

    paired_summary = paired.groupby(
        ["lambda", "model", "grouping", "group"], as_index=False
    ).agg(
        n_seeds=("seed", "nunique"),
        delta_bias_mean=("delta_bias", "mean"),
        delta_bias_std=("delta_bias", "std"),
        delta_mae_mean=("delta_mae", "mean"),
        delta_mae_std=("delta_mae", "std"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "bias_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "bias_summary.csv", index=False)
    paired.to_csv(args.output_dir / "bias_paired_seed.csv", index=False)
    paired_summary.to_csv(args.output_dir / "bias_paired_summary.csv", index=False)

    for lam in args.lambdas:
        print(f"\n{'=' * 90}\nLAMBDA = {lam:g}: UP/DOWN & IN/OUT BIAS\n{'=' * 90}")
        view = summary[
            (summary["lambda"] == lam) &
            (summary["grouping"].isin(["UpDown", "InOut"]))
        ][["model", "group", "mean_bias", "std_bias", "mean_mae", "mean_group_size"]]
        print(view.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print(f"\nSaved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
