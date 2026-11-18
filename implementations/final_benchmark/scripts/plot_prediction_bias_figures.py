from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["dfl_mvo", "ridge", "ledoit_wolf", "rank_tail", "topk"]
MODEL_LABELS = {
    "dfl_mvo": "DFL-MVO",
    "ridge": "Ridge",
    "ledoit_wolf": "Ledoit-Wolf",
    "rank_tail": "Rank-Tail",
    "topk": "Top-K",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="implementations/final_benchmark/outputs")
    parser.add_argument("--bias-dir", default="implementations/final_benchmark/outputs/prediction_bias")
    parser.add_argument("--output-dir", default="implementations/final_benchmark/outputs/prediction_bias/figures")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    parser.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 25.0])
    parser.add_argument("--active-threshold", type=float, default=1e-3)
    return parser.parse_args()

def read_wide(path, value_name):
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Missing date column: {path}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.melt(id_vars="date", var_name="ticker", value_name=value_name)

def load_run(root: Path, model: str, lam: float, seed: int):
    run = root / model / f"lambda_{lam:g}" / f"seed_{seed}"

    if model == "dfl_mvo":
        df = pd.read_csv(run / "asset_predictions.csv")
        need = ["date", "ticker", "predicted_return", "true_return", "predicted_weight"]
        missing = set(need) - set(df.columns)
        if missing:
            raise ValueError(f"{run}: missing {sorted(missing)}")
        df = df[need].copy()
    else:
        pred = read_wide(run / "predicted_returns.csv", "predicted_return")
        true = read_wide(run / "true_returns.csv", "true_return")
        wgt = read_wide(run / "portfolio_weights.csv", "predicted_weight")
        df = pred.merge(true, on=["date", "ticker"], how="inner", validate="one_to_one")
        df = df.merge(wgt, on=["date", "ticker"], how="inner", validate="one_to_one")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)
    df["bias"] = df["predicted_return"] - df["true_return"]
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)

def build_full_panel(root: Path, lambdas, seeds, threshold):
    rows = []

    for lam in lambdas:
        for model in MODELS:
            for seed in seeds:
                df = load_run(root, model, lam, seed)
                df["lambda"] = lam
                df["model"] = model
                df["seed"] = seed
                df["direction"] = np.where(df["predicted_return"] > 0, "Up", "Down")
                df["membership"] = np.where(df["predicted_weight"] > threshold, "In", "Out")
                rows.append(df)

                print(f"LOAD | lambda={lam:g} | {model:12s} | seed={seed}")

    panel = pd.concat(rows, ignore_index=True)
    return panel

def plot_updown_bias(panel: pd.DataFrame, lam: float, out_dir: Path):
    df = panel[panel["lambda"] == lam].copy()

    seed_stats = (
        df.groupby(["model", "seed", "direction"], as_index=False)["bias"]
        .mean()
    )

    summary = (
        seed_stats.groupby(["model", "direction"], as_index=False)
        .agg(mean_bias=("bias", "mean"), std_bias=("bias", "std"))
    )

    x = np.arange(len(MODELS))
    width = 0.35

    up = summary[summary["direction"] == "Up"].set_index("model").reindex(MODELS)
    down = summary[summary["direction"] == "Down"].set_index("model").reindex(MODELS)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, up["mean_bias"], width, yerr=up["std_bias"], capsize=4, label="Up")
    ax.bar(x + width / 2, down["mean_bias"], width, yerr=down["std_bias"], capsize=4, label="Down")

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15)
    ax.set_ylabel("Prediction Bias")
    ax.set_title(f"Up/Down Prediction Bias (lambda={lam:g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"updown_bias_lambda_{lam:g}.png", dpi=200)
    plt.close(fig)

def plot_inout_distribution(panel: pd.DataFrame, lam: float, out_dir: Path):
    df = panel[panel["lambda"] == lam].copy()

    fig, ax = plt.subplots(figsize=(13, 5))

    positions = []
    data = []
    labels = []
    pos = 1

    for model in MODELS:
        for group in ["In", "Out"]:
            subset = df[(df["model"] == model) & (df["membership"] == group)]["predicted_return"].to_numpy()
            data.append(subset)
            positions.append(pos)
            labels.append(f"{MODEL_LABELS[model]}\n{group}")
            pos += 1
        pos += 0.6

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=False,
        showfliers=False,
        whis=(5, 95),
    )

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Predicted Return")
    ax.set_title(f"In/Out Predicted Return Distribution (lambda={lam:g})")
    fig.tight_layout()
    fig.savefig(out_dir / f"inout_predicted_return_lambda_{lam:g}.png", dpi=200)
    plt.close(fig)

def plot_inout_gap(panel: pd.DataFrame, lam: float, out_dir: Path):
    df = panel[panel["lambda"] == lam].copy()

    seed_means = (
        df.groupby(["model", "seed", "membership"], as_index=False)["predicted_return"]
        .mean()
    )

    pivot = seed_means.pivot(
        index=["model", "seed"],
        columns="membership",
        values="predicted_return",
    ).reset_index()

    pivot["gap_in_minus_out"] = pivot["In"] - pivot["Out"]

    summary = (
        pivot.groupby("model", as_index=False)
        .agg(mean_gap=("gap_in_minus_out", "mean"), std_gap=("gap_in_minus_out", "std"))
    ).set_index("model").reindex(MODELS)

    x = np.arange(len(MODELS))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, summary["mean_gap"], yerr=summary["std_gap"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15)
    ax.set_ylabel("Mean Predicted Return Gap (In - Out)")
    ax.set_title(f"In-Out Predicted Return Gap (lambda={lam:g})")
    fig.tight_layout()
    fig.savefig(out_dir / f"inout_gap_lambda_{lam:g}.png", dpi=200)
    plt.close(fig)

    pivot.to_csv(out_dir / f"inout_gap_by_seed_lambda_{lam:g}.csv", index=False)

def plot_weight_bias(panel: pd.DataFrame, lam: float, out_dir: Path):
    df = panel[panel["lambda"] == lam].copy()

    # 각 모델/seed/date에서 predicted_weight 기준 5분위
    rows = []

    for (model, seed, date), group in df.groupby(["model", "seed", "date"]):
        group = group.sort_values("predicted_weight").copy()
        n = len(group)
        if n < 5:
            continue

        ranks = np.arange(1, n + 1)
        bins = pd.qcut(ranks, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        group["weight_quintile"] = bins.to_numpy()

        tmp = (
            group.groupby("weight_quintile", as_index=False)["bias"]
            .mean()
        )
        tmp["model"] = model
        tmp["seed"] = seed
        tmp["date"] = date
        rows.append(tmp)

    qdf = pd.concat(rows, ignore_index=True)

    summary = (
        qdf.groupby(["model", "weight_quintile"], as_index=False)
        .agg(mean_bias=("bias", "mean"), std_bias=("bias", "std"))
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    quintiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(len(quintiles))
    width = 0.16

    for i, model in enumerate(MODELS):
        sub = summary[summary["model"] == model].set_index("weight_quintile").reindex(quintiles)
        ax.bar(
            x + (i - 2) * width,
            sub["mean_bias"],
            width,
            yerr=sub["std_bias"],
            capsize=2,
            label=MODEL_LABELS[model],
        )

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(quintiles)
    ax.set_ylabel("Prediction Bias")
    ax.set_title(f"Prediction Bias by Weight Quintile (lambda={lam:g})")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"weight_quintile_bias_lambda_{lam:g}.png", dpi=200)
    plt.close(fig)

def main():
    args = parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = build_full_panel(
        root=root,
        lambdas=args.lambdas,
        seeds=args.seeds,
        threshold=args.active_threshold,
    )

    panel.to_csv(out_dir / "full_prediction_panel.csv", index=False)

    for lam in args.lambdas:
        plot_updown_bias(panel, lam, out_dir)
        plot_inout_distribution(panel, lam, out_dir)
        plot_inout_gap(panel, lam, out_dir)
        plot_weight_bias(panel, lam, out_dir)

    print("\nSaved figures to:")
    print(out_dir.resolve())

if __name__ == "__main__":
    main()