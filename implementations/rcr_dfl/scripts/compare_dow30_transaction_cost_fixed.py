from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_dow30_transaction_cost import (
    discover_runs,
    build_run_map,
    load_run_series,
    equal_weight_from_asset_predictions,
    apply_cost,
    metrics,
)


DFL_ROOT = (
    "implementations/dfl_mvo_lee2025/"
    "outputs/combined/dow30/"
    "final_full_grid_h64_d0_s30_std"
)

RCR_ROOT = (
    "implementations/rcr_dfl/"
    "outputs/grid25_multiseed"
)

OUTPUT_DIR = Path(
    "implementations/rcr_dfl/"
    "outputs/dow30_transaction_cost_fixed"
)

ALPHA = 0.5
LAMBDA = 1.0
MAX_WEIGHT = 0.2
SEEDS = [42, 43, 44]
COSTS_BPS = [0, 5, 10]

CFG = (
    round(ALPHA, 10),
    round(LAMBDA, 10),
    round(MAX_WEIGHT, 10),
)


def summarize(seed_results):
    rows = []

    for cost in COSTS_BPS:
        for model in ["Equal Weight", "DFL-MVO", "RCR-DFL"]:
            x = seed_results[
                (seed_results["cost_bps"] == cost)
                & (seed_results["model"] == model)
            ]

            if x.empty:
                continue

            row = {
                "cost_bps": cost,
                "model": model,
                "alpha": np.nan if model == "Equal Weight" else ALPHA,
                "lambda": np.nan if model == "Equal Weight" else LAMBDA,
                "max_weight": np.nan if model == "Equal Weight" else MAX_WEIGHT,
                "n_seeds": 1 if model == "Equal Weight" else x["seed"].nunique(),
            }

            for col in [
                "cagr",
                "volatility",
                "sharpe",
                "mdd",
                "cumulative_return",
                "turnover",
            ]:
                row[f"{col}_mean"] = x[col].mean()

                if model == "Equal Weight":
                    row[f"{col}_std"] = 0.0
                else:
                    row[f"{col}_std"] = x[col].std(ddof=1)

            rows.append(row)

    return pd.DataFrame(rows)


def mean_wealth(run_map, model, cost_bps):
    curves = []

    for seed in SEEDS:
        df = load_run_series(run_map[CFG][seed])
        net = apply_cost(df, cost_bps)

        curve = pd.Series(
            net["wealth"].to_numpy(),
            index=pd.to_datetime(net["date"]),
            name=f"{model}_{seed}",
        )

        curves.append(curve)

    curves = pd.concat(curves, axis=1, join="inner")

    return curves.mean(axis=1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("DOW30 FIXED-PARAMETER TRANSACTION COST COMPARISON")
    print("=" * 90)

    print(
        f"\nFixed configuration:"
        f"\n  alpha      = {ALPHA}"
        f"\n  lambda     = {LAMBDA}"
        f"\n  max_weight = {MAX_WEIGHT}"
        f"\n  seeds      = {SEEDS}"
    )

    print("\n[1] Discovering DFL-MVO runs...")
    dfl_runs = discover_runs(
        DFL_ROOT,
        "DFL-MVO",
        MAX_WEIGHT,
    )

    print(f"DFL-MVO runs found: {len(dfl_runs)}")

    print("\n[2] Discovering RCR-DFL runs...")
    rcr_runs = discover_runs(
        RCR_ROOT,
        "RCR-DFL",
        MAX_WEIGHT,
    )

    print(f"RCR-DFL runs found: {len(rcr_runs)}")

    dfl_map = build_run_map(dfl_runs)
    rcr_map = build_run_map(rcr_runs)

    if CFG not in dfl_map:
        raise RuntimeError(
            f"DFL-MVO configuration not found: {CFG}"
        )

    if CFG not in rcr_map:
        raise RuntimeError(
            f"RCR-DFL configuration not found: {CFG}"
        )

    for seed in SEEDS:
        if seed not in dfl_map[CFG]:
            raise RuntimeError(
                f"DFL-MVO seed {seed} missing for {CFG}"
            )

        if seed not in rcr_map[CFG]:
            raise RuntimeError(
                f"RCR-DFL seed {seed} missing for {CFG}"
            )

    print("\n[3] All paired seeds found.")
    print(f"Common seeds: {SEEDS}")

    print("\n[4] Building Equal Weight baseline...")

    ew_source = dfl_map[CFG][SEEDS[0]]["asset_path"]

    if ew_source is None:
        raise RuntimeError(
            "DFL asset_predictions.csv not found."
        )

    ew_df = equal_weight_from_asset_predictions(
        ew_source
    )

    print(f"EW dates: {len(ew_df)}")
    print(
        f"EW average daily turnover: "
        f"{ew_df['turnover'].mean():.6f}"
    )

    print("\n[5] Calculating 0 / 5 / 10 bp performance...")

    results = []

    for model, run_map in [
        ("DFL-MVO", dfl_map),
        ("RCR-DFL", rcr_map),
    ]:
        for seed in SEEDS:
            gross = load_run_series(
                run_map[CFG][seed]
            )

            for cost in COSTS_BPS:
                net = apply_cost(
                    gross,
                    cost,
                )

                m = metrics(net)

                results.append({
                    "model": model,
                    "seed": seed,
                    "cost_bps": cost,
                    "alpha": ALPHA,
                    "lambda": LAMBDA,
                    "max_weight": MAX_WEIGHT,
                    **m,
                })

    for cost in COSTS_BPS:
        ew_net = apply_cost(
            ew_df,
            cost,
        )

        m = metrics(ew_net)

        results.append({
            "model": "Equal Weight",
            "seed": -1,
            "cost_bps": cost,
            "alpha": np.nan,
            "lambda": np.nan,
            "max_weight": np.nan,
            **m,
        })

    seed_results = pd.DataFrame(results)

    seed_results.to_csv(
        OUTPUT_DIR / "fixed_seed_metrics.csv",
        index=False,
    )

    summary = summarize(seed_results)

    summary.to_csv(
        OUTPUT_DIR / "fixed_summary.csv",
        index=False,
    )

    print("\n[6] Creating cumulative wealth plots...")

    for cost in COSTS_BPS:
        dfl_wealth = mean_wealth(
            dfl_map,
            "DFL-MVO",
            cost,
        )

        rcr_wealth = mean_wealth(
            rcr_map,
            "RCR-DFL",
            cost,
        )

        ew_net = apply_cost(
            ew_df,
            cost,
        )

        ew_wealth = pd.Series(
            ew_net["wealth"].to_numpy(),
            index=pd.to_datetime(ew_net["date"]),
        )

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        ax.plot(
            ew_wealth.index,
            ew_wealth.values,
            label="Equal Weight",
            linewidth=2,
        )

        ax.plot(
            dfl_wealth.index,
            dfl_wealth.values,
            label="DFL-MVO",
            linewidth=2,
        )

        ax.plot(
            rcr_wealth.index,
            rcr_wealth.values,
            label="RCR-DFL",
            linewidth=2,
        )

        ax.set_title(
            f"DOW30 Transaction Cost = {cost} bp\n"
            f"Fixed alpha={ALPHA}, "
            f"lambda={LAMBDA}, "
            f"max_weight={MAX_WEIGHT}"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Wealth")
        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"fixed_alpha0.5_lambda1_{cost}bp.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    print("\n" + "=" * 90)
    print("FIXED-PARAMETER RESULTS")
    print("=" * 90)

    display_cols = [
        "cost_bps",
        "model",
        "cagr_mean",
        "cagr_std",
        "volatility_mean",
        "volatility_std",
        "sharpe_mean",
        "sharpe_std",
        "mdd_mean",
        "mdd_std",
        "turnover_mean",
    ]

    print(
        summary[display_cols].to_string(
            index=False
        )
    )

    print("\n" + "=" * 90)
    print("SAVED FILES")
    print("=" * 90)

    print(
        OUTPUT_DIR
        / "fixed_seed_metrics.csv"
    )

    print(
        OUTPUT_DIR
        / "fixed_summary.csv"
    )

    for cost in COSTS_BPS:
        print(
            OUTPUT_DIR
            / f"fixed_alpha0.5_lambda1_{cost}bp.png"
        )


if __name__ == "__main__":
    main()