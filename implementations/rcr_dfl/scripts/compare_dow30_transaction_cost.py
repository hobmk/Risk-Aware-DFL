import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COSTS_BPS = [0, 5, 10]


def parse_path_value(path, name):
    m = re.search(rf"{name}_([0-9.]+)", str(path).replace("\\", "/"))
    return float(m.group(1)) if m else None


def parse_seed(path):
    m = re.search(r"seed_(\d+)", str(path).replace("\\", "/"))
    return int(m.group(1)) if m else None


def read_config(seed_dir):
    path = seed_dir / "config.json"
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_arg(config, key, default=None):
    args = config.get("arguments", {})
    return args.get(key, config.get(key, default))


def discover_runs(root, model, max_weight):
    root = Path(root)
    runs = []

    for seed_dir in root.rglob("seed_*"):
        if not seed_dir.is_dir():
            continue

        seed = parse_seed(seed_dir)
        if seed is None:
            continue

        config = read_config(seed_dir)

        alpha = get_arg(config, "alpha")
        lam = get_arg(config, "lambda")

        if alpha is None:
            alpha = parse_path_value(seed_dir, "alpha")

        if lam is None:
            lam = parse_path_value(seed_dir, "lambda")

        run_maxw = get_arg(config, "max_weight")

        if run_maxw is None:
            run_maxw = parse_path_value(seed_dir, "maxw")

        if run_maxw is None and model == "RCR-DFL":
            run_maxw = max_weight

        if alpha is None or lam is None:
            continue

        alpha = float(alpha)
        lam = float(lam)
        run_maxw = float(run_maxw)

        if not np.isclose(run_maxw, max_weight):
            continue

        asset_path = seed_dir / "asset_predictions.csv"
        daily_path = seed_dir / "daily_portfolio.csv"

        if not asset_path.exists() and not daily_path.exists():
            continue

        runs.append({
            "model": model,
            "alpha": alpha,
            "lambda": lam,
            "max_weight": run_maxw,
            "seed": seed,
            "dir": seed_dir,
            "asset_path": asset_path if asset_path.exists() else None,
            "daily_path": daily_path if daily_path.exists() else None,
        })

    return runs


def reconstruct_from_asset_predictions(path):
    df = pd.read_csv(path)

    required = {
        "date",
        "ticker",
        "true_return",
        "predicted_weight",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}\n"
            f"asset_predictions.csv missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df["true_return"] = pd.to_numeric(df["true_return"], errors="coerce")
    df["predicted_weight"] = pd.to_numeric(
        df["predicted_weight"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date", "ticker", "true_return", "predicted_weight"]
    )

    dates = sorted(df["date"].unique())

    rows = []

    prev_weights = None
    prev_returns = None

    for date in dates:
        x = (
            df[df["date"] == date]
            .sort_values("ticker")
            .reset_index(drop=True)
        )

        tickers = x["ticker"].tolist()

        w = x["predicted_weight"].to_numpy(dtype=float)
        r = x["true_return"].to_numpy(dtype=float)

        gross_return = float(np.dot(w, r))

        if prev_weights is None:
            turnover = 0.0
        else:
            prev_tickers = prev_weights.index.tolist()

            if prev_tickers != tickers:
                raise ValueError(
                    f"Ticker ordering/universe changed on {date}."
                )

            w_prev = prev_weights.to_numpy(dtype=float)
            r_prev = prev_returns.to_numpy(dtype=float)

            denom = 1.0 + float(np.dot(w_prev, r_prev))

            if abs(denom) < 1e-12:
                raise ValueError(
                    f"Invalid drift denominator on {date}"
                )

            w_pre = w_prev * (1.0 + r_prev) / denom

            turnover = 0.5 * np.abs(w - w_pre).sum()

        rows.append({
            "date": pd.Timestamp(date),
            "gross_return_reconstructed": gross_return,
            "turnover_reconstructed": float(turnover),
        })

        prev_weights = pd.Series(w, index=tickers)
        prev_returns = pd.Series(r, index=tickers)

    return pd.DataFrame(rows)


def equal_weight_from_asset_predictions(path):
    df = pd.read_csv(path)

    required = {"date", "ticker", "true_return"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}\n"
            f"asset_predictions.csv missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df["true_return"] = pd.to_numeric(df["true_return"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "true_return"])

    dates = sorted(df["date"].unique())
    rows = []

    prev_weights = None
    prev_returns = None
    prev_tickers = None

    for date in dates:
        x = (
            df[df["date"] == date]
            .sort_values("ticker")
            .reset_index(drop=True)
        )

        tickers = x["ticker"].tolist()
        r = x["true_return"].to_numpy(dtype=float)

        n_assets = len(tickers)
        w = np.full(n_assets, 1.0 / n_assets)

        # Daily-rebalanced Equal Weight gross return
        gross_return = float(np.dot(w, r))

        # 첫 거래일은 DFL/RCR과 동일하게 turnover=0 처리
        if prev_weights is None:
            turnover = 0.0

        else:
            if tickers != prev_tickers:
                raise ValueError(
                    f"Equal Weight universe changed on {date}."
                )

            # 전일 목표비중이 시장수익률에 의해 drift된 오늘 장 시작 전 비중
            denom = 1.0 + float(np.dot(prev_weights, prev_returns))

            if abs(denom) < 1e-12:
                raise ValueError(
                    f"Invalid EW drift denominator on {date}"
                )

            w_pre = (
                prev_weights
                * (1.0 + prev_returns)
                / denom
            )

            # 다시 1/N으로 맞추기 위해 필요한 one-way turnover
            turnover = 0.5 * np.abs(w - w_pre).sum()

        rows.append({
            "date": pd.Timestamp(date),
            "gross_return": gross_return,
            "turnover": float(turnover),
        })

        prev_weights = w.copy()
        prev_returns = r.copy()
        prev_tickers = tickers

    return pd.DataFrame(rows)

def find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    for col in df.columns:
        name = str(col).lower()

        for candidate in candidates:
            if candidate.lower() in name:
                return col

    return None


def load_run_series(run):
    reconstructed = None

    if run["asset_path"] is not None:
        reconstructed = reconstruct_from_asset_predictions(
            run["asset_path"]
        )

    if run["daily_path"] is None:
        if reconstructed is None:
            raise ValueError(
                f"No usable data in {run['dir']}"
            )

        return reconstructed.rename(columns={
            "gross_return_reconstructed": "gross_return",
            "turnover_reconstructed": "turnover",
        })

    daily = pd.read_csv(run["daily_path"])

    date_col = find_col(
        daily,
        ["date", "datetime", "timestamp"],
    )

    return_col = find_col(
        daily,
        [
            "portfolio_return",
            "gross_return",
            "daily_return",
            "portfolio_ret",
        ],
    )

    turnover_col = find_col(
        daily,
        [
            "daily_turnover",
            "portfolio_turnover",
            "turnover",
        ],
    )

    if date_col is None or return_col is None:
        if reconstructed is None:
            raise ValueError(
                f"Could not determine return columns: "
                f"{run['daily_path']}"
            )

        return reconstructed.rename(columns={
            "gross_return_reconstructed": "gross_return",
            "turnover_reconstructed": "turnover",
        })

    result = pd.DataFrame({
        "date": pd.to_datetime(daily[date_col]),
        "gross_return": pd.to_numeric(
            daily[return_col],
            errors="coerce",
        ),
    })

    # RCR에 기존 turnover가 저장되어 있으면 그것을 우선 사용.
    if turnover_col is not None:
        result["turnover"] = pd.to_numeric(
            daily[turnover_col],
            errors="coerce",
        ).fillna(0.0)

    elif reconstructed is not None:
        result = result.merge(
            reconstructed[
                [
                    "date",
                    "gross_return_reconstructed",
                    "turnover_reconstructed",
                ]
            ],
            on="date",
            how="inner",
        )

        diff = (
            result["gross_return"]
            - result["gross_return_reconstructed"]
        ).abs().max()

        if diff > 1e-6:
            print(
                f"[WARNING] Return reconstruction mismatch "
                f"{run['dir']} | max diff={diff:.8g}"
            )

        result["turnover"] = result[
            "turnover_reconstructed"
        ]

        result = result[
            ["date", "gross_return", "turnover"]
        ]

    else:
        raise ValueError(
            f"No turnover information or weights: {run['dir']}"
        )

    return (
        result
        .dropna(subset=["date", "gross_return", "turnover"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def apply_cost(df, cost_bps):
    result = df.copy()

    result["net_return"] = (
        result["gross_return"]
        - (cost_bps / 10000.0) * result["turnover"]
    )

    result["wealth"] = (
        1.0 + result["net_return"]
    ).cumprod()

    return result


def metrics(df):
    r = df["net_return"].dropna()

    if len(r) == 0:
        return {
            "cagr": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "mdd": np.nan,
            "cumulative_return": np.nan,
            "turnover": np.nan,
        }

    wealth = (1.0 + r).cumprod()

    years = len(r) / 252.0

    if years > 0 and wealth.iloc[-1] > 0:
        cagr = wealth.iloc[-1] ** (1.0 / years) - 1.0
    else:
        cagr = np.nan

    std = r.std(ddof=1)

    volatility = std * np.sqrt(252)

    sharpe = (
        r.mean() / std * np.sqrt(252)
        if std > 0
        else np.nan
    )

    drawdown = 1.0 - wealth / wealth.cummax()

    mdd = drawdown.max()

    return {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "mdd": mdd,
        "cumulative_return": wealth.iloc[-1] - 1.0,
        "turnover": df["turnover"].mean(),
    }


def config_key(run):
    return (
        round(run["alpha"], 10),
        round(run["lambda"], 10),
        round(run["max_weight"], 10),
    )


def build_run_map(runs):
    result = {}

    for run in runs:
        key = config_key(run)

        if key not in result:
            result[key] = {}

        result[key][run["seed"]] = run

    return result


def create_coverage_report(
    dfl_map,
    rcr_map,
    required_seeds,
):
    configs = sorted(
        set(dfl_map.keys()) | set(rcr_map.keys())
    )

    rows = []

    for cfg in configs:
        dfl_seeds = sorted(dfl_map.get(cfg, {}).keys())
        rcr_seeds = sorted(rcr_map.get(cfg, {}).keys())

        missing_dfl = [
            s for s in required_seeds
            if s not in dfl_seeds
        ]

        missing_rcr = [
            s for s in required_seeds
            if s not in rcr_seeds
        ]

        rows.append({
            "alpha": cfg[0],
            "lambda": cfg[1],
            "max_weight": cfg[2],
            "dfl_seeds": ",".join(map(str, dfl_seeds)),
            "rcr_seeds": ",".join(map(str, rcr_seeds)),
            "missing_dfl": ",".join(map(str, missing_dfl)),
            "missing_rcr": ",".join(map(str, missing_rcr)),
            "paired_complete": (
                len(missing_dfl) == 0
                and len(missing_rcr) == 0
            ),
        })

    return pd.DataFrame(rows)


def evaluate_full_grid(
    dfl_map,
    rcr_map,
    required_seeds,
):
    rows = []
    series_cache = {}

    common_configs = sorted(
        set(dfl_map.keys()) & set(rcr_map.keys())
    )

    for cfg in common_configs:
        if not all(
            seed in dfl_map[cfg]
            and seed in rcr_map[cfg]
            for seed in required_seeds
        ):
            continue

        for model, run_map in [
            ("DFL-MVO", dfl_map),
            ("RCR-DFL", rcr_map),
        ]:
            for seed in required_seeds:
                run = run_map[cfg][seed]

                cache_key = (
                    model,
                    cfg,
                    seed,
                )

                if cache_key not in series_cache:
                    series_cache[cache_key] = (
                        load_run_series(run)
                    )

                gross_df = series_cache[cache_key]

                for cost in COSTS_BPS:
                    net_df = apply_cost(
                        gross_df,
                        cost,
                    )

                    m = metrics(net_df)

                    rows.append({
                        "model": model,
                        "alpha": cfg[0],
                        "lambda": cfg[1],
                        "max_weight": cfg[2],
                        "seed": seed,
                        "cost_bps": cost,
                        **m,
                    })

    return pd.DataFrame(rows), series_cache


def summarize_grid(seed_metrics):
    group = [
        "model",
        "alpha",
        "lambda",
        "max_weight",
        "cost_bps",
    ]

    metrics_cols = [
        "cagr",
        "volatility",
        "sharpe",
        "mdd",
        "cumulative_return",
        "turnover",
    ]

    rows = []

    for keys, g in seed_metrics.groupby(group):
        row = dict(zip(group, keys))

        row["n_seeds"] = g["seed"].nunique()

        for col in metrics_cols:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"] = g[col].std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows)


def select_best_anchors(summary):
    rows = []

    for cost in COSTS_BPS:
        for model in [
            "DFL-MVO",
            "RCR-DFL",
        ]:
            x = summary[
                (summary["cost_bps"] == cost)
                & (summary["model"] == model)
            ]

            if x.empty:
                continue

            best = x.loc[
                x["sharpe_mean"].idxmax()
            ].copy()

            best["anchor_model"] = model

            rows.append(best)

    return pd.DataFrame(rows)


def ew_metrics_for_costs(ew_df):
    rows = []

    for cost in COSTS_BPS:
        net = apply_cost(
            ew_df,
            cost,
        )

        m = metrics(net)

        rows.append({
            "model": "Equal Weight",
            "cost_bps": cost,
            **m,
        })

    return pd.DataFrame(rows)


def create_anchor_comparison(
    summary,
    anchors,
    ew_summary,
):
    rows = []

    for _, anchor in anchors.iterrows():
        cfg_mask = (
            np.isclose(
                summary["alpha"],
                anchor["alpha"],
            )
            & np.isclose(
                summary["lambda"],
                anchor["lambda"],
            )
            & np.isclose(
                summary["max_weight"],
                anchor["max_weight"],
            )
            & (
                summary["cost_bps"]
                == anchor["cost_bps"]
            )
        )

        for model in [
            "DFL-MVO",
            "RCR-DFL",
        ]:
            x = summary[
                cfg_mask
                & (summary["model"] == model)
            ]

            if x.empty:
                continue

            x = x.iloc[0]

            rows.append({
                "anchor_model": anchor["anchor_model"],
                "cost_bps": anchor["cost_bps"],
                "alpha": anchor["alpha"],
                "lambda": anchor["lambda"],
                "max_weight": anchor["max_weight"],
                "model": model,
                "cagr_mean": x["cagr_mean"],
                "cagr_std": x["cagr_std"],
                "volatility_mean": x["volatility_mean"],
                "volatility_std": x["volatility_std"],
                "sharpe_mean": x["sharpe_mean"],
                "sharpe_std": x["sharpe_std"],
                "mdd_mean": x["mdd_mean"],
                "mdd_std": x["mdd_std"],
                "turnover_mean": x["turnover_mean"],
                "turnover_std": x["turnover_std"],
            })

        ew = ew_summary[
            ew_summary["cost_bps"]
            == anchor["cost_bps"]
        ].iloc[0]

        rows.append({
            "anchor_model": anchor["anchor_model"],
            "cost_bps": anchor["cost_bps"],
            "alpha": anchor["alpha"],
            "lambda": anchor["lambda"],
            "max_weight": anchor["max_weight"],
            "model": "Equal Weight",
            "cagr_mean": ew["cagr"],
            "cagr_std": 0.0,
            "volatility_mean": ew["volatility"],
            "volatility_std": 0.0,
            "sharpe_mean": ew["sharpe"],
            "sharpe_std": 0.0,
            "mdd_mean": ew["mdd"],
            "mdd_std": 0.0,
            "turnover_mean": ew["turnover"],
            "turnover_std": 0.0,
        })

    return pd.DataFrame(rows)


def mean_wealth_curve(
    run_map,
    cfg,
    seeds,
    cost_bps,
):
    wealths = []

    for seed in seeds:
        df = load_run_series(
            run_map[cfg][seed]
        )

        net = apply_cost(
            df,
            cost_bps,
        )

        s = pd.Series(
            net["wealth"].to_numpy(),
            index=pd.to_datetime(net["date"]),
            name=str(seed),
        )

        wealths.append(s)

    wealth = pd.concat(
        wealths,
        axis=1,
        join="inner",
    )

    return wealth.mean(axis=1)


def plot_anchors(
    anchors,
    dfl_map,
    rcr_map,
    ew_df,
    required_seeds,
    output_dir,
):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ew_indexed = ew_df.set_index("date")

    for _, anchor in anchors.iterrows():
        cfg = (
            round(anchor["alpha"], 10),
            round(anchor["lambda"], 10),
            round(anchor["max_weight"], 10),
        )

        cost = int(anchor["cost_bps"])
        anchor_model = anchor["anchor_model"]

        dfl_wealth = mean_wealth_curve(
            dfl_map,
            cfg,
            required_seeds,
            cost,
        )

        rcr_wealth = mean_wealth_curve(
            rcr_map,
            cfg,
            required_seeds,
            cost,
        )

        ew_net = apply_cost(
            ew_indexed.reset_index(),
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
            f"DOW30 — {anchor_model} Best Anchor — "
            f"{cost} bp\n"
            f"alpha={cfg[0]:.2f}, "
            f"lambda={cfg[1]:.2f}, "
            f"max_weight={cfg[2]:.2f}"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Wealth")
        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()

        filename = (
            f"{anchor_model.lower().replace('-', '_')}"
            f"_best_anchor_{cost}bp.png"
        )

        fig.savefig(
            plot_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dfl-root",
        default=(
            "implementations/dfl_mvo_lee2025/"
            "outputs/combined/dow30/"
            "final_full_grid_h64_d0_s30_std"
        ),
    )

    parser.add_argument(
        "--rcr-root",
        default=(
            "implementations/rcr_dfl/"
            "outputs/grid25_multiseed"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "implementations/rcr_dfl/"
            "outputs/dow30_transaction_cost_comparison"
        ),
    )

    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("DOW30 TRANSACTION COST COMPARISON")
    print("=" * 90)

    print("\n[1] Discovering DFL-MVO runs...")

    dfl_runs = discover_runs(
        args.dfl_root,
        "DFL-MVO",
        args.max_weight,
    )

    print(
        f"DFL-MVO runs found: {len(dfl_runs)}"
    )

    print("\n[2] Discovering RCR-DFL runs...")

    rcr_runs = discover_runs(
        args.rcr_root,
        "RCR-DFL",
        args.max_weight,
    )

    print(
        f"RCR-DFL runs found: {len(rcr_runs)}"
    )

    if not dfl_runs:
        raise RuntimeError(
            "No DFL-MVO runs found."
        )

    if not rcr_runs:
        raise RuntimeError(
            "No RCR-DFL runs found."
        )

    dfl_map = build_run_map(
        dfl_runs
    )

    rcr_map = build_run_map(
        rcr_runs
    )

    print("\n[3] Checking paired coverage...")

    coverage = create_coverage_report(
        dfl_map,
        rcr_map,
        args.seeds,
    )

    coverage.to_csv(
        output_dir / "coverage_report.csv",
        index=False,
    )

    n_complete = coverage[
        coverage["paired_complete"]
    ].shape[0]

    print(
        f"Complete paired configurations: "
        f"{n_complete}"
    )

    print(
        f"Required seeds: {args.seeds}"
    )

    print("\n[4] Building Equal Weight baseline...")

    ew_source = next(
        run["asset_path"]
        for run in dfl_runs
        if run["asset_path"] is not None
    )

    ew_df = equal_weight_from_asset_predictions(
        ew_source
    )

    print(
        f"EW dates: {len(ew_df)}"
    )

    print("\n[5] Evaluating full paired grid...")

    seed_metrics, _ = evaluate_full_grid(
        dfl_map,
        rcr_map,
        args.seeds,
    )

    if seed_metrics.empty:
        raise RuntimeError(
            "No fully paired configuration exists "
            "for the requested seeds."
        )

    seed_metrics.to_csv(
        output_dir
        / "all_seed_cost_metrics.csv",
        index=False,
    )

    summary = summarize_grid(
        seed_metrics
    )

    summary.to_csv(
        output_dir
        / "paired_grid_summary.csv",
        index=False,
    )

    print("\n[6] Selecting best anchors...")

    anchors = select_best_anchors(
        summary
    )

    anchors.to_csv(
        output_dir
        / "best_anchors.csv",
        index=False,
    )

    ew_summary = ew_metrics_for_costs(
        ew_df
    )

    ew_summary.to_csv(
        output_dir
        / "equal_weight_metrics.csv",
        index=False,
    )

    comparison = create_anchor_comparison(
        summary,
        anchors,
        ew_summary,
    )

    comparison.to_csv(
        output_dir
        / "anchor_comparison.csv",
        index=False,
    )

    print("\n[7] Creating wealth plots...")

    plot_anchors(
        anchors,
        dfl_map,
        rcr_map,
        ew_df,
        args.seeds,
        output_dir,
    )

    print("\n" + "=" * 90)
    print("BEST ANCHORS")
    print("=" * 90)

    print(
        anchors[
            [
                "anchor_model",
                "cost_bps",
                "alpha",
                "lambda",
                "max_weight",
                "cagr_mean",
                "sharpe_mean",
                "mdd_mean",
                "turnover_mean",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("ANCHOR COMPARISON")
    print("=" * 90)

    print(
        comparison[
            [
                "anchor_model",
                "cost_bps",
                "alpha",
                "lambda",
                "model",
                "cagr_mean",
                "sharpe_mean",
                "mdd_mean",
                "turnover_mean",
            ]
        ].to_string(index=False)
    )

    print("\nSaved to:")
    print(output_dir)


if __name__ == "__main__":
    main()