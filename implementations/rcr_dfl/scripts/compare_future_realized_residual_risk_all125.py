from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy.stats import ttest_rel, wilcoxon

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset, chronological_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="All-125 DFL-MVO vs RCR-DFL future realized CAPM-residual risk analysis."
    )
    p.add_argument("--price-csv", default="data/raw/sp100_adjusted_close.csv")
    p.add_argument("--paired-csv", default="sp100_dfl_vs_rcr_paired_all_125.csv")
    p.add_argument(
        "--dfl-root",
        default="implementations/dfl_mvo_lee2025/outputs/combined/sp100/sp100_full_grid_h64_d0_s30_std",
    )
    p.add_argument(
        "--rcr-root",
        default="implementations/rcr_dfl/outputs/sp100_grid25_multiseed",
    )
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--future-horizon", type=int, default=20)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--risk-free-rate", type=float, default=0.0)
    p.add_argument("--hac-lags", type=int, default=19)
    p.add_argument(
        "--output-dir",
        default="implementations/rcr_dfl/outputs/sp100_future_realized_residual_risk_all125",
    )
    return p.parse_args()


def hac_mean_test(values: np.ndarray, maxlags: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    fit = sm.OLS(x, np.ones((len(x), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return float(fit.params[0]), float(fit.bse[0]), float(fit.pvalues[0])


def downside_deviation(returns: np.ndarray) -> float:
    downside = np.minimum(np.asarray(returns, dtype=np.float64), 0.0)
    return float(np.sqrt(np.mean(downside**2) * 252.0))


def resolve_paired_csv(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p

    matches = list(Path.cwd().rglob(p.name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        exact = [x for x in matches if x.name == p.name]
        if len(exact) == 1:
            return exact[0]
        raise FileNotFoundError(
            f"Multiple {p.name} files found. Pass --paired-csv with the exact path."
        )
    raise FileNotFoundError(
        f"{path_arg} not found. Pass --paired-csv with the exact path to "
        "sp100_dfl_vs_rcr_paired_all_125.csv."
    )


def find_run_dir(
    root: Path,
    model_name: str,
    alpha: float,
    risk_lambda: float,
    max_weight: float,
    seed: int,
) -> Path:
    if model_name == "DFL":
        candidates = [
            root / f"alpha_{alpha:.2f}" / f"lambda_{risk_lambda:.2f}"
            / f"maxw_{max_weight:.2f}" / f"seed_{seed}",
            root / f"alpha_{alpha:.2f}" / f"lambda_{risk_lambda:.2f}" / f"seed_{seed}",
        ]
    else:
        candidates = [
            root / f"alpha_{alpha:.2f}" / f"lambda_{risk_lambda:.2f}" / f"seed_{seed}",
            root / f"alpha_{alpha:.2f}" / f"lambda_{risk_lambda:.2f}"
            / f"maxw_{max_weight:.2f}" / f"seed_{seed}",
            root / f"alpha_{alpha:.2f}_lambda_{risk_lambda:.2f}_seed_{seed}",
        ]

    for path in candidates:
        if path.exists():
            return path

    seed_dirs = [p for p in root.rglob(f"seed_{seed}") if p.is_dir()]
    tokens = [f"alpha_{alpha:.2f}", f"lambda_{risk_lambda:.2f}"]
    filtered = [p for p in seed_dirs if all(token in str(p) for token in tokens)]

    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) > 1:
        maxw_token = f"maxw_{max_weight:.2f}"
        maxw_filtered = [p for p in filtered if maxw_token in str(p)]
        if len(maxw_filtered) == 1:
            return maxw_filtered[0]

    raise FileNotFoundError(
        f"{model_name} run directory not found: "
        f"alpha={alpha}, lambda={risk_lambda}, seed={seed}, root={root}"
    )


def find_weight_csv(run_dir: Path) -> Path:
    preferred = [
        run_dir / "asset_predictions.csv",
        run_dir / "test_asset_predictions.csv",
        run_dir / "asset_level_predictions.csv",
    ]
    for path in preferred:
        if path.exists():
            return path

    for path in sorted(run_dir.rglob("*.csv")):
        try:
            sample = pd.read_csv(path, nrows=3)
        except Exception:
            continue
        cols = {str(c).lower() for c in sample.columns}
        if {"date", "ticker"}.issubset(cols) and any(
            c in cols for c in ["predicted_weight", "weight", "portfolio_weight"]
        ):
            return path

    raise FileNotFoundError(f"No asset-level weight CSV found under {run_dir}")


def load_weights(path: Path, tickers: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {str(c).lower(): c for c in df.columns}

    for required in ["date", "ticker"]:
        if required not in lower:
            raise ValueError(f"{path}: missing '{required}' column.")

    weight_col = next(
        (
            lower[c]
            for c in ["predicted_weight", "weight", "portfolio_weight"]
            if c in lower
        ),
        None,
    )
    if weight_col is None:
        raise ValueError(f"{path}: no portfolio-weight column found.")

    df = df.rename(
        columns={
            lower["date"]: "date",
            lower["ticker"]: "ticker",
            weight_col: "weight",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].astype(str)

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(f"{path}: duplicate date/ticker rows.")

    pivot = df.pivot(index="date", columns="ticker", values="weight")

    missing = [t for t in tickers if t not in pivot.columns]
    if missing:
        raise ValueError(
            f"{path}: missing {len(missing)} dataset tickers; first={missing[:10]}"
        )

    pivot = pivot.reindex(columns=tickers)
    if pivot.isna().any().any():
        bad_dates = pivot.index[pivot.isna().any(axis=1)].tolist()[:5]
        raise ValueError(f"{path}: missing weights on dates {bad_dates}")

    sum_error = float((pivot.sum(axis=1) - 1.0).abs().max())
    if sum_error > 1e-3:
        raise ValueError(f"{path}: weights do not sum to 1; max error={sum_error:.6g}")

    return pivot


def future_residual_metrics(weights: np.ndarray, future_residuals: np.ndarray) -> dict[str, float]:
    portfolio_residual = future_residuals @ weights
    daily_var = float(np.var(portfolio_residual, ddof=1))

    return {
        "future_residual_variance": daily_var * 252.0,
        "future_residual_volatility": float(np.sqrt(daily_var * 252.0)),
        "future_residual_downside_deviation": downside_deviation(portfolio_residual),
        "future_residual_abs_mean": float(np.mean(np.abs(portfolio_residual))),
    }


def direction_label(x: float, positive_label: str, negative_label: str) -> str:
    if x > 0:
        return positive_label
    if x < 0:
        return negative_label
    return "No change"


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paired_path = resolve_paired_csv(args.paired_csv)
    paired = pd.read_csv(paired_path)

    required_cols = {
        "alpha", "lambda", "seed",
        "CAGR_DFL", "Volatility_DFL", "Sharpe_DFL",
        "CAGR_RCR", "Volatility_RCR", "Sharpe_RCR",
        "Delta_CAGR", "Delta_Volatility", "Delta_Sharpe",
    }
    missing_cols = required_cols - set(paired.columns)
    if missing_cols:
        raise ValueError(f"{paired_path} missing columns: {sorted(missing_cols)}")

    grid = (
        paired[["alpha", "lambda", "seed"]]
        .drop_duplicates()
        .sort_values(["lambda", "alpha", "seed"])
        .reset_index(drop=True)
    )
    if len(grid) != 125:
        print(f"WARNING: expected 125 unique runs, found {len(grid)}.")

    dataset = RCRRollingMVODataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=args.risk_free_rate,
        fit_intercept=True,
        residual_correlation_shrinkage=args.rho,
        correlation_scaling="trace",
        dtype=torch.float64,
    )
    _, _, test_subset = chronological_split(
        dataset,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )

    tickers = list(dataset.tickers)
    horizon = args.future_horizon

    # Precompute future realized CAPM residuals once per date.
    # This is shared across all 125 runs.
    events: dict[str, np.ndarray] = {}
    for idx in test_subset.indices:
        target_pos = int(dataset.target_positions[idx])
        if target_pos + horizon > len(dataset.returns):
            continue

        item = dataset[idx]
        date = str(item["target_date"])
        alpha_t = item["capm_alpha"].detach().cpu().numpy().astype(np.float64)
        beta_t = item["capm_beta"].detach().cpu().numpy().astype(np.float64)

        future_assets = np.asarray(
            dataset.returns[target_pos:target_pos + horizon], dtype=np.float64
        )
        future_market = np.asarray(
            dataset.market_returns[target_pos:target_pos + horizon], dtype=np.float64
        )

        asset_excess = future_assets - args.risk_free_rate
        market_excess = future_market - args.risk_free_rate

        future_residuals = (
            asset_excess
            - alpha_t[None, :]
            - market_excess[:, None] * beta_t[None, :]
        )
        events[date] = future_residuals

    print(f"Precomputed future residual events: {len(events)} dates")

    run_summary_rows = []
    daily_rows = []
    resolved_rows = []

    for run_no, row in grid.iterrows():
        alpha = float(row["alpha"])
        risk_lambda = float(row["lambda"])
        seed = int(row["seed"])

        dfl_run = find_run_dir(
            Path(args.dfl_root), "DFL",
            alpha, risk_lambda, args.max_weight, seed,
        )
        rcr_run = find_run_dir(
            Path(args.rcr_root), "RCR",
            alpha, risk_lambda, args.max_weight, seed,
        )

        dfl_csv = find_weight_csv(dfl_run)
        rcr_csv = find_weight_csv(rcr_run)
        dfl_w = load_weights(dfl_csv, tickers)
        rcr_w = load_weights(rcr_csv, tickers)

        resolved_rows.append({
            "alpha": alpha,
            "lambda": risk_lambda,
            "seed": seed,
            "dfl_weight_csv": str(dfl_csv),
            "rcr_weight_csv": str(rcr_csv),
        })

        matched_dates = sorted(
            set(events).intersection(dfl_w.index).intersection(rcr_w.index)
        )
        if not matched_dates:
            raise RuntimeError(
                f"No matched dates for alpha={alpha}, lambda={risk_lambda}, seed={seed}"
            )

        per_run = []

        for date in matched_dates:
            future_residuals = events[date]
            w_dfl = dfl_w.loc[date].to_numpy(dtype=np.float64)
            w_rcr = rcr_w.loc[date].to_numpy(dtype=np.float64)

            dfl_m = future_residual_metrics(w_dfl, future_residuals)
            rcr_m = future_residual_metrics(w_rcr, future_residuals)

            d = {
                "alpha": alpha,
                "lambda": risk_lambda,
                "seed": seed,
                "date": date,
            }
            for metric in dfl_m:
                d[f"{metric}_dfl"] = dfl_m[metric]
                d[f"{metric}_rcr"] = rcr_m[metric]
                d[f"{metric}_reduction_dfl_minus_rcr"] = (
                    dfl_m[metric] - rcr_m[metric]
                )
            daily_rows.append(d)
            per_run.append(d)

        run_df = pd.DataFrame(per_run)
        summary = {
            "alpha": alpha,
            "lambda": risk_lambda,
            "seed": seed,
            "n_dates": len(run_df),
        }

        for metric in [
            "future_residual_variance",
            "future_residual_volatility",
            "future_residual_downside_deviation",
            "future_residual_abs_mean",
        ]:
            dfl = run_df[f"{metric}_dfl"].to_numpy()
            rcr = run_df[f"{metric}_rcr"].to_numpy()
            reduction = dfl - rcr

            mean_reduction, hac_se, hac_p = hac_mean_test(
                reduction, args.hac_lags
            )
            dfl_mean = float(np.mean(dfl))
            rcr_mean = float(np.mean(rcr))

            summary[f"{metric}_dfl"] = dfl_mean
            summary[f"{metric}_rcr"] = rcr_mean
            summary[f"{metric}_reduction_dfl_minus_rcr"] = mean_reduction
            summary[f"{metric}_relative_reduction"] = (
                mean_reduction / dfl_mean if dfl_mean != 0 else np.nan
            )
            summary[f"{metric}_hac_se"] = hac_se
            summary[f"{metric}_hac_p"] = hac_p

        run_summary_rows.append(summary)

        if (run_no + 1) % 5 == 0 or run_no + 1 == len(grid):
            print(f"[{run_no + 1:3d}/{len(grid)}] completed")

    daily = pd.DataFrame(daily_rows)
    run_summary = pd.DataFrame(run_summary_rows)
    resolved = pd.DataFrame(resolved_rows)

    daily.to_csv(out / "all125_daily_future_residual_risk.csv", index=False)
    resolved.to_csv(out / "resolved_weight_files_all125.csv", index=False)

    merged = paired.merge(
        run_summary,
        on=["alpha", "lambda", "seed"],
        how="inner",
        validate="one_to_one",
    )

    # Sign conventions:
    # Delta_CAGR = RCR - DFL, positive => RCR return higher.
    # Delta_Volatility = RCR - DFL, negative => RCR total volatility lower.
    # residual reduction = DFL - RCR, positive => RCR future residual risk lower.
    merged["Return_Direction"] = merged["Delta_CAGR"].apply(
        lambda x: direction_label(x, "Return ↑", "Return ↓")
    )
    merged["TotalRisk_Direction"] = merged["Delta_Volatility"].apply(
        lambda x: direction_label(-x, "Risk ↓", "Risk ↑")
    )
    merged["Sharpe_Direction"] = merged["Delta_Sharpe"].apply(
        lambda x: direction_label(x, "Sharpe ↑", "Sharpe ↓")
    )

    resid_red = merged[
        "future_residual_volatility_reduction_dfl_minus_rcr"
    ]
    merged["ResidualRisk_Direction"] = resid_red.apply(
        lambda x: direction_label(x, "Residual Risk ↓", "Residual Risk ↑")
    )

    merged.to_csv(out / "all125_run_summary_with_performance.csv", index=False)

    # 25 hyperparameter cells × 5 seeds summary.
    grid25_rows = []
    for (alpha, risk_lambda), g in merged.groupby(["alpha", "lambda"], sort=False):
        dfl = g["future_residual_volatility_dfl"].to_numpy()
        rcr = g["future_residual_volatility_rcr"].to_numpy()
        delta = dfl - rcr

        t_p = float(ttest_rel(rcr, dfl, nan_policy="omit").pvalue)
        try:
            w_p = float(wilcoxon(rcr, dfl, zero_method="wilcox").pvalue)
        except ValueError:
            w_p = np.nan

        grid25_rows.append({
            "alpha": alpha,
            "lambda": risk_lambda,
            "n_seeds": len(g),
            "dfl_future_residual_vol_mean": float(np.mean(dfl)),
            "rcr_future_residual_vol_mean": float(np.mean(rcr)),
            "mean_reduction_dfl_minus_rcr": float(np.mean(delta)),
            "relative_reduction": float(np.mean(delta) / np.mean(dfl)),
            "rcr_lower_seed_count": int(np.sum(delta > 0)),
            "paired_t_p_value": t_p,
            "wilcoxon_p_value": w_p,
            "mean_delta_cagr": float(g["Delta_CAGR"].mean()),
            "mean_delta_volatility": float(g["Delta_Volatility"].mean()),
            "mean_delta_sharpe": float(g["Delta_Sharpe"].mean()),
        })

    grid25_summary = pd.DataFrame(grid25_rows).sort_values(
        ["lambda", "alpha"]
    )
    grid25_summary.to_csv(
        out / "grid25_future_residual_risk_5seed_summary.csv",
        index=False,
    )

    # 4-case Return × total-volatility table.
    rt_rows = []
    order_rt = [
        ("Return ↑", "Risk ↑"),
        ("Return ↑", "Risk ↓"),
        ("Return ↓", "Risk ↑"),
        ("Return ↓", "Risk ↓"),
    ]
    for ret_dir, risk_dir in order_rt:
        g = merged[
            (merged["Return_Direction"] == ret_dir)
            & (merged["TotalRisk_Direction"] == risk_dir)
        ]
        rt_rows.append({
            "Return_Direction": ret_dir,
            "TotalRisk_Direction": risk_dir,
            "Runs": len(g),
            "Pct_of_125": len(g) / len(merged),
            "Mean_Delta_CAGR": g["Delta_CAGR"].mean(),
            "Mean_Delta_TotalVol": g["Delta_Volatility"].mean(),
            "Mean_Delta_Sharpe": g["Delta_Sharpe"].mean(),
            "Sharpe_Improved_Runs": int((g["Delta_Sharpe"] > 0).sum()),
            "ResidualRisk_Lower_Runs": int(
                (
                    g["future_residual_volatility_reduction_dfl_minus_rcr"]
                    > 0
                ).sum()
            ),
            "Mean_Delta_FutureResidualVol_RCRminusDFL": (
                g["future_residual_volatility_rcr"]
                - g["future_residual_volatility_dfl"]
            ).mean(),
        })

    pattern_return_totalrisk = pd.DataFrame(rt_rows)
    pattern_return_totalrisk.to_csv(
        out / "pattern_return_vs_totalrisk_4case.csv",
        index=False,
    )

    # 4-case Future residual risk × Sharpe.
    rs_rows = []
    order_rs = [
        ("Residual Risk ↓", "Sharpe ↑"),
        ("Residual Risk ↑", "Sharpe ↑"),
        ("Residual Risk ↓", "Sharpe ↓"),
        ("Residual Risk ↑", "Sharpe ↓"),
    ]
    for resid_dir, sharpe_dir in order_rs:
        g = merged[
            (merged["ResidualRisk_Direction"] == resid_dir)
            & (merged["Sharpe_Direction"] == sharpe_dir)
        ]
        rs_rows.append({
            "ResidualRisk_Direction": resid_dir,
            "Sharpe_Direction": sharpe_dir,
            "Runs": len(g),
            "Pct_of_125": len(g) / len(merged),
            "Mean_Delta_CAGR": g["Delta_CAGR"].mean(),
            "Mean_Delta_TotalVol": g["Delta_Volatility"].mean(),
            "Mean_Delta_Sharpe": g["Delta_Sharpe"].mean(),
            "Mean_Delta_FutureResidualVol_RCRminusDFL": (
                g["future_residual_volatility_rcr"]
                - g["future_residual_volatility_dfl"]
            ).mean(),
        })

    pattern_resid_sharpe = pd.DataFrame(rs_rows)
    pattern_resid_sharpe.to_csv(
        out / "pattern_residualrisk_vs_sharpe_4case.csv",
        index=False,
    )

    # Full 8-case Return × total risk × future residual risk decomposition.
    full8 = (
        merged.groupby(
            ["Return_Direction", "TotalRisk_Direction", "ResidualRisk_Direction"],
            dropna=False,
        )
        .agg(
            Runs=("seed", "size"),
            Mean_Delta_CAGR=("Delta_CAGR", "mean"),
            Mean_Delta_TotalVol=("Delta_Volatility", "mean"),
            Mean_Delta_Sharpe=("Delta_Sharpe", "mean"),
            Sharpe_Improved_Runs=("Delta_Sharpe", lambda x: int((x > 0).sum())),
            Mean_FutureResidualVol_DFL=("future_residual_volatility_dfl", "mean"),
            Mean_FutureResidualVol_RCR=("future_residual_volatility_rcr", "mean"),
        )
        .reset_index()
    )
    full8["Pct_of_125"] = full8["Runs"] / len(merged)
    full8.to_csv(out / "pattern_return_totalrisk_residualrisk_8case.csv", index=False)

    # Sharpe-improved subset: how often residual risk rose/fell.
    sharpe_up = merged[merged["Delta_Sharpe"] > 0]
    sharpe_breakdown = (
        sharpe_up.groupby(
            ["Return_Direction", "TotalRisk_Direction", "ResidualRisk_Direction"],
            dropna=False,
        )
        .agg(
            Runs=("seed", "size"),
            Mean_Delta_CAGR=("Delta_CAGR", "mean"),
            Mean_Delta_TotalVol=("Delta_Volatility", "mean"),
            Mean_Delta_Sharpe=("Delta_Sharpe", "mean"),
        )
        .reset_index()
    )
    sharpe_breakdown["Pct_of_SharpeImproved"] = (
        sharpe_breakdown["Runs"] / len(sharpe_up)
    )
    sharpe_breakdown.to_csv(
        out / "sharpe_improved_mechanism_breakdown.csv",
        index=False,
    )

    # Overall descriptive headline.
    overall = pd.DataFrame([{
        "Runs": len(merged),
        "Return_Improved": int((merged["Delta_CAGR"] > 0).sum()),
        "TotalVol_Lower": int((merged["Delta_Volatility"] < 0).sum()),
        "Sharpe_Improved": int((merged["Delta_Sharpe"] > 0).sum()),
        "FutureResidualVol_Lower": int((resid_red > 0).sum()),
        "FutureResidualVol_Higher": int((resid_red < 0).sum()),
        "FutureResidualVol_Lower_HAC_sig": int(
            ((resid_red > 0) &
             (merged["future_residual_volatility_hac_p"] < 0.05)).sum()
        ),
        "FutureResidualVol_Higher_HAC_sig": int(
            ((resid_red < 0) &
             (merged["future_residual_volatility_hac_p"] < 0.05)).sum()
        ),
        "Mean_DFL_FutureResidualVol": merged[
            "future_residual_volatility_dfl"
        ].mean(),
        "Mean_RCR_FutureResidualVol": merged[
            "future_residual_volatility_rcr"
        ].mean(),
    }])
    overall.to_csv(out / "overall_125_summary.csv", index=False)

    print("\n" + "=" * 120)
    print("ALL-125 FUTURE REALIZED RESIDUAL RISK SUMMARY")
    print("=" * 120)
    print(overall.to_string(index=False))

    print("\n" + "=" * 120)
    print("RETURN × TOTAL VOLATILITY — 4 CASES")
    print("=" * 120)
    print(pattern_return_totalrisk.to_string(index=False))

    print("\n" + "=" * 120)
    print("FUTURE RESIDUAL RISK × SHARPE — 4 CASES")
    print("=" * 120)
    print(pattern_resid_sharpe.to_string(index=False))

    print("\n" + "=" * 120)
    print("SHARPE-IMPROVED RUNS — MECHANISM BREAKDOWN")
    print("=" * 120)
    print(sharpe_breakdown.to_string(index=False))

    print("\nSaved to:", out)
    for name in [
        "overall_125_summary.csv",
        "all125_run_summary_with_performance.csv",
        "grid25_future_residual_risk_5seed_summary.csv",
        "pattern_return_vs_totalrisk_4case.csv",
        "pattern_residualrisk_vs_sharpe_4case.csv",
        "pattern_return_totalrisk_residualrisk_8case.csv",
        "sharpe_improved_mechanism_breakdown.csv",
        "all125_daily_future_residual_risk.csv",
        "resolved_weight_files_all125.csv",
    ]:
        print(" -", name)


if __name__ == "__main__":
    main()
