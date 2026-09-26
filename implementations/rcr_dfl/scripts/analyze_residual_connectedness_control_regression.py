from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset, chronological_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Control regression for stock-level residual connectedness and future risk."
    )
    p.add_argument("--price-csv", default="data/raw/sp100_adjusted_close.csv")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--future-horizon", type=int, default=20)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--risk-free-rate", type=float, default=0.0)
    p.add_argument(
        "--output-dir",
        default="implementations/rcr_dfl/outputs/sp100_residual_connectedness_control_regression",
    )
    return p.parse_args()


def maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    peak = np.maximum.accumulate(wealth)
    drawdown = 1.0 - wealth / peak
    return float(drawdown.max())


def cross_sectional_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    mean = g.transform("mean")
    std = g.transform("std")
    return (df[col] - mean) / std.replace(0.0, np.nan)


def two_way_demean(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Exact two-way FE transformation for this balanced date x ticker panel:
    x_it - mean_i(x) - mean_t(x) + grand_mean(x)
    """
    return (
        df[col]
        - df.groupby("ticker")[col].transform("mean")
        - df.groupby("date")[col].transform("mean")
        + df[col].mean()
    )


def fit_two_way_clustered(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, np.ndarray, pd.DataFrame]:
    cols = [outcome] + predictors + ["date", "ticker"]
    work = df[cols].dropna().copy()

    y = two_way_demean(work, outcome).to_numpy(dtype=np.float64)
    x_cols = []
    x_arrays = []

    for col in predictors:
        demeaned = two_way_demean(work, col).to_numpy(dtype=np.float64)
        x_cols.append(col)
        x_arrays.append(demeaned)

    x = np.column_stack(x_arrays)
    fit = sm.OLS(y, x).fit()

    date_codes = pd.Categorical(work["date"]).codes
    ticker_codes = pd.Categorical(work["ticker"]).codes
    cov_both, _, _ = cov_cluster_2groups(fit, date_codes, ticker_codes)

    bse = np.sqrt(np.diag(cov_both))
    zstat = fit.params / bse
    pvalue = 2.0 * norm.sf(np.abs(zstat))

    table = pd.DataFrame({
        "term": x_cols,
        "coef": fit.params,
        "two_way_cluster_se": bse,
        "z": zstat,
        "p_value": pvalue,
        "n_obs": len(work),
        "r2_within": fit.rsquared,
    })

    return fit, cov_both, table


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    tickers = np.asarray(dataset.tickers)
    all_returns = dataset.returns
    n_assets = dataset.n_assets
    horizon = args.future_horizon

    rows: list[dict] = []

    for idx in test_subset.indices:
        target_pos = int(dataset.target_positions[idx])

        if target_pos + horizon > len(all_returns):
            continue

        item = dataset[idx]
        date = str(item["target_date"])

        c_res = item["residual_correlation_raw"].detach().cpu().numpy().astype(np.float64)
        covariance = item["covariance"].detach().cpu().numpy().astype(np.float64)
        beta = item["capm_beta"].detach().cpu().numpy().astype(np.float64)

        c_pos = np.maximum(c_res, 0.0)
        np.fill_diagonal(c_pos, 0.0)
        rc = c_pos.sum(axis=1) / (n_assets - 1)

        past_window = all_returns[target_pos - args.lookback:target_pos]
        past_vol = past_window.std(axis=0, ddof=1) * np.sqrt(252.0)

        standard_asset_vol = np.sqrt(
            np.maximum(np.diag(covariance), 0.0) * 252.0
        )

        future_window = all_returns[target_pos:target_pos + horizon]

        for i in range(n_assets):
            future_r = future_window[:, i]
            future_var = float(np.var(future_r, ddof=1) * 252.0)
            future_vol = float(np.sqrt(future_var))
            future_mdd = maximum_drawdown(future_r)

            rows.append({
                "date": date,
                "ticker": str(tickers[i]),
                "residual_connectedness": float(rc[i]),
                "past_volatility": float(past_vol[i]),
                "beta": float(beta[i]),
                "standard_asset_volatility": float(standard_asset_vol[i]),
                "future_variance": future_var,
                "future_volatility": future_vol,
                "future_mdd": future_mdd,
            })

    panel = pd.DataFrame(rows)

    # Cross-sectional standardization at each formation date.
    for col in ["residual_connectedness", "past_volatility", "beta"]:
        panel[f"z_{col}"] = cross_sectional_zscore(panel, col)

    panel["rc_x_pastvol"] = (
        panel["z_residual_connectedness"] * panel["z_past_volatility"]
    )

    panel = panel.dropna().reset_index(drop=True)
    panel.to_csv(out / "stock_level_regression_panel.csv", index=False)

    outcomes = ["future_variance", "future_volatility", "future_mdd"]

    model_specs = {
        "M0_controls": ["z_past_volatility", "z_beta"],
        "M1_add_RC": [
            "z_residual_connectedness",
            "z_past_volatility",
            "z_beta",
        ],
        "M2_add_interaction": [
            "z_residual_connectedness",
            "z_past_volatility",
            "z_beta",
            "rc_x_pastvol",
        ],
    }

    all_results = []
    model_fit_stats = []
    marginal_rows = []

    for outcome in outcomes:
        fitted = {}

        for model_name, predictors in model_specs.items():
            fit, cov, table = fit_two_way_clustered(panel, outcome, predictors)
            table.insert(0, "model", model_name)
            table.insert(0, "outcome", outcome)
            all_results.append(table)

            fitted[model_name] = (fit, cov, predictors)

            model_fit_stats.append({
                "outcome": outcome,
                "model": model_name,
                "n_obs": int(table["n_obs"].iloc[0]),
                "r2_within": float(fit.rsquared),
            })

        # Incremental within-R2.
        m0_r2 = fitted["M0_controls"][0].rsquared
        m1_r2 = fitted["M1_add_RC"][0].rsquared
        m2_r2 = fitted["M2_add_interaction"][0].rsquared

        model_fit_stats[-3]["delta_r2_vs_previous"] = np.nan
        model_fit_stats[-2]["delta_r2_vs_previous"] = float(m1_r2 - m0_r2)
        model_fit_stats[-1]["delta_r2_vs_previous"] = float(m2_r2 - m1_r2)

        # Marginal effect of RC from interaction model:
        # dY/d(z_RC) = beta_RC + beta_interaction * z_past_vol
        fit, cov, predictors = fitted["M2_add_interaction"]
        rc_idx = predictors.index("z_residual_connectedness")
        int_idx = predictors.index("rc_x_pastvol")

        b_rc = float(fit.params[rc_idx])
        b_int = float(fit.params[int_idx])

        for z_past in [-1.5, -1.0, 0.0, 1.0, 1.5]:
            effect = b_rc + b_int * z_past
            var_effect = (
                cov[rc_idx, rc_idx]
                + (z_past ** 2) * cov[int_idx, int_idx]
                + 2.0 * z_past * cov[rc_idx, int_idx]
            )
            se = float(np.sqrt(max(var_effect, 0.0)))
            zstat = effect / se if se > 0 else np.nan
            pvalue = 2.0 * norm.sf(abs(zstat)) if np.isfinite(zstat) else np.nan

            marginal_rows.append({
                "outcome": outcome,
                "z_past_volatility": z_past,
                "marginal_effect_of_RC": effect,
                "two_way_cluster_se": se,
                "z": zstat,
                "p_value": pvalue,
            })

    results = pd.concat(all_results, ignore_index=True)
    fit_stats = pd.DataFrame(model_fit_stats)
    marginals = pd.DataFrame(marginal_rows)

    results.to_csv(out / "control_regression_coefficients.csv", index=False)
    fit_stats.to_csv(out / "control_regression_fit_stats.csv", index=False)
    marginals.to_csv(out / "rc_marginal_effect_by_past_volatility.csv", index=False)

    print("=" * 120)
    print("CONTROL REGRESSION — RESIDUAL CONNECTEDNESS AND FUTURE RISK")
    print("=" * 120)
    print(f"Assets               : {n_assets}")
    print(f"Observations         : {len(panel)}")
    print(f"Lookback             : {args.lookback}")
    print(f"Future horizon       : {horizon}")
    print("Fixed effects        : date + ticker (two-way demeaned)")
    print("SE                   : two-way clustered by date + ticker")
    print("Predictors           : cross-sectional z-scores by date")
    print()

    for outcome in outcomes:
        print("\n" + "-" * 120)
        print(outcome.upper())
        print("-" * 120)

        sub = results[
            (results["outcome"] == outcome)
            & (results["model"] == "M2_add_interaction")
        ][
            ["term", "coef", "two_way_cluster_se", "z", "p_value"]
        ]
        print("M2: controls + RC + RC×PastVol")
        print(sub.to_string(index=False))

        print("\nIncremental within-R²")
        print(
            fit_stats[fit_stats["outcome"] == outcome][
                ["model", "r2_within", "delta_r2_vs_previous"]
            ].to_string(index=False)
        )

        print("\nMarginal effect of RC by past-volatility level")
        print(
            marginals[marginals["outcome"] == outcome][
                [
                    "z_past_volatility",
                    "marginal_effect_of_RC",
                    "two_way_cluster_se",
                    "p_value",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    for name in [
        "stock_level_regression_panel.csv",
        "control_regression_coefficients.csv",
        "control_regression_fit_stats.csv",
        "rc_marginal_effect_by_past_volatility.csv",
    ]:
        print(out / name)


if __name__ == "__main__":
    main()
