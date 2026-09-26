from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset, chronological_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-sectional residual-connectedness quartile portfolio validation.")
    p.add_argument("--price-csv", default="data/raw/sp100_adjusted_close.csv")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--future-horizon", type=int, default=20)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--risk-free-rate", type=float, default=0.0)
    p.add_argument("--hac-lags", type=int, default=19)
    p.add_argument("--output-dir", default="implementations/rcr_dfl/outputs/sp100_residual_connectedness_quartiles")
    return p.parse_args()


def fixed_initial_weight_returns(asset_returns: np.ndarray) -> np.ndarray:
    """Equal weight at formation, fixed membership, then weights drift naturally."""
    h, k = asset_returns.shape
    values = np.full(k, 1.0 / k, dtype=np.float64)
    portfolio_returns = np.empty(h, dtype=np.float64)
    for t in range(h):
        old_wealth = values.sum()
        values *= 1.0 + asset_returns[t]
        new_wealth = values.sum()
        portfolio_returns[t] = new_wealth / old_wealth - 1.0
    return portfolio_returns


def maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    wealth = np.r_[1.0, wealth]
    peak = np.maximum.accumulate(wealth)
    return float((1.0 - wealth / peak).max())


def future_risk_metrics(returns: np.ndarray) -> dict[str, float]:
    daily_var = float(np.var(returns, ddof=1))
    annualized_variance = daily_var * 252.0
    return {
        "future_variance": annualized_variance,
        "future_volatility": float(np.sqrt(annualized_variance)),
        "future_mdd": maximum_drawdown(returns),
        "future_cumulative_return": float(np.prod(1.0 + returns) - 1.0),
    }


def hac_mean_test(values: np.ndarray, maxlags: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan, np.nan, np.nan
    x = np.ones((len(values), 1), dtype=np.float64)
    fit = sm.OLS(values, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(fit.params[0]), float(fit.bse[0]), float(fit.pvalues[0])


def quartile_slope(values: np.ndarray) -> float:
    return float(np.polyfit(np.arange(1.0, 5.0), values, deg=1)[0])


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

    _, _, test_subset = chronological_split(dataset, args.train_end, args.validation_end)

    tickers = np.asarray(dataset.tickers)
    all_returns = dataset.returns
    n_assets = dataset.n_assets
    horizon = args.future_horizon

    if n_assets < 4:
        raise ValueError("Quartile sorting requires at least 4 assets.")

    risk_rows, membership_rows = [], []
    usable = 0
    rank_mismatch_dates = 0

    for idx in test_subset.indices:
        target_pos = int(dataset.target_positions[idx])
        if target_pos + horizon > len(all_returns):
            continue

        item = dataset[idx]
        date = str(item["target_date"])
        c_res = item["residual_correlation_raw"].detach().cpu().numpy().astype(np.float64)
        a_res = item["a_res"].detach().cpu().numpy().astype(np.float64)
        covariance = item["covariance"].detach().cpu().numpy().astype(np.float64)
        beta = item["capm_beta"].detach().cpu().numpy().astype(np.float64)

        c_pos = np.maximum(c_res, 0.0)
        a_pos = np.maximum(a_res, 0.0)
        np.fill_diagonal(c_pos, 0.0)
        np.fill_diagonal(a_pos, 0.0)

        connectedness_corr = c_pos.sum(axis=1) / (n_assets - 1)
        connectedness_ares = a_pos.sum(axis=1) / (n_assets - 1)

        order_corr = np.argsort(connectedness_corr, kind="stable")
        order_ares = np.argsort(connectedness_ares, kind="stable")
        if not np.array_equal(order_corr, order_ares):
            rank_mismatch_dates += 1

        quartiles = np.array_split(order_corr, 4)
        past_window = all_returns[target_pos - args.lookback:target_pos]
        past_asset_vol = past_window.std(axis=0, ddof=1) * np.sqrt(252.0)
        future_window = all_returns[target_pos:target_pos + horizon]

        for q_idx, members in enumerate(quartiles, start=1):
            k = len(members)
            w = np.full(k, 1.0 / k, dtype=np.float64)
            future_portfolio_returns = fixed_initial_weight_returns(future_window[:, members])
            sigma_q = covariance[np.ix_(members, members)]

            risk_rows.append({
                "date": date,
                "quartile": f"Q{q_idx}",
                "quartile_rank": q_idx,
                "n_assets": k,
                "mean_residual_connectedness": float(connectedness_corr[members].mean()),
                "mean_ares_connectedness": float(connectedness_ares[members].mean()),
                "mean_past_asset_volatility": float(past_asset_vol[members].mean()),
                "mean_beta": float(beta[members].mean()),
                "mean_abs_beta": float(np.abs(beta[members]).mean()),
                "standard_cov_portfolio_volatility": float(np.sqrt(252.0 * (w @ sigma_q @ w))),
                **future_risk_metrics(future_portfolio_returns),
            })

            for asset_idx in members:
                membership_rows.append({
                    "date": date,
                    "ticker": str(tickers[asset_idx]),
                    "quartile": f"Q{q_idx}",
                    "quartile_rank": q_idx,
                    "residual_connectedness": float(connectedness_corr[asset_idx]),
                    "ares_connectedness": float(connectedness_ares[asset_idx]),
                    "past_asset_volatility": float(past_asset_vol[asset_idx]),
                    "beta": float(beta[asset_idx]),
                })

        usable += 1

    risk = pd.DataFrame(risk_rows)
    membership = pd.DataFrame(membership_rows)
    if risk.empty:
        raise RuntimeError("No usable test formation dates.")

    quartile_order = ["Q1", "Q2", "Q3", "Q4"]
    risk["quartile"] = pd.Categorical(risk["quartile"], quartile_order, ordered=True)
    membership["quartile"] = pd.Categorical(membership["quartile"], quartile_order, ordered=True)

    summary_cols = [
        "mean_residual_connectedness", "mean_ares_connectedness",
        "mean_past_asset_volatility", "mean_beta", "mean_abs_beta",
        "standard_cov_portfolio_volatility", "future_variance",
        "future_volatility", "future_mdd", "future_cumulative_return",
    ]
    summary = risk.groupby("quartile", observed=False)[summary_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["quartile" if a == "quartile" else f"{a}_{b}" for a, b in summary.columns]

    hac_rows = []
    for metric in ["future_variance", "future_volatility", "future_mdd"]:
        pivot = risk.pivot(index="date", columns="quartile", values=metric).dropna()
        diff = (pivot["Q4"] - pivot["Q1"]).to_numpy()
        mean_diff, hac_se, p_value = hac_mean_test(diff, args.hac_lags)
        q1_mean, q4_mean = float(pivot["Q1"].mean()), float(pivot["Q4"].mean())
        hac_rows.append({
            "metric": metric,
            "q1_mean": q1_mean,
            "q4_mean": q4_mean,
            "q4_minus_q1": mean_diff,
            "relative_change": q4_mean / q1_mean - 1.0 if q1_mean != 0 else np.nan,
            "hac_se": hac_se,
            "hac_p_value_two_sided": p_value,
            "hac_lags": args.hac_lags,
            "n_dates": len(pivot),
        })
    hac_df = pd.DataFrame(hac_rows)

    trend_rows = []
    for metric in ["future_variance", "future_volatility", "future_mdd"]:
        pivot = risk.pivot(index="date", columns="quartile_rank", values=metric).dropna().sort_index()
        slopes = np.array([quartile_slope(row) for row in pivot[[1, 2, 3, 4]].to_numpy()], dtype=np.float64)
        mean_slope, hac_se, p_value = hac_mean_test(slopes, args.hac_lags)
        trend_rows.append({
            "metric": metric,
            "mean_q1_to_q4_slope": mean_slope,
            "hac_se": hac_se,
            "hac_p_value_two_sided": p_value,
            "hac_lags": args.hac_lags,
            "n_dates": len(slopes),
        })
    trend_df = pd.DataFrame(trend_rows)

    freq = membership.groupby(["ticker", "quartile"], observed=False).size().rename("n_dates").reset_index()
    total_dates = membership.groupby("ticker")["date"].nunique().rename("total_dates")
    freq = freq.merge(total_dates, on="ticker", how="left")
    freq["share"] = freq["n_dates"] / freq["total_dates"]

    risk.to_csv(out / "quartile_daily_future_risk.csv", index=False)
    membership.to_csv(out / "quartile_membership.csv", index=False)
    summary.to_csv(out / "quartile_summary.csv", index=False)
    hac_df.to_csv(out / "q4_vs_q1_hac.csv", index=False)
    trend_df.to_csv(out / "monotonic_trend_hac.csv", index=False)
    freq.to_csv(out / "quartile_constituent_frequency.csv", index=False)

    print("=" * 100)
    print("RESIDUAL CONNECTEDNESS QUARTILE VALIDATION")
    print("=" * 100)
    print(f"Dataset assets       : {n_assets}")
    print(f"Lookback             : {args.lookback}")
    print(f"Future horizon       : {horizon}")
    print(f"Test formation dates : {usable}")
    print(f"rho                  : {args.rho}")
    print(f"Rank mismatch dates  : {rank_mismatch_dates}")

    display_cols = [
        "quartile", "mean_residual_connectedness_mean",
        "mean_past_asset_volatility_mean", "standard_cov_portfolio_volatility_mean",
        "future_variance_mean", "future_volatility_mean", "future_mdd_mean",
    ]
    print("\nQUARTILE SUMMARY")
    print(summary[display_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("Q4 VS Q1 — PAIRED HAC")
    print("=" * 100)
    print(hac_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("MONOTONIC Q1 -> Q4 TREND — HAC")
    print("=" * 100)
    print(trend_df.to_string(index=False))

    print("\nSaved:")
    for name in [
        "quartile_daily_future_risk.csv", "quartile_membership.csv",
        "quartile_summary.csv", "q4_vs_q1_hac.csv",
        "monotonic_trend_hac.csv", "quartile_constituent_frequency.csv",
    ]:
        print(out / name)


if __name__ == "__main__":
    main()
