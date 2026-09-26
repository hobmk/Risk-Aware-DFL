from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset, chronological_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Volatility-controlled double sort for residual connectedness."
    )
    p.add_argument("--price-csv", default="data/raw/sp100_adjusted_close.csv")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--future-horizon", type=int, default=20)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--risk-free-rate", type=float, default=0.0)
    p.add_argument("--hac-lags", type=int, default=19)
    p.add_argument(
        "--output-dir",
        default="implementations/rcr_dfl/outputs/sp100_residual_connectedness_double_sort",
    )
    return p.parse_args()


def fixed_initial_weight_returns(
    asset_returns: np.ndarray,
    initial_weights: np.ndarray,
) -> np.ndarray:
    """
    asset_returns: [H, K]
    initial_weights: [K], sums to 1

    Membership/initial allocation are fixed at formation.
    Weights then drift naturally with realized returns.
    """
    values = initial_weights.astype(np.float64).copy()
    portfolio_returns = np.empty(asset_returns.shape[0], dtype=np.float64)

    for t in range(asset_returns.shape[0]):
        old_wealth = values.sum()
        values *= 1.0 + asset_returns[t]
        new_wealth = values.sum()
        portfolio_returns[t] = new_wealth / old_wealth - 1.0

    return portfolio_returns


def maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    peak = np.maximum.accumulate(wealth)
    drawdown = 1.0 - wealth / peak
    return float(drawdown.max())


def future_risk_metrics(returns: np.ndarray) -> dict[str, float]:
    daily_var = float(np.var(returns, ddof=1))
    annualized_variance = daily_var * 252.0
    annualized_volatility = float(np.sqrt(annualized_variance))
    cumulative_return = float(np.prod(1.0 + returns) - 1.0)

    return {
        "future_variance": annualized_variance,
        "future_volatility": annualized_volatility,
        "future_mdd": maximum_drawdown(returns),
        "future_cumulative_return": cumulative_return,
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


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights))


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

    if n_assets < 16:
        raise ValueError("Double 4x4 sorting requires at least 16 assets.")

    daily_rows: list[dict] = []
    membership_rows: list[dict] = []
    cell_rows: list[dict] = []

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

        rc = c_pos.sum(axis=1) / (n_assets - 1)
        ares_connectedness = a_pos.sum(axis=1) / (n_assets - 1)

        if not np.array_equal(
            np.argsort(rc, kind="stable"),
            np.argsort(ares_connectedness, kind="stable"),
        ):
            rank_mismatch_dates += 1

        past_window = all_returns[target_pos - args.lookback:target_pos]
        past_vol = past_window.std(axis=0, ddof=1) * np.sqrt(252.0)
        future_window = all_returns[target_pos:target_pos + horizon]

        # Stage 1: sort all stocks into 4 buckets by past volatility.
        vol_order = np.argsort(past_vol, kind="stable")
        vol_buckets = np.array_split(vol_order, 4)

        # Stage 2: inside each volatility bucket, sort again by residual connectedness.
        rc_cells: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: [], 4: []}

        for vol_rank, vol_members in enumerate(vol_buckets, start=1):
            local_order = vol_members[
                np.argsort(rc[vol_members], kind="stable")
            ]
            rc_quartiles = np.array_split(local_order, 4)

            for rc_rank, members in enumerate(rc_quartiles, start=1):
                rc_cells[rc_rank].append(members)

                # 16-cell diagnostic: equal weight within each cell.
                cell_weights = np.full(len(members), 1.0 / len(members), dtype=np.float64)
                cell_future_returns = fixed_initial_weight_returns(
                    future_window[:, members],
                    cell_weights,
                )
                cell_metrics = future_risk_metrics(cell_future_returns)

                cell_rows.append({
                    "date": date,
                    "vol_bucket": f"V{vol_rank}",
                    "vol_bucket_rank": vol_rank,
                    "rc_bucket": f"RC{rc_rank}",
                    "rc_bucket_rank": rc_rank,
                    "n_assets": len(members),
                    "mean_past_asset_volatility": float(past_vol[members].mean()),
                    "mean_residual_connectedness": float(rc[members].mean()),
                    **cell_metrics,
                })

                for asset_idx in members:
                    membership_rows.append({
                        "date": date,
                        "ticker": str(tickers[asset_idx]),
                        "vol_bucket": f"V{vol_rank}",
                        "vol_bucket_rank": vol_rank,
                        "rc_bucket": f"RC{rc_rank}",
                        "rc_bucket_rank": rc_rank,
                        "past_asset_volatility": float(past_vol[asset_idx]),
                        "residual_connectedness": float(rc[asset_idx]),
                        "ares_connectedness": float(ares_connectedness[asset_idx]),
                        "beta": float(beta[asset_idx]),
                    })

        # Build four volatility-controlled RC portfolios.
        # Each of the four volatility buckets receives exactly 25% initial capital.
        # Within each Vx-RCy cell, capital is equal-weighted across constituents.
        for rc_rank in range(1, 5):
            members_parts = rc_cells[rc_rank]
            members = np.concatenate(members_parts)

            weights_parts = []
            for cell_members in members_parts:
                weights_parts.append(
                    np.full(
                        len(cell_members),
                        0.25 / len(cell_members),
                        dtype=np.float64,
                    )
                )
            weights = np.concatenate(weights_parts)

            if not np.isclose(weights.sum(), 1.0):
                raise RuntimeError("Controlled portfolio weights do not sum to 1.")

            future_returns = fixed_initial_weight_returns(
                future_window[:, members],
                weights,
            )
            metrics = future_risk_metrics(future_returns)

            sigma_q = covariance[np.ix_(members, members)]
            standard_cov_var_daily = float(weights @ sigma_q @ weights)
            standard_cov_vol = float(np.sqrt(252.0 * standard_cov_var_daily))

            daily_rows.append({
                "date": date,
                "rc_bucket": f"RC{rc_rank}",
                "rc_bucket_rank": rc_rank,
                "n_assets": len(members),
                "mean_residual_connectedness": weighted_mean(rc[members], weights),
                "mean_ares_connectedness": weighted_mean(ares_connectedness[members], weights),
                "mean_past_asset_volatility": weighted_mean(past_vol[members], weights),
                "mean_beta": weighted_mean(beta[members], weights),
                "mean_abs_beta": weighted_mean(np.abs(beta[members]), weights),
                "standard_cov_portfolio_volatility": standard_cov_vol,
                **metrics,
            })

        usable += 1

    daily = pd.DataFrame(daily_rows)
    membership = pd.DataFrame(membership_rows)
    cells = pd.DataFrame(cell_rows)

    if daily.empty:
        raise RuntimeError("No usable test formation dates.")

    rc_order = ["RC1", "RC2", "RC3", "RC4"]
    daily["rc_bucket"] = pd.Categorical(
        daily["rc_bucket"], categories=rc_order, ordered=True
    )
    membership["rc_bucket"] = pd.Categorical(
        membership["rc_bucket"], categories=rc_order, ordered=True
    )
    cells["rc_bucket"] = pd.Categorical(
        cells["rc_bucket"], categories=rc_order, ordered=True
    )

    summary_cols = [
        "mean_residual_connectedness",
        "mean_ares_connectedness",
        "mean_past_asset_volatility",
        "mean_beta",
        "mean_abs_beta",
        "standard_cov_portfolio_volatility",
        "future_variance",
        "future_volatility",
        "future_mdd",
        "future_cumulative_return",
    ]

    summary = (
        daily.groupby("rc_bucket", observed=False)[summary_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "rc_bucket" if a == "rc_bucket" else f"{a}_{b}"
        for a, b in summary.columns
    ]

    cell_summary = (
        cells.groupby(
            ["vol_bucket", "vol_bucket_rank", "rc_bucket", "rc_bucket_rank"],
            observed=False,
        )[
            [
                "mean_past_asset_volatility",
                "mean_residual_connectedness",
                "future_variance",
                "future_volatility",
                "future_mdd",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(["vol_bucket_rank", "rc_bucket_rank"])
    )

    # Aggregate RC4 - RC1 paired HAC.
    hac_rows = []
    for metric in ["future_variance", "future_volatility", "future_mdd"]:
        pivot = daily.pivot(index="date", columns="rc_bucket", values=metric).dropna()
        diff = (pivot["RC4"] - pivot["RC1"]).to_numpy()
        mean_diff, hac_se, p_value = hac_mean_test(diff, args.hac_lags)

        rc1_mean = float(pivot["RC1"].mean())
        rc4_mean = float(pivot["RC4"].mean())

        hac_rows.append({
            "metric": metric,
            "rc1_mean": rc1_mean,
            "rc4_mean": rc4_mean,
            "rc4_minus_rc1": mean_diff,
            "relative_change": rc4_mean / rc1_mean - 1.0 if rc1_mean != 0 else np.nan,
            "hac_se": hac_se,
            "hac_p_value_two_sided": p_value,
            "hac_lags": args.hac_lags,
            "n_dates": len(pivot),
        })
    hac_df = pd.DataFrame(hac_rows)

    # Monotonic RC1 -> RC4 trend after volatility control.
    trend_rows = []
    for metric in ["future_variance", "future_volatility", "future_mdd"]:
        pivot = (
            daily.pivot(index="date", columns="rc_bucket_rank", values=metric)
            .dropna()
            .sort_index()
        )
        slopes = np.array(
            [quartile_slope(row) for row in pivot[[1, 2, 3, 4]].to_numpy()],
            dtype=np.float64,
        )
        mean_slope, hac_se, p_value = hac_mean_test(slopes, args.hac_lags)
        trend_rows.append({
            "metric": metric,
            "mean_rc1_to_rc4_slope": mean_slope,
            "hac_se": hac_se,
            "hac_p_value_two_sided": p_value,
            "hac_lags": args.hac_lags,
            "n_dates": len(slopes),
        })
    trend_df = pd.DataFrame(trend_rows)

    # Within each volatility bucket: RC4 - RC1.
    within_rows = []
    for vol_rank in range(1, 5):
        subset = cells[cells["vol_bucket_rank"] == vol_rank]
        for metric in ["future_variance", "future_volatility", "future_mdd"]:
            pivot = subset.pivot(index="date", columns="rc_bucket", values=metric).dropna()
            diff = (pivot["RC4"] - pivot["RC1"]).to_numpy()
            mean_diff, hac_se, p_value = hac_mean_test(diff, args.hac_lags)

            rc1_mean = float(pivot["RC1"].mean())
            rc4_mean = float(pivot["RC4"].mean())

            within_rows.append({
                "vol_bucket": f"V{vol_rank}",
                "metric": metric,
                "rc1_mean": rc1_mean,
                "rc4_mean": rc4_mean,
                "rc4_minus_rc1": mean_diff,
                "relative_change": rc4_mean / rc1_mean - 1.0 if rc1_mean != 0 else np.nan,
                "hac_p_value_two_sided": p_value,
                "hac_lags": args.hac_lags,
                "n_dates": len(pivot),
            })
    within_df = pd.DataFrame(within_rows)

    daily.to_csv(out / "double_sort_daily_future_risk.csv", index=False)
    membership.to_csv(out / "double_sort_membership.csv", index=False)
    cells.to_csv(out / "double_sort_16cell_daily.csv", index=False)
    summary.to_csv(out / "double_sort_summary.csv", index=False)
    cell_summary.to_csv(out / "double_sort_16cell_summary.csv", index=False)
    hac_df.to_csv(out / "rc4_vs_rc1_hac.csv", index=False)
    trend_df.to_csv(out / "monotonic_trend_hac.csv", index=False)
    within_df.to_csv(out / "within_vol_bucket_rc4_vs_rc1_hac.csv", index=False)

    print("=" * 110)
    print("VOLATILITY-CONTROLLED RESIDUAL CONNECTEDNESS DOUBLE SORT")
    print("=" * 110)
    print(f"Dataset assets       : {n_assets}")
    print(f"Lookback             : {args.lookback}")
    print(f"Future horizon       : {horizon}")
    print(f"Test formation dates : {usable}")
    print(f"rho                  : {args.rho}")
    print(f"Rank mismatch dates  : {rank_mismatch_dates}")
    print()

    display_cols = [
        "rc_bucket",
        "mean_residual_connectedness_mean",
        "mean_past_asset_volatility_mean",
        "standard_cov_portfolio_volatility_mean",
        "future_variance_mean",
        "future_volatility_mean",
        "future_mdd_mean",
    ]

    print("CONTROLLED RC PORTFOLIO SUMMARY")
    print(summary[display_cols].to_string(index=False))

    print("\n" + "=" * 110)
    print("RC4 VS RC1 — PAIRED HAC AFTER VOLATILITY CONTROL")
    print("=" * 110)
    print(hac_df.to_string(index=False))

    print("\n" + "=" * 110)
    print("MONOTONIC RC1 -> RC4 TREND — HAC AFTER VOLATILITY CONTROL")
    print("=" * 110)
    print(trend_df.to_string(index=False))

    print("\n" + "=" * 110)
    print("WITHIN EACH PAST-VOLATILITY BUCKET: RC4 VS RC1")
    print("=" * 110)
    print(within_df.to_string(index=False))

    print("\nSaved:")
    for name in [
        "double_sort_daily_future_risk.csv",
        "double_sort_membership.csv",
        "double_sort_16cell_daily.csv",
        "double_sort_summary.csv",
        "double_sort_16cell_summary.csv",
        "rc4_vs_rc1_hac.csv",
        "monotonic_trend_hac.csv",
        "within_vol_bucket_rc4_vs_rc1_hac.csv",
    ]:
        print(out / name)


if __name__ == "__main__":
    main()
