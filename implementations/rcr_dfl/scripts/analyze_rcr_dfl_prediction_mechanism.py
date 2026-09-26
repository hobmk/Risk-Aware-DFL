from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, pearsonr, spearmanr

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dfl-run-dir",
        default=(
            "implementations/dfl_mvo_lee2025/outputs/combined/dow30/"
            "final_full_grid_h64_d0_s30_std/"
            "alpha_0.50/lambda_1.00/maxw_0.20/seed_42"
        ),
    )

    parser.add_argument(
        "--rcr-run-dir",
        default=(
            "implementations/rcr_dfl/outputs/"
            "eta_sensitivity_maxw020_standardized/eta_0.50"
        ),
    )

    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--hac-lags", type=int, default=19)

    parser.add_argument(
        "--output-dir",
        default=(
            "implementations/rcr_dfl/outputs/"
            "rcr_dfl_prediction_mechanism_seed42"
        ),
    )

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def prepare_output_dir(path, overwrite):
    path = Path(path)

    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"출력 폴더가 이미 존재합니다: {path}\n"
                "--overwrite를 추가하세요."
            )

        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


def load_predictions(run_dir, label):
    path = Path(run_dir) / "asset_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{label} asset_predictions.csv 없음:\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "date",
        "ticker",
        "true_return",
        "predicted_return",
        "predicted_weight",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"{label} missing columns: {sorted(missing)}"
        )

    df = df[
        [
            "date",
            "ticker",
            "true_return",
            "predicted_return",
            "predicted_weight",
        ]
    ].copy()

    df["date"] = pd.to_datetime(df["date"])

    for col in [
        "true_return",
        "predicted_return",
        "predicted_weight",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(
            f"{label}: duplicate date-ticker rows"
        )

    return df


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(
        pearsonr(x, y).statistic
    )


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan

    return float(
        spearmanr(x, y).statistic
    )


def hac_mean_test(values, max_lag):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n < 3:
        return {
            "n": n,
            "mean": np.nan,
            "hac_se": np.nan,
            "hac_t": np.nan,
            "hac_p": np.nan,
        }

    mean = float(values.mean())
    residuals = values - mean

    lag_max = min(
        max_lag,
        n - 1,
    )

    long_run = float(
        np.dot(residuals, residuals)
    )

    for lag in range(1, lag_max + 1):
        bartlett = 1.0 - lag / (lag_max + 1.0)

        gamma = float(
            np.dot(
                residuals[lag:],
                residuals[:-lag],
            )
        )

        long_run += (
            2.0
            * bartlett
            * gamma
        )

    variance_mean = max(
        long_run / (n * n),
        0.0,
    )

    se = float(
        np.sqrt(variance_mean)
    )

    if se == 0:
        t_value = np.nan
        p_value = np.nan
    else:
        t_value = mean / se
        p_value = float(
            2.0 * norm.sf(abs(t_value))
        )

    return {
        "n": n,
        "mean": mean,
        "hac_se": se,
        "hac_t": t_value,
        "hac_p": p_value,
    }


def build_residual_centrality(
    dataset,
    common_dates,
    periods_per_year,
):
    date_to_index = {
        pd.Timestamp(date): i
        for i, date in enumerate(
            dataset.target_dates
        )
    }

    rows = []

    for date in common_dates:
        date = pd.Timestamp(date)

        if date not in date_to_index:
            continue

        idx = date_to_index[date]

        a_res = (
            dataset.a_res_matrices[idx]
            .detach()
            .cpu()
            .to(torch.float64)
            .numpy()
        )

        diag = np.diag(
            np.diag(a_res)
        )

        offdiag = (
            a_res - diag
        )

        positive = np.maximum(
            offdiag,
            0.0,
        )

        positive_centrality = (
            positive.sum(axis=1)
            * periods_per_year
        )

        absolute_centrality = (
            np.abs(offdiag).sum(axis=1)
            * periods_per_year
        )

        signed_centrality = (
            offdiag.sum(axis=1)
            * periods_per_year
        )

        for ticker, c_pos, c_abs, c_signed in zip(
            dataset.tickers,
            positive_centrality,
            absolute_centrality,
            signed_centrality,
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "residual_centrality_pos": float(c_pos),
                    "residual_centrality_abs": float(c_abs),
                    "residual_centrality_signed": float(c_signed),
                }
            )

    return pd.DataFrame(rows)


def merge_models(dfl, rcr):
    dfl = dfl.rename(
        columns={
            "true_return": "true_return_dfl",
            "predicted_return": "pred_dfl",
            "predicted_weight": "weight_dfl",
        }
    )

    rcr = rcr.rename(
        columns={
            "true_return": "true_return_rcr",
            "predicted_return": "pred_rcr",
            "predicted_weight": "weight_rcr",
        }
    )

    frame = dfl.merge(
        rcr,
        on=["date", "ticker"],
        how="inner",
        validate="one_to_one",
    )

    true_diff = np.abs(
        frame["true_return_dfl"]
        - frame["true_return_rcr"]
    )

    print(
        "Max true-return difference:",
        f"{true_diff.max():.12f}",
    )

    if true_diff.max() > 1e-8:
        raise ValueError(
            "DFL과 RCR true_return이 일치하지 않습니다."
        )

    frame["true_return"] = (
        frame["true_return_dfl"]
        + frame["true_return_rcr"]
    ) / 2.0

    frame = frame.drop(
        columns=[
            "true_return_dfl",
            "true_return_rcr",
        ]
    )

    return frame


def add_asset_metrics(frame):
    df = frame.copy()

    df["pred_diff"] = (
        df["pred_rcr"]
        - df["pred_dfl"]
    )

    df["abs_pred_diff"] = (
        df["pred_diff"].abs()
    )

    df["weight_diff"] = (
        df["weight_rcr"]
        - df["weight_dfl"]
    )

    df["abs_weight_diff"] = (
        df["weight_diff"].abs()
    )

    df["abs_error_dfl"] = (
        df["pred_dfl"]
        - df["true_return"]
    ).abs()

    df["abs_error_rcr"] = (
        df["pred_rcr"]
        - df["true_return"]
    ).abs()

    # 양수 = RCR의 절대예측오차가 더 작음
    df["error_improvement"] = (
        df["abs_error_dfl"]
        - df["abs_error_rcr"]
    )

    # 같은 날짜 내에서 residual centrality가 몇 번째로 높은지
    df["centrality_rank"] = (
        df.groupby("date")[
            "residual_centrality_pos"
        ]
        .rank(
            method="first",
            ascending=True,
        )
    )

    df["centrality_pct_rank"] = (
        df.groupby("date")[
            "residual_centrality_pos"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    return df


def build_daily_metrics(frame):
    rows = []

    for date, x in frame.groupby("date"):
        rows.append(
            {
                "date": date,
                "n_assets": len(x),

                # 1. DFL vs RCR 예측값 자체의 유사도
                "pred_pearson": safe_pearson(
                    x["pred_dfl"],
                    x["pred_rcr"],
                ),

                "pred_spearman": safe_spearman(
                    x["pred_dfl"],
                    x["pred_rcr"],
                ),

                # 2. Centrality가 높을수록 prediction 차이가 큰가?
                "centrality_abs_pred_spearman": safe_spearman(
                    x["residual_centrality_pos"],
                    x["abs_pred_diff"],
                ),

                # 3. Centrality가 높을수록 어느 방향으로 prediction이 바뀌는가?
                "centrality_signed_pred_spearman": safe_spearman(
                    x["residual_centrality_pos"],
                    x["pred_diff"],
                ),

                # 4. Centrality가 높을수록 portfolio weight가 더 바뀌는가?
                "centrality_abs_weight_spearman": safe_spearman(
                    x["residual_centrality_pos"],
                    x["abs_weight_diff"],
                ),

                # 5. Prediction 변화가 클수록 실제 weight 변화도 큰가?
                "abs_pred_abs_weight_spearman": safe_spearman(
                    x["abs_pred_diff"],
                    x["abs_weight_diff"],
                ),

                # 6. Centrality가 높은 종목에서 RCR prediction error가 개선되는가?
                "centrality_error_improvement_spearman": safe_spearman(
                    x["residual_centrality_pos"],
                    x["error_improvement"],
                ),

                "mean_abs_pred_diff": float(
                    x["abs_pred_diff"].mean()
                ),

                "mean_abs_weight_diff": float(
                    x["abs_weight_diff"].mean()
                ),

                "mean_error_improvement": float(
                    x["error_improvement"].mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("date")


def build_daily_summary(daily):
    metrics = [
        "pred_pearson",
        "pred_spearman",
        "centrality_abs_pred_spearman",
        "centrality_signed_pred_spearman",
        "centrality_abs_weight_spearman",
        "abs_pred_abs_weight_spearman",
        "centrality_error_improvement_spearman",
        "mean_abs_pred_diff",
        "mean_abs_weight_diff",
        "mean_error_improvement",
    ]

    rows = []

    for metric in metrics:
        x = daily[metric].dropna()

        rows.append(
            {
                "metric": metric,
                "n": len(x),
                "mean": x.mean(),
                "std": x.std(ddof=1),
                "median": x.median(),
                "q25": x.quantile(0.25),
                "q75": x.quantile(0.75),
                "min": x.min(),
                "max": x.max(),
            }
        )

    return pd.DataFrame(rows)


def build_hac_tests(daily, hac_lags):
    metrics = [
        "centrality_abs_pred_spearman",
        "centrality_signed_pred_spearman",
        "centrality_abs_weight_spearman",
        "abs_pred_abs_weight_spearman",
        "centrality_error_improvement_spearman",
        "mean_error_improvement",
    ]

    rows = []

    for metric in metrics:
        result = hac_mean_test(
            daily[metric].to_numpy(),
            hac_lags,
        )

        rows.append(
            {
                "metric": metric,
                **result,
            }
        )

    return pd.DataFrame(rows)


def build_pooled_correlations(frame):
    pairs = [
        (
            "centrality_pct_rank",
            "abs_pred_diff",
            "Centrality rank",
            "|RCR prediction - DFL prediction|",
        ),
        (
            "centrality_pct_rank",
            "pred_diff",
            "Centrality rank",
            "RCR prediction - DFL prediction",
        ),
        (
            "centrality_pct_rank",
            "abs_weight_diff",
            "Centrality rank",
            "|RCR weight - DFL weight|",
        ),
        (
            "abs_pred_diff",
            "abs_weight_diff",
            "|Prediction difference|",
            "|Weight difference|",
        ),
        (
            "centrality_pct_rank",
            "error_improvement",
            "Centrality rank",
            "DFL absolute error - RCR absolute error",
        ),
    ]

    rows = []

    for x_col, y_col, x_name, y_name in pairs:
        x = frame[[x_col, y_col]].dropna()

        pearson = pearsonr(
            x[x_col],
            x[y_col],
        )

        spearman = spearmanr(
            x[x_col],
            x[y_col],
        )

        rows.append(
            {
                "x": x_name,
                "y": y_name,
                "n": len(x),
                "pearson_r": float(
                    pearson.statistic
                ),
                "pearson_p": float(
                    pearson.pvalue
                ),
                "spearman_rho": float(
                    spearman.statistic
                ),
                "spearman_p": float(
                    spearman.pvalue
                ),
            }
        )

    return pd.DataFrame(rows)


def build_rank_profile(frame):
    profile = (
        frame.groupby(
            "centrality_rank",
            as_index=False,
        )
        .agg(
            mean_centrality=(
                "residual_centrality_pos",
                "mean",
            ),
            mean_abs_pred_diff=(
                "abs_pred_diff",
                "mean",
            ),
            mean_signed_pred_diff=(
                "pred_diff",
                "mean",
            ),
            mean_abs_weight_diff=(
                "abs_weight_diff",
                "mean",
            ),
            mean_error_improvement=(
                "error_improvement",
                "mean",
            ),
            dfl_mae=(
                "abs_error_dfl",
                "mean",
            ),
            rcr_mae=(
                "abs_error_rcr",
                "mean",
            ),
            n=(
                "ticker",
                "size",
            ),
        )
        .sort_values(
            "centrality_rank"
        )
    )

    return profile


def plot_daily_prediction_correlation(daily, output_dir):
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        daily["date"],
        daily["pred_spearman"],
        linewidth=1.2,
    )

    ax.axhline(
        daily["pred_spearman"].mean(),
        linestyle="--",
        label=(
            "Mean = "
            f"{daily['pred_spearman'].mean():.3f}"
        ),
    )

    ax.set_title(
        "DFL-MVO vs RCR-DFL Daily Cross-Sectional Prediction Correlation"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Spearman Correlation")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "daily_prediction_cross_sectional_correlation.png",
        dpi=300,
    )

    plt.close(fig)


def plot_rank_profile(profile, output_dir):
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        profile["centrality_rank"],
        profile["mean_abs_pred_diff"],
        marker="o",
        markersize=3,
    )

    ax.set_title(
        "Residual Centrality Rank vs Prediction Difference"
    )
    ax.set_xlabel(
        "Residual Centrality Rank (Low -> High)"
    )
    ax.set_ylabel(
        "Mean |RCR Prediction - DFL Prediction|"
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "centrality_rank_vs_prediction_difference.png",
        dpi=300,
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        profile["centrality_rank"],
        profile["mean_abs_weight_diff"],
        marker="o",
        markersize=3,
    )

    ax.set_title(
        "Residual Centrality Rank vs Portfolio Weight Difference"
    )
    ax.set_xlabel(
        "Residual Centrality Rank (Low -> High)"
    )
    ax.set_ylabel(
        "Mean |RCR Weight - DFL Weight|"
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "centrality_rank_vs_weight_difference.png",
        dpi=300,
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        profile["centrality_rank"],
        profile["mean_error_improvement"],
        marker="o",
        markersize=3,
    )

    ax.axhline(
        0.0,
        linestyle="--",
    )

    ax.set_title(
        "Residual Centrality Rank vs Prediction Error Improvement"
    )
    ax.set_xlabel(
        "Residual Centrality Rank (Low -> High)"
    )
    ax.set_ylabel(
        "DFL Absolute Error - RCR Absolute Error"
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "centrality_rank_vs_prediction_error_improvement.png",
        dpi=300,
    )

    plt.close(fig)


def main():
    args = parse_args()

    output_dir = prepare_output_dir(
        args.output_dir,
        args.overwrite,
    )

    print("=" * 90)
    print("RCR-DFL vs DFL-MVO PREDICTION MECHANISM ANALYSIS")
    print("=" * 90)

    print("\n[1] Loading predictions...")

    dfl = load_predictions(
        args.dfl_run_dir,
        "DFL-MVO",
    )

    rcr = load_predictions(
        args.rcr_run_dir,
        "RCR-DFL",
    )

    frame = merge_models(
        dfl,
        rcr,
    )

    print(
        f"Matched asset observations: {len(frame)}"
    )

    print(
        f"Matched dates: {frame['date'].nunique()}"
    )

    print(
        f"Matched assets: {frame['ticker'].nunique()}"
    )

    print("\n[2] Building A_res residual centrality...")

    dataset = RCRRollingMVODataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        residual_correlation_shrinkage=args.rho,
        correlation_scaling="trace",
        dtype=torch.float32,
    )

    common_dates = sorted(
        frame["date"].unique()
    )

    centrality = build_residual_centrality(
        dataset,
        common_dates,
        args.periods_per_year,
    )

    frame = frame.merge(
        centrality,
        on=["date", "ticker"],
        how="inner",
        validate="one_to_one",
    )

    frame = add_asset_metrics(
        frame
    )

    print(
        f"Centrality matched observations: {len(frame)}"
    )

    print("\n[3] Daily cross-sectional analysis...")

    daily = build_daily_metrics(
        frame
    )

    daily_summary = build_daily_summary(
        daily
    )

    hac_tests = build_hac_tests(
        daily,
        args.hac_lags,
    )

    pooled = build_pooled_correlations(
        frame
    )

    rank_profile = build_rank_profile(
        frame
    )

    print("\n[4] Saving outputs...")

    frame.to_csv(
        output_dir
        / "asset_level_mechanism.csv",
        index=False,
        encoding="utf-8-sig",
    )

    daily.to_csv(
        output_dir
        / "daily_cross_sectional_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    daily_summary.to_csv(
        output_dir
        / "daily_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hac_tests.to_csv(
        output_dir
        / "daily_spearman_hac_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled.to_csv(
        output_dir
        / "pooled_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rank_profile.to_csv(
        output_dir
        / "centrality_rank_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_daily_prediction_correlation(
        daily,
        output_dir,
    )

    plot_rank_profile(
        rank_profile,
        output_dir,
    )

    print("\n" + "=" * 90)
    print("DAILY CROSS-SECTIONAL SUMMARY")
    print("=" * 90)

    print(
        daily_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 90)
    print("HAC TESTS")
    print("=" * 90)

    print(
        hac_tests.to_string(
            index=False
        )
    )

    print("\n" + "=" * 90)
    print("POOLED CORRELATIONS")
    print("=" * 90)

    print(
        pooled.to_string(
            index=False
        )
    )

    print("\nSaved to:")
    print(output_dir)


if __name__ == "__main__":
    main()