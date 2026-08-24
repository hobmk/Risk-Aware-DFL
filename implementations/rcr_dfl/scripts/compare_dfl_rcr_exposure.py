from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, pearsonr, spearmanr

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DFL-MVO와 RCR-DFL의 residual collective risk exposure를 비교한다."
    )
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
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--covariance-jitter", type=float, default=1e-6)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--hac-lags", type=int, default=59)
    parser.add_argument(
        "--output-dir",
        default=(
            "implementations/rcr_dfl/outputs/"
            "dfl_vs_rcr_exposure_validation"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: str | Path, overwrite: bool) -> Path:
    output_dir = Path(path)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"출력 폴더가 이미 존재합니다: {output_dir}\n"
                "--overwrite를 추가하세요."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    return output_dir


def load_weights(run_dir: str | Path, label: str) -> pd.DataFrame:
    path = Path(run_dir) / "asset_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{label} asset_predictions.csv 없음: {path}"
        )

    frame = pd.read_csv(path)

    required = {
        "date",
        "ticker",
        "predicted_weight",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise KeyError(
            f"{label}에 필요한 column 없음: {sorted(missing)}"
        )

    frame = frame[
        [
            "date",
            "ticker",
            "predicted_weight",
        ]
    ].copy()

    frame["date"] = pd.to_datetime(frame["date"])

    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError(
            f"{label}: date-ticker 중복이 있습니다."
        )

    return frame


def make_weight_pivot(
    frame: pd.DataFrame,
    tickers: list[str],
    label: str,
) -> pd.DataFrame:
    pivot = (
        frame.pivot(
            index="date",
            columns="ticker",
            values="predicted_weight",
        )
        .sort_index()
    )

    missing = set(tickers).difference(pivot.columns)
    extra = set(pivot.columns).difference(tickers)

    if missing or extra:
        raise ValueError(
            f"{label} ticker mismatch\n"
            f"missing={sorted(missing)}\n"
            f"extra={sorted(extra)}"
        )

    pivot = pivot[tickers]

    if pivot.isna().any().any():
        raise ValueError(
            f"{label}: weight에 NaN이 있습니다."
        )

    weight_sums = pivot.sum(axis=1)

    if not np.allclose(
        weight_sums.to_numpy(),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"{label}: weight sum이 1이 아닌 날짜가 있습니다."
        )

    return pivot


def quadratic_form(
    weight: np.ndarray,
    matrix: np.ndarray,
) -> float:
    return float(
        weight @ matrix @ weight
    )


def max_drawdown(
    returns: np.ndarray,
) -> float:
    wealth = np.cumprod(
        1.0 + returns
    )

    wealth = np.concatenate(
        ([1.0], wealth)
    )

    peak = np.maximum.accumulate(
        wealth
    )

    drawdown = (
        1.0
        - wealth / peak
    )

    return float(
        drawdown.max()
    )


def hac_mean_test(
    values: np.ndarray,
    max_lag: int,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n < 3:
        return {
            "n": n,
            "mean": np.nan,
            "hac_se": np.nan,
            "hac_t": np.nan,
            "hac_p": np.nan,
        }

    mean = float(
        values.mean()
    )

    residuals = (
        values - mean
    )

    lag_max = min(
        max_lag,
        n - 1,
    )

    long_run = float(
        np.dot(
            residuals,
            residuals,
        )
    )

    for lag in range(
        1,
        lag_max + 1,
    ):
        weight = (
            1.0
            - lag
            / (lag_max + 1.0)
        )

        gamma = float(
            np.dot(
                residuals[lag:],
                residuals[:-lag],
            )
        )

        long_run += (
            2.0
            * weight
            * gamma
        )

    variance_mean = (
        long_run
        / (n * n)
    )

    variance_mean = max(
        variance_mean,
        0.0,
    )

    se = float(
        np.sqrt(
            variance_mean
        )
    )

    if se == 0.0:
        t_value = np.nan
        p_value = np.nan
    else:
        t_value = (
            mean / se
        )

        p_value = float(
            2.0
            * norm.sf(
                abs(t_value)
            )
        )

    return {
        "n": n,
        "mean": mean,
        "hac_se": se,
        "hac_t": t_value,
        "hac_p": p_value,
    }


def build_comparison(
    dataset: RCRRollingMVODataset,
    dfl_weights: pd.DataFrame,
    rcr_weights: pd.DataFrame,
    horizon: int,
    periods_per_year: int,
) -> pd.DataFrame:
    common_dates = (
        dfl_weights.index
        .intersection(
            rcr_weights.index
        )
        .intersection(
            dataset.target_dates
        )
        .sort_values()
    )

    if len(common_dates) == 0:
        raise RuntimeError(
            "DFL, RCR, dataset 사이에 공통 날짜가 없습니다."
        )

    date_to_index = {
        pd.Timestamp(date): index
        for index, date in enumerate(
            dataset.target_dates
        )
    }

    tickers = dataset.tickers
    returns = np.asarray(
        dataset.returns,
        dtype=np.float64,
    )

    rows = []

    for date in common_dates:
        dataset_index = date_to_index[
            pd.Timestamp(date)
        ]

        target_position = int(
            dataset.target_positions[
                dataset_index
            ]
        )

        sigma = (
            dataset.covariances[
                dataset_index
            ]
            .detach()
            .cpu()
            .to(torch.float64)
            .numpy()
        )

        a_res = (
            dataset.a_res_matrices[
                dataset_index
            ]
            .detach()
            .cpu()
            .to(torch.float64)
            .numpy()
        )

        c_res = (
            dataset.residual_correlations[
                dataset_index
            ]
            .detach()
            .cpu()
            .to(torch.float64)
            .numpy()
        )

        w_dfl = (
            dfl_weights
            .loc[date]
            .to_numpy(
                dtype=np.float64
            )
        )

        w_rcr = (
            rcr_weights
            .loc[date]
            .to_numpy(
                dtype=np.float64
            )
        )

        # --------------------------------------------------
        # A_res decomposition
        # --------------------------------------------------
        a_res_diag = np.diag(
            np.diag(a_res)
        )

        a_res_offdiag = (
            a_res
            - a_res_diag
        )

        a_res_positive_offdiag = np.maximum(
            a_res_offdiag,
            0.0,
        )

        # Residual correlation 자체의 positive off-diagonal
        c_res_offdiag = (
            c_res
            - np.diag(
                np.diag(c_res)
            )
        )

        c_res_positive_offdiag = np.maximum(
            c_res_offdiag,
            0.0,
        )

        # --------------------------------------------------
        # Current risk exposure
        # --------------------------------------------------
        metrics = {}

        for label, weight in [
            ("dfl", w_dfl),
            ("rcr", w_rcr),
        ]:
            metrics[
                f"{label}_sigma_exposure"
            ] = (
                quadratic_form(
                    weight,
                    sigma,
                )
                * periods_per_year
            )

            metrics[
                f"{label}_ares_total"
            ] = (
                quadratic_form(
                    weight,
                    a_res,
                )
                * periods_per_year
            )

            metrics[
                f"{label}_ares_diag"
            ] = (
                quadratic_form(
                    weight,
                    a_res_diag,
                )
                * periods_per_year
            )

            metrics[
                f"{label}_ares_offdiag"
            ] = (
                quadratic_form(
                    weight,
                    a_res_offdiag,
                )
                * periods_per_year
            )

            metrics[
                f"{label}_ares_positive_offdiag"
            ] = (
                quadratic_form(
                    weight,
                    a_res_positive_offdiag,
                )
                * periods_per_year
            )

            metrics[
                f"{label}_cres_total"
            ] = quadratic_form(
                weight,
                c_res,
            )

            metrics[
                f"{label}_cres_positive_offdiag"
            ] = quadratic_form(
                weight,
                c_res_positive_offdiag,
            )

            concentration = float(
                np.sum(
                    weight ** 2
                )
            )

            metrics[
                f"{label}_concentration"
            ] = concentration

            metrics[
                f"{label}_effective_assets"
            ] = (
                1.0 / concentration
                if concentration > 0
                else np.nan
            )

        # --------------------------------------------------
        # Future 20-day fixed-weight realized risk
        #
        # 현재 date의 weight를 미래 horizon 동안 고정하여
        # 해당 allocation의 forward risk를 측정
        # --------------------------------------------------
        future_end = (
            target_position
            + horizon
        )

        if future_end <= len(
            returns
        ):
            future_asset_returns = (
                returns[
                    target_position:
                    future_end
                ]
            )

            for label, weight in [
                ("dfl", w_dfl),
                ("rcr", w_rcr),
            ]:
                future_portfolio = (
                    future_asset_returns
                    @ weight
                )

                variance = float(
                    np.var(
                        future_portfolio,
                        ddof=1,
                    )
                    * periods_per_year
                )

                metrics[
                    f"{label}_future_variance"
                ] = variance

                metrics[
                    f"{label}_future_volatility"
                ] = float(
                    np.sqrt(
                        max(
                            variance,
                            0.0,
                        )
                    )
                )

                metrics[
                    f"{label}_future_mdd"
                ] = max_drawdown(
                    future_portfolio
                )

        else:
            for label in [
                "dfl",
                "rcr",
            ]:
                metrics[
                    f"{label}_future_variance"
                ] = np.nan

                metrics[
                    f"{label}_future_volatility"
                ] = np.nan

                metrics[
                    f"{label}_future_mdd"
                ] = np.nan

        row = {
            "date": date,
            **metrics,
        }

        rows.append(row)

    frame = pd.DataFrame(
        rows
    )

    # ------------------------------------------------------
    # RCR - DFL
    #
    # exposure/risk에서 음수면 RCR-DFL이 더 낮음
    # ------------------------------------------------------
    pair_metrics = [
        "sigma_exposure",
        "ares_total",
        "ares_diag",
        "ares_offdiag",
        "ares_positive_offdiag",
        "cres_total",
        "cres_positive_offdiag",
        "concentration",
        "effective_assets",
        "future_variance",
        "future_volatility",
        "future_mdd",
    ]

    for metric in pair_metrics:
        frame[
            f"delta_{metric}"
        ] = (
            frame[
                f"rcr_{metric}"
            ]
            - frame[
                f"dfl_{metric}"
            ]
        )

    return frame


def build_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "sigma_exposure",
        "ares_total",
        "ares_diag",
        "ares_offdiag",
        "ares_positive_offdiag",
        "cres_total",
        "cres_positive_offdiag",
        "concentration",
        "effective_assets",
        "future_variance",
        "future_volatility",
        "future_mdd",
    ]

    rows = []

    for metric in metrics:
        dfl = frame[
            f"dfl_{metric}"
        ]

        rcr = frame[
            f"rcr_{metric}"
        ]

        dfl_mean = float(
            dfl.mean()
        )

        rcr_mean = float(
            rcr.mean()
        )

        difference = (
            rcr_mean
            - dfl_mean
        )

        relative = (
            difference
            / abs(dfl_mean)
            * 100.0
            if dfl_mean != 0
            else np.nan
        )

        rows.append(
            {
                "metric": metric,
                "dfl_mean": dfl_mean,
                "rcr_mean": rcr_mean,
                "rcr_minus_dfl": difference,
                "relative_change_pct": relative,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_hac_tests(
    frame: pd.DataFrame,
    hac_lags: int,
) -> pd.DataFrame:
    metrics = [
        "sigma_exposure",
        "ares_total",
        "ares_diag",
        "ares_offdiag",
        "ares_positive_offdiag",
        "cres_total",
        "cres_positive_offdiag",
        "concentration",
        "effective_assets",
        "future_variance",
        "future_volatility",
        "future_mdd",
    ]

    rows = []

    for metric in metrics:
        test = hac_mean_test(
            frame[
                f"delta_{metric}"
            ].to_numpy(),
            max_lag=hac_lags,
        )

        rows.append(
            {
                "metric": metric,
                "comparison": "RCR - DFL",
                "negative_means_rcr_lower": (
                    metric
                    not in {
                        "effective_assets",
                    }
                ),
                **test,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_high_exposure_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    sample = frame.dropna(
        subset=[
            "dfl_ares_positive_offdiag",
            "dfl_future_volatility",
            "rcr_future_volatility",
        ]
    ).copy()

    threshold = float(
        sample[
            "dfl_ares_positive_offdiag"
        ].quantile(0.75)
    )

    high = sample.loc[
        sample[
            "dfl_ares_positive_offdiag"
        ]
        >= threshold
    ]

    all_rows = []

    for label, data in [
        ("All dates", sample),
        ("High DFL RCR exposure Q4", high),
    ]:
        all_rows.append(
            {
                "regime": label,
                "n": len(data),

                "dfl_positive_ares":
                    float(
                        data[
                            "dfl_ares_positive_offdiag"
                        ].mean()
                    ),

                "rcr_positive_ares":
                    float(
                        data[
                            "rcr_ares_positive_offdiag"
                        ].mean()
                    ),

                "positive_ares_reduction_pct":
                    float(
                        (
                            1.0
                            - data[
                                "rcr_ares_positive_offdiag"
                            ].mean()
                            / data[
                                "dfl_ares_positive_offdiag"
                            ].mean()
                        )
                        * 100.0
                    ),

                "dfl_future_vol":
                    float(
                        data[
                            "dfl_future_volatility"
                        ].mean()
                    ),

                "rcr_future_vol":
                    float(
                        data[
                            "rcr_future_volatility"
                        ].mean()
                    ),

                "future_vol_diff":
                    float(
                        data[
                            "delta_future_volatility"
                        ].mean()
                    ),

                "dfl_future_mdd":
                    float(
                        data[
                            "dfl_future_mdd"
                        ].mean()
                    ),

                "rcr_future_mdd":
                    float(
                        data[
                            "rcr_future_mdd"
                        ].mean()
                    ),

                "future_mdd_diff":
                    float(
                        data[
                            "delta_future_mdd"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        all_rows
    )


def delta_correlations(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    pairs = [
        (
            "delta_ares_total",
            "delta_future_volatility",
        ),
        (
            "delta_ares_positive_offdiag",
            "delta_future_volatility",
        ),
        (
            "delta_cres_positive_offdiag",
            "delta_future_volatility",
        ),
        (
            "delta_ares_positive_offdiag",
            "delta_future_mdd",
        ),
    ]

    rows = []

    for x, y in pairs:
        sample = frame[
            [x, y]
        ].dropna()

        pearson = pearsonr(
            sample[x],
            sample[y],
        )

        spearman = spearmanr(
            sample[x],
            sample[y],
        )

        rows.append(
            {
                "x": x,
                "y": y,
                "n": len(sample),
                "pearson_r":
                    float(
                        pearson.statistic
                    ),
                "pearson_p":
                    float(
                        pearson.pvalue
                    ),
                "spearman_rho":
                    float(
                        spearman.statistic
                    ),
                "spearman_p":
                    float(
                        spearman.pvalue
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    args = parse_args()

    output_dir = prepare_output_dir(
        args.output_dir,
        args.overwrite,
    )

    dfl_assets = load_weights(
        args.dfl_run_dir,
        "DFL-MVO",
    )

    rcr_assets = load_weights(
        args.rcr_run_dir,
        "RCR-DFL",
    )

    dataset = RCRRollingMVODataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type="simple",
        covariance_jitter=
            args.covariance_jitter,
        market_mode="equal_weight",
        risk_free_rate=
            args.risk_free_rate,
        fit_intercept=True,
        residual_correlation_shrinkage=
            args.rho,
        correlation_scaling="trace",
        dtype=torch.float32,
    )

    dfl_weights = make_weight_pivot(
        dfl_assets,
        dataset.tickers,
        "DFL-MVO",
    )

    rcr_weights = make_weight_pivot(
        rcr_assets,
        dataset.tickers,
        "RCR-DFL",
    )

    frame = build_comparison(
        dataset=dataset,
        dfl_weights=dfl_weights,
        rcr_weights=rcr_weights,
        horizon=args.horizon,
        periods_per_year=
            args.periods_per_year,
    )

    summary = build_summary(
        frame
    )

    hac_tests = build_hac_tests(
        frame,
        hac_lags=args.hac_lags,
    )

    high_exposure = (
        build_high_exposure_summary(
            frame
        )
    )

    correlations = (
        delta_correlations(
            frame
        )
    )

    frame.to_csv(
        output_dir
        / "exposure_comparison_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        output_dir
        / "exposure_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hac_tests.to_csv(
        output_dir
        / "paired_hac_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    high_exposure.to_csv(
        output_dir
        / "high_exposure_regime_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    correlations.to_csv(
        output_dir
        / "delta_exposure_future_risk_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = vars(
        args
    ).copy()

    config.update(
        {
            "n_assets":
                dataset.n_assets,
            "common_dates":
                len(frame),
            "first_date":
                str(
                    frame[
                        "date"
                    ].min().date()
                ),
            "last_date":
                str(
                    frame[
                        "date"
                    ].max().date()
                ),
        }
    )

    with (
        output_dir
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ======================================================
    # Terminal
    # ======================================================
    print(
        "\n" + "=" * 110
    )
    print(
        "DFL-MVO vs RCR-DFL RESIDUAL EXPOSURE VALIDATION"
    )
    print(
        "=" * 110
    )

    print(
        f"assets      : {dataset.n_assets}"
    )
    print(
        f"common dates: {len(frame)}"
    )
    print(
        f"date range  : "
        f"{frame['date'].min().date()} "
        f"~ "
        f"{frame['date'].max().date()}"
    )
    print(
        f"lookback    : {args.lookback}"
    )
    print(
        f"horizon     : {args.horizon}"
    )
    print(
        f"rho         : {args.rho}"
    )
    print(
        f"HAC lags    : {args.hac_lags}"
    )

    print(
        "\n" + "=" * 110
    )
    print(
        "MEAN EXPOSURE COMPARISON"
    )
    print(
        "RCR - DFL < 0 means RCR-DFL has lower exposure "
        "(except effective_assets)"
    )
    print(
        "=" * 110
    )

    key_metrics = [
        "sigma_exposure",
        "ares_total",
        "ares_diag",
        "ares_offdiag",
        "ares_positive_offdiag",
        "cres_positive_offdiag",
        "concentration",
        "effective_assets",
    ]

    print(
        summary.loc[
            summary[
                "metric"
            ].isin(
                key_metrics
            )
        ]
        .round(8)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 110
    )
    print(
        "PAIRED HAC TESTS"
    )
    print(
        "=" * 110
    )

    print(
        hac_tests.loc[
            hac_tests[
                "metric"
            ].isin(
                key_metrics
            ),
            [
                "metric",
                "n",
                "mean",
                "hac_se",
                "hac_t",
                "hac_p",
            ],
        ]
        .round(8)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 110
    )
    print(
        "FUTURE 20-DAY FIXED-WEIGHT RISK"
    )
    print(
        "RCR - DFL < 0 means lower future risk"
    )
    print(
        "=" * 110
    )

    risk_metrics = [
        "future_variance",
        "future_volatility",
        "future_mdd",
    ]

    print(
        summary.loc[
            summary[
                "metric"
            ].isin(
                risk_metrics
            )
        ]
        .round(8)
        .to_string(index=False)
    )

    print(
        "\nPaired HAC:"
    )

    print(
        hac_tests.loc[
            hac_tests[
                "metric"
            ].isin(
                risk_metrics
            ),
            [
                "metric",
                "n",
                "mean",
                "hac_se",
                "hac_t",
                "hac_p",
            ],
        ]
        .round(8)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 110
    )
    print(
        "HIGH DFL RESIDUAL-EXPOSURE REGIME"
    )
    print(
        "=" * 110
    )

    print(
        high_exposure
        .round(8)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 110
    )
    print(
        "DELTA EXPOSURE vs DELTA FUTURE RISK"
    )
    print(
        "=" * 110
    )

    print(
        correlations
        .round(8)
        .to_string(index=False)
    )

    print(
        "\nSaved:"
    )

    for filename in [
        "exposure_comparison_timeseries.csv",
        "exposure_comparison_summary.csv",
        "paired_hac_tests.csv",
        "high_exposure_regime_summary.csv",
        "delta_exposure_future_risk_correlations.csv",
    ]:
        print(
            output_dir / filename
        )


if __name__ == "__main__":
    main()