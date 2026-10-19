from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank-Tail exposure -> future risk validation"
    )

    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--lookback", type=int, default=60)

    parser.add_argument("--tail-q", type=float, default=0.35)

    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--hac-lags", type=int, default=19)
    parser.add_argument("--periods-per-year", type=int, default=252)

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )

    parser.add_argument(
        "--baseline-root",
        default=(
            "implementations/"
            "dfl_mvo_lee2025/"
            "outputs/combined/dow30/"
            "final_full_grid_h64_d0_s30_std/"
            "alpha_0.50/"
            "lambda_1.00/"
            "maxw_0.20"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    return parser.parse_args()


def max_drawdown(
    returns: np.ndarray,
) -> float:
    returns = np.asarray(
        returns,
        dtype=np.float64,
    )

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
        wealth / peak - 1.0
    )

    return float(
        -drawdown.min()
    )


def future_fixed_weight_risk(
    weights: np.ndarray,
    returns: np.ndarray,
    horizon: int,
    periods_per_year: int,
):
    n_dates = returns.shape[0]

    future_variance = np.full(
        n_dates,
        np.nan,
        dtype=np.float64,
    )

    future_volatility = np.full(
        n_dates,
        np.nan,
        dtype=np.float64,
    )

    future_mdd = np.full(
        n_dates,
        np.nan,
        dtype=np.float64,
    )

    n_valid = (
        n_dates - horizon + 1
    )

    for t in range(n_valid):
        future_returns = (
            returns[t:t + horizon]
        )

        # t 시점의 포트폴리오 weight를
        # 향후 20일 동안 고정해서 위험 측정
        portfolio_returns = (
            future_returns @ weights[t]
        )

        variance = (
            np.var(
                portfolio_returns,
                ddof=1,
            )
            * periods_per_year
        )

        future_variance[t] = (
            variance
        )

        future_volatility[t] = (
            np.sqrt(
                max(
                    variance,
                    0.0,
                )
            )
        )

        future_mdd[t] = (
            max_drawdown(
                portfolio_returns
            )
        )

    return (
        future_variance,
        future_volatility,
        future_mdd,
    )


def load_baseline(
    baseline_root: Path,
    seed: int,
    tickers: list[str],
):
    path = (
        baseline_root
        / f"seed_{seed}"
        / "asset_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Baseline result not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    required_columns = {
        "date",
        "ticker",
        "predicted_weight",
        "true_return",
    }

    missing = (
        required_columns
        .difference(
            dataframe.columns
        )
    )

    if missing:
        raise KeyError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    dataframe["date"] = (
        pd.to_datetime(
            dataframe["date"]
        )
    )

    dataframe["ticker"] = (
        dataframe["ticker"]
        .astype(str)
    )

    weights = (
        dataframe
        .pivot(
            index="date",
            columns="ticker",
            values="predicted_weight",
        )
        .sort_index()
        .reindex(
            columns=tickers
        )
    )

    returns = (
        dataframe
        .pivot(
            index="date",
            columns="ticker",
            values="true_return",
        )
        .sort_index()
        .reindex(
            columns=tickers
        )
    )

    if (
        weights.isna().any().any()
        or returns.isna().any().any()
    ):
        raise RuntimeError(
            f"NaN after pivot for seed {seed}"
        )

    return (
        pd.DatetimeIndex(
            weights.index
        ),
        weights.to_numpy(
            dtype=np.float64
        ),
        returns.to_numpy(
            dtype=np.float64
        ),
    )


def build_matrix_map(
    dataset: RankTailRCRDataset,
):
    matrix_map = {}

    for index in range(
        len(dataset)
    ):
        item = dataset[index]

        date = pd.Timestamp(
            item["target_date"]
        ).normalize()

        matrix = (
            dataset
            .rank_tail_risk[index]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=False,
            )
        )

        matrix_map[date] = matrix

    return matrix_map


def quadratic_exposure(
    weights: np.ndarray,
    matrices: np.ndarray,
):
    return np.einsum(
        "bi,bij,bj->b",
        weights,
        matrices,
        weights,
        optimize=True,
    )


def compute_exposures(
    weights: np.ndarray,
    matrices: np.ndarray,
):
    n_assets = (
        matrices.shape[1]
    )

    diagonal = np.zeros_like(
        matrices
    )

    diagonal_values = (
        np.diagonal(
            matrices,
            axis1=1,
            axis2=2,
        )
    )

    indices = np.arange(
        n_assets
    )

    diagonal[
        :,
        indices,
        indices,
    ] = diagonal_values

    offdiag = (
        matrices - diagonal
    )

    # 분석용 diagnostic.
    # 최적화 matrix를 이렇게 바꾸는 것이 아님.
    positive_offdiag = (
        np.maximum(
            offdiag,
            0.0,
        )
    )

    return {
        "rt_total_exposure":
            quadratic_exposure(
                weights,
                matrices,
            ),

        "residual_diag_exposure":
            quadratic_exposure(
                weights,
                diagonal,
            ),

        "rt_offdiag_exposure":
            quadratic_exposure(
                weights,
                offdiag,
            ),

        "rt_positive_offdiag_exposure":
            quadratic_exposure(
                weights,
                positive_offdiag,
            ),
    }


def hac_test(
    dataframe: pd.DataFrame,
    exposure_column: str,
    future_column: str,
    hac_lags: int,
):
    work = dataframe[
        [
            exposure_column,
            future_column,
        ]
    ].dropna().copy()

    threshold = float(
        work[
            exposure_column
        ].quantile(0.75)
    )

    work["high_q4"] = (
        work[exposure_column]
        >= threshold
    ).astype(float)

    low = work.loc[
        work["high_q4"] == 0,
        future_column,
    ]

    high = work.loc[
        work["high_q4"] == 1,
        future_column,
    ]

    low_mean = float(
        low.mean()
    )

    high_mean = float(
        high.mean()
    )

    if abs(low_mean) > 1e-15:
        increase_pct = (
            (
                high_mean
                / low_mean
                - 1.0
            )
            * 100.0
        )
    else:
        increase_pct = np.nan

    # Q4 dummy HAC regression
    X = sm.add_constant(
        work["high_q4"]
    )

    model = sm.OLS(
        work[
            future_column
        ].to_numpy(),
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": hac_lags
        },
    )

    q4_beta = float(
        model.params["high_q4"]
    )

    q4_se = float(
        model.bse["high_q4"]
    )

    q4_t = float(
        model.tvalues["high_q4"]
    )

    q4_p = float(
        model.pvalues["high_q4"]
    )

    # Continuous exposure도 같이 확인.
    x = work[
        exposure_column
    ].to_numpy(
        dtype=np.float64
    )

    x_std = float(
        x.std(ddof=1)
    )

    if x_std > 0:
        z = (
            x - x.mean()
        ) / x_std

        X_cont = sm.add_constant(
            z
        )

        continuous = sm.OLS(
            work[
                future_column
            ].to_numpy(
                dtype=np.float64
            ),
            X_cont,
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": hac_lags
            },
        )

        beta_1sd = float(
            continuous.params[1]
        )

        p_1sd = float(
            continuous.pvalues[1]
        )

    else:
        beta_1sd = np.nan
        p_1sd = np.nan

    return {
        "n": len(work),

        "q4_threshold":
            threshold,

        "q1_q3_mean":
            low_mean,

        "q4_mean":
            high_mean,

        "q4_minus_q1_q3":
            high_mean - low_mean,

        "increase_pct":
            increase_pct,

        "hac_beta_q4":
            q4_beta,

        "hac_se_q4":
            q4_se,

        "hac_t_q4":
            q4_t,

        "hac_p_q4":
            q4_p,

        "hac_beta_per_1sd_exposure":
            beta_1sd,

        "hac_p_per_1sd_exposure":
            p_1sd,
    }


def main():
    args = parse_args()

    if not (
        0.0 < args.tail_q < 1.0
    ):
        raise ValueError(
            "--tail-q must be in (0, 1)"
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            Path(
                "implementations/"
                "rank_tail_rcr_dfl/"
                "outputs/mechanism"
            )
            / f"q_{args.tail_q:.2f}"
            / "step1_future_risk"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 100
    )
    print(
        "RANK-TAIL STEP 1: "
        "EXPOSURE -> FUTURE RISK"
    )
    print(
        "=" * 100
    )

    print(
        f"Tail q    : "
        f"{args.tail_q:.2f}"
    )

    print(
        f"Horizon   : "
        f"{args.horizon}"
    )

    print(
        f"HAC lags  : "
        f"{args.hac_lags}"
    )

    print(
        f"Seeds     : "
        f"{args.seeds}"
    )

    # ---------------------------------------------------------
    # Rank-Tail matrix 생성
    # ---------------------------------------------------------

    dataset = RankTailRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,

        return_type="simple",
        covariance_jitter=1e-6,

        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,

        tail_q=args.tail_q,
    )

    tickers = [
        str(ticker)
        for ticker in dataset.tickers
    ]

    matrix_map = (
        build_matrix_map(
            dataset
        )
    )

    baseline_root = Path(
        args.baseline_root
    )

    daily_frames = []
    test_rows = []

    exposure_columns = [
        "rt_total_exposure",
        "rt_offdiag_exposure",
        "rt_positive_offdiag_exposure",
        "residual_diag_exposure",
    ]

    future_columns = [
        "future_variance",
        "future_volatility",
        "future_mdd",
    ]

    # ---------------------------------------------------------
    # Seed별 검증
    # ---------------------------------------------------------

    for seed in args.seeds:
        (
            dates,
            weights,
            returns,
        ) = load_baseline(
            baseline_root,
            seed,
            tickers,
        )

        normalized_dates = (
            dates.normalize()
        )

        missing_dates = [
            date
            for date in normalized_dates
            if date not in matrix_map
        ]

        if missing_dates:
            raise KeyError(
                "Rank-Tail matrix date "
                f"missing: {missing_dates[0]}"
            )

        matrices = np.stack(
            [
                matrix_map[date]
                for date
                in normalized_dates
            ],
            axis=0,
        )

        exposures = (
            compute_exposures(
                weights,
                matrices,
            )
        )

        (
            future_variance,
            future_volatility,
            future_mdd,
        ) = future_fixed_weight_risk(
            weights=weights,
            returns=returns,
            horizon=args.horizon,
            periods_per_year=(
                args.periods_per_year
            ),
        )

        daily = pd.DataFrame(
            {
                "seed": seed,
                "date": dates,

                **exposures,

                "future_variance":
                    future_variance,

                "future_volatility":
                    future_volatility,

                "future_mdd":
                    future_mdd,
            }
        )

        daily_frames.append(
            daily
        )

        for exposure_column in (
            exposure_columns
        ):
            for future_column in (
                future_columns
            ):
                result = hac_test(
                    dataframe=daily,
                    exposure_column=(
                        exposure_column
                    ),
                    future_column=(
                        future_column
                    ),
                    hac_lags=(
                        args.hac_lags
                    ),
                )

                test_rows.append(
                    {
                        "seed": seed,
                        "tail_q":
                            args.tail_q,

                        "exposure_metric":
                            exposure_column,

                        "future_metric":
                            future_column,

                        **result,
                    }
                )

        valid_windows = int(
            np.isfinite(
                future_variance
            ).sum()
        )

        print(
            f"seed {seed} | "
            f"dates={len(dates)} | "
            f"future windows="
            f"{valid_windows}"
        )

    # ---------------------------------------------------------
    # 결과 저장
    # ---------------------------------------------------------

    daily_all = pd.concat(
        daily_frames,
        ignore_index=True,
    )

    tests = pd.DataFrame(
        test_rows
    )

    significant = (
        (
            tests["hac_beta_q4"] > 0
        )
        & (
            tests["hac_p_q4"] < 0.05
        )
    ).astype(int)

    tests[
        "significant_positive"
    ] = significant

    summary = (
        tests
        .groupby(
            [
                "tail_q",
                "exposure_metric",
                "future_metric",
            ],
            as_index=False,
        )
        .agg(
            seeds=(
                "seed",
                "nunique",
            ),

            mean_increase_pct=(
                "increase_pct",
                "mean",
            ),

            std_increase_pct=(
                "increase_pct",
                "std",
            ),

            median_increase_pct=(
                "increase_pct",
                "median",
            ),

            positive_direction_seeds=(
                "increase_pct",
                lambda x:
                    int(
                        (x > 0).sum()
                    ),
            ),

            significant_positive_seeds=(
                "significant_positive",
                "sum",
            ),

            mean_hac_beta_q4=(
                "hac_beta_q4",
                "mean",
            ),

            mean_hac_beta_per_1sd=(
                "hac_beta_per_1sd_exposure",
                "mean",
            ),
        )
    )

    daily_path = (
        output_dir
        / "rank_tail_exposure_future_risk_daily.csv"
    )

    test_path = (
        output_dir
        / "rank_tail_exposure_future_risk_by_seed.csv"
    )

    summary_path = (
        output_dir
        / "rank_tail_exposure_future_risk_summary.csv"
    )

    daily_all.to_csv(
        daily_path,
        index=False,
    )

    tests.to_csv(
        test_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 화면에는 핵심 Rank-Tail dependence만 출력
    # ---------------------------------------------------------

    primary = summary[
        summary[
            "exposure_metric"
        ].isin(
            [
                "rt_offdiag_exposure",
                "rt_positive_offdiag_exposure",
            ]
        )
    ]

    print()
    print(
        "=" * 100
    )
    print(
        "PRIMARY RANK-TAIL "
        "DEPENDENCE RESULTS"
    )
    print(
        "=" * 100
    )

    print(
        primary[
            [
                "exposure_metric",
                "future_metric",
                "mean_increase_pct",
                "std_increase_pct",
                "positive_direction_seeds",
                "significant_positive_seeds",
                "mean_hac_beta_q4",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(daily_path)
    print(test_path)
    print(summary_path)


if __name__ == "__main__":
    main()