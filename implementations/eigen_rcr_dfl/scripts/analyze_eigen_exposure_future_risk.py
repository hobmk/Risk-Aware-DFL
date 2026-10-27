from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from implementations.eigen_rcr_dfl.src.dataset import EigenRCRDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment A: Top/Bottom/Full residual "
            "eigenspace exposure -> future risk"
        )
    )

    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument("--date-column", default="Date")
    parser.add_argument(
        "--return-type",
        choices=["simple", "log"],
        default="simple",
    )
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument(
        "--covariance-jitter",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--market-mode",
        choices=["equal_weight", "external"],
        default="equal_weight",
    )
    parser.add_argument("--market-price-csv", default=None)
    parser.add_argument("--market-column", default=None)
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--no-capm-intercept",
        action="store_true",
    )

    parser.add_argument(
        "--validation-end",
        default="2022-12-31",
    )

    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
    )

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[25, 50, 100, 200],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(42, 52)),
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
    )

    parser.add_argument(
        "--hac-lags",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--baseline-root",
        default=(
            "implementations/"
            "dfl_mvo_lee2025/"
            "outputs/"
            "lambda_multiseed_standardized"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "implementations/"
            "eigen_rcr_dfl/"
            "outputs/"
            "mechanism/"
            "stepA_risk_validity"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def lambda_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))

    return str(value).replace(".", "p")


def prepare_output_dir(
    path: str | Path,
    overwrite: bool,
) -> Path:
    output_dir = Path(path)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"출력 폴더가 이미 존재합니다: "
                f"{output_dir}\n"
                f"--overwrite를 추가하세요."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return output_dir


def max_drawdown(
    returns: np.ndarray,
    return_type: str,
) -> float:
    returns = np.asarray(
        returns,
        dtype=np.float64,
    )

    if return_type == "simple":
        wealth = np.cumprod(
            1.0 + returns
        )

    else:
        wealth = np.exp(
            np.cumsum(returns)
        )

    wealth = np.concatenate(
        ([1.0], wealth)
    )

    peak = np.maximum.accumulate(
        wealth
    )

    drawdown = (
        wealth / peak
        - 1.0
    )

    return float(
        -drawdown.min()
    )


def future_fixed_weight_risk(
    weights: np.ndarray,
    returns: np.ndarray,
    horizon: int,
    periods_per_year: int,
    return_type: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    t 시점에서 결정된 weight를
    t ~ t+horizon-1 기간 동안 고정하여
    future realized risk를 측정한다.

    t일 return은 weight 결정 이후의
    첫 번째 out-of-sample return이다.
    """

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
        n_dates
        - horizon
        + 1
    )

    for t in range(n_valid):

        future_asset_returns = (
            returns[
                t:t + horizon
            ]
        )

        portfolio_returns = (
            future_asset_returns
            @ weights[t]
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
                portfolio_returns,
                return_type,
            )
        )

    return (
        future_variance,
        future_volatility,
        future_mdd,
    )


def quadratic_exposure(
    weights: np.ndarray,
    matrices: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "bi,bij,bj->b",
        weights,
        matrices,
        weights,
        optimize=True,
    )


def build_test_arrays(
    dataset: EigenRCRDataset,
    validation_end: str,
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
]:
    validation_end = pd.Timestamp(
        validation_end
    )

    all_dates = (
        dataset.target_dates
    )

    indices = np.flatnonzero(
        all_dates
        > validation_end
    )

    if len(indices) == 0:
        raise RuntimeError(
            "Test 기간 sample이 없습니다."
        )

    dates = pd.DatetimeIndex(
        all_dates[indices]
    ).normalize()

    residual_covariances = (
        dataset
        .residual_covariances[
            indices
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    target_positions = (
        dataset.target_positions[
            indices
        ]
    )

    returns = (
        dataset.returns[
            target_positions
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    return (
        dates,
        residual_covariances,
        returns,
    )


def build_eigenspace_matrices(
    residual_covariances: np.ndarray,
    ks: list[int],
) -> dict[
    tuple[str, int],
    np.ndarray,
]:
    """
    residual covariance를 eigen decomposition하여

    Top-K
    Bottom-K
    Full

    matrix를 생성한다.

    여기서는 preserve_residual_trace scaling을
    적용하지 않는다.

    Raw exposure:
        원래 eigenvalue magnitude 유지

    Normalized exposure:
        trace로 scale 제거
    """

    n_assets = (
        residual_covariances.shape[-1]
    )

    for k in ks:
        if not (
            1 <= k < n_assets
        ):
            raise ValueError(
                "Experiment A의 K는 "
                f"1~{n_assets - 1}이어야 합니다. "
                f"현재 K={k}"
            )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            residual_covariances
        )
    )

    eigenvalues = np.clip(
        eigenvalues,
        0.0,
        None,
    )

    matrices = {
        (
            "full",
            n_assets,
        ): residual_covariances
    }

    for k in ks:

        # --------------------------------------
        # Bottom-K
        # eigh는 eigenvalue 오름차순
        # --------------------------------------

        bottom_values = (
            eigenvalues[
                :,
                :k,
            ]
        )

        bottom_vectors = (
            eigenvectors[
                :,
                :,
                :k,
            ]
        )

        bottom = (
            bottom_vectors
            * bottom_values[
                :,
                None,
                :,
            ]
        ) @ np.swapaxes(
            bottom_vectors,
            -1,
            -2,
        )

        # --------------------------------------
        # Top-K
        # --------------------------------------

        top_values = (
            eigenvalues[
                :,
                -k:,
            ]
        )

        top_vectors = (
            eigenvectors[
                :,
                :,
                -k:,
            ]
        )

        top = (
            top_vectors
            * top_values[
                :,
                None,
                :,
            ]
        ) @ np.swapaxes(
            top_vectors,
            -1,
            -2,
        )

        top = 0.5 * (
            top
            + np.swapaxes(
                top,
                -1,
                -2,
            )
        )

        bottom = 0.5 * (
            bottom
            + np.swapaxes(
                bottom,
                -1,
                -2,
            )
        )

        matrices[
            (
                "top",
                k,
            )
        ] = top

        matrices[
            (
                "bottom",
                k,
            )
        ] = bottom

    return matrices


def load_dfl_weights(
    baseline_root: Path,
    risk_aversion: float,
    seed: int,
    tickers: list[str],
    reference_dates: pd.DatetimeIndex,
    reference_returns: np.ndarray,
) -> np.ndarray:

    path = (
        baseline_root
        / (
            "lambda_"
            + lambda_tag(
                risk_aversion
            )
        )
        / f"seed_{seed}"
        / "asset_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "DFL-MVO result not found: "
            f"{path}"
        )

    frame = pd.read_csv(
        path
    )

    required = {
        "date",
        "ticker",
        "predicted_weight",
        "true_return",
    }

    missing = (
        required.difference(
            frame.columns
        )
    )

    if missing:
        raise KeyError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    frame["date"] = (
        pd.to_datetime(
            frame["date"]
        )
        .dt
        .normalize()
    )

    frame["ticker"] = (
        frame["ticker"]
        .astype(str)
    )

    weights = (
        frame
        .pivot(
            index="date",
            columns="ticker",
            values="predicted_weight",
        )
        .reindex(
            index=reference_dates,
            columns=tickers,
        )
    )

    true_returns = (
        frame
        .pivot(
            index="date",
            columns="ticker",
            values="true_return",
        )
        .reindex(
            index=reference_dates,
            columns=tickers,
        )
    )

    if (
        weights
        .isna()
        .any()
        .any()
        or
        true_returns
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "DFL-MVO 결과를 dataset의 "
            "test date/ticker에 정렬할 수 없습니다: "
            f"lambda={risk_aversion:g}, "
            f"seed={seed}"
        )

    true_returns_np = (
        true_returns.to_numpy(
            dtype=np.float64
        )
    )

    if not np.allclose(
        true_returns_np,
        reference_returns,
        rtol=1e-7,
        atol=1e-10,
    ):
        max_abs = float(
            np.max(
                np.abs(
                    true_returns_np
                    - reference_returns
                )
            )
        )

        raise RuntimeError(
            "DFL-MVO true_return과 "
            "EigenRCRDataset return이 "
            "일치하지 않습니다: "
            f"lambda={risk_aversion:g}, "
            f"seed={seed}, "
            f"max_abs_diff={max_abs:.3e}"
        )

    weights_np = (
        weights.to_numpy(
            dtype=np.float64
        )
    )

    if not np.all(
        np.isfinite(
            weights_np
        )
    ):
        raise RuntimeError(
            "DFL-MVO weight에 "
            "non-finite 값이 있습니다: "
            f"lambda={risk_aversion:g}, "
            f"seed={seed}"
        )

    return weights_np


def exposure_arrays(
    weights: np.ndarray,
    matrices: np.ndarray,
    full_trace: np.ndarray,
    eps: float = 1e-15,
):
    """
    Raw exposure:
        w' A w

    Normalized exposure:
        (w' A w) / trace(Sigma_res)

    trace_share:
        trace(A) / trace(Sigma_res)

    모든 Top / Bottom / Full matrix가
    동일한 full residual trace를 분모로 사용한다.
    """

    raw = quadratic_exposure(
        weights,
        matrices,
    )

    matrix_trace = np.trace(
        matrices,
        axis1=1,
        axis2=2,
    )

    normalized = np.divide(
        raw,
        full_trace,
        out=np.full_like(
            raw,
            np.nan,
        ),
        where=(
            np.abs(
                full_trace
            )
            > eps
        ),
    )

    trace_share = np.divide(
        matrix_trace,
        full_trace,
        out=np.full_like(
            matrix_trace,
            np.nan,
        ),
        where=(
            np.abs(
                full_trace
            )
            > eps
        ),
    )

    return (
        raw,
        normalized,
        matrix_trace,
        trace_share,
    )

def build_daily_rows(
    *,
    weight_type: str,
    risk_aversion: float | None,
    seed: int | None,
    dates: pd.DatetimeIndex,
    weights: np.ndarray,
    returns: np.ndarray,
    matrix_bank: dict,
    horizon: int,
    periods_per_year: int,
    return_type: str,
) -> pd.DataFrame:

    (
        future_variance,
        future_volatility,
        future_mdd,
    ) = future_fixed_weight_risk(
        weights=weights,
        returns=returns,
        horizon=horizon,
        periods_per_year=(
            periods_per_year
        ),
        return_type=return_type,
    )

    n_assets = (
        weights.shape[1]
    )

    full_matrix = (
        matrix_bank[
            (
                "full",
                n_assets,
            )
        ]
    )

    full_trace = np.trace(
        full_matrix,
        axis1=1,
        axis2=2,
    )

    frames = []

    for (
        space,
        k,
    ), matrices in (
        matrix_bank.items()
    ):

        (
            raw,
            normalized,
            matrix_trace,
            trace_share,
        ) = exposure_arrays(
            weights=weights,
            matrices=matrices,
            full_trace=full_trace,
        )

        frames.append(
            pd.DataFrame(
                {
                    "weight_type":
                        weight_type,

                    "lambda":
                        risk_aversion,

                    "seed":
                        seed,

                    "date":
                        dates,

                    "space":
                        space,

                    "k":
                        k,

                    "raw_exposure":
                        raw,

                    "normalized_exposure":
                        normalized,

                    "matrix_trace":
                        matrix_trace,

                    "trace_share_of_full":
                        trace_share,

                    "future_variance":
                        future_variance,

                    "future_volatility":
                        future_volatility,

                    "future_mdd":
                        future_mdd,
                }
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def quartile_labels(
    values: pd.Series,
) -> pd.Series:
    """
    동일 exposure 값으로 qcut edge가
    중복되는 문제를 막기 위해
    rank(method='first')를 사용한다.
    """

    ranked = values.rank(
        method="first"
    )

    return pd.qcut(
        ranked,
        q=4,
        labels=[
            "Q1_Low",
            "Q2",
            "Q3",
            "Q4_High",
        ],
    )


def quartile_summary(
    daily: pd.DataFrame,
    exposure_column: str,
) -> pd.DataFrame:

    sample = (
        daily[
            [
                exposure_column,
                "future_variance",
                "future_volatility",
                "future_mdd",
            ]
        ]
        .dropna()
        .copy()
    )

    if len(sample) < 4:
        raise RuntimeError(
            "Quartile 분석 sample이 부족합니다."
        )

    sample["quartile"] = (
        quartile_labels(
            sample[
                exposure_column
            ]
        )
    )

    summary = (
        sample
        .groupby(
            "quartile",
            observed=True,
        )
        .agg(
            n=(
                exposure_column,
                "size",
            ),

            mean_exposure=(
                exposure_column,
                "mean",
            ),

            future_variance=(
                "future_variance",
                "mean",
            ),

            future_volatility=(
                "future_volatility",
                "mean",
            ),

            future_mdd=(
                "future_mdd",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.insert(
        0,
        "exposure_type",
        exposure_column,
    )

    return summary


def hac_test(
    daily: pd.DataFrame,
    exposure_column: str,
    future_column: str,
    hac_lags: int,
):
    work = (
        daily[
            [
                exposure_column,
                future_column,
            ]
        ]
        .dropna()
        .copy()
    )

    if len(work) < 8:
        raise RuntimeError(
            "HAC 분석 sample이 부족합니다."
        )

    threshold = float(
        work[
            exposure_column
        ].quantile(
            0.75
        )
    )

    work["high_q4"] = (
        work[
            exposure_column
        ]
        >= threshold
    ).astype(float)

    low = work.loc[
        work[
            "high_q4"
        ] == 0,
        future_column,
    ]

    high = work.loc[
        work[
            "high_q4"
        ] == 1,
        future_column,
    ]

    low_mean = float(
        low.mean()
    )

    high_mean = float(
        high.mean()
    )

    if abs(
        low_mean
    ) > 1e-15:

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

    # --------------------------------------
    # Q4 dummy HAC regression
    # --------------------------------------

    q4_model = sm.OLS(
        work[
            future_column
        ].to_numpy(
            dtype=np.float64
        ),

        sm.add_constant(
            work[
                "high_q4"
            ]
        ),

    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags":
                hac_lags
        },
    )

    # --------------------------------------
    # Continuous exposure:
    # exposure +1 standard deviation
    # --------------------------------------

    x = (
        work[
            exposure_column
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    x_std = float(
        x.std(
            ddof=1
        )
    )

    if x_std > 0:

        z = (
            x
            - x.mean()
        ) / x_std

        continuous_model = (
            sm.OLS(
                work[
                    future_column
                ].to_numpy(
                    dtype=np.float64
                ),
                sm.add_constant(
                    z
                ),
            )
            .fit(
                cov_type="HAC",
                cov_kwds={
                    "maxlags":
                        hac_lags
                },
            )
        )

        beta_1sd = float(
            continuous_model.params[1]
        )

        p_1sd = float(
            continuous_model.pvalues[1]
        )

    else:
        beta_1sd = np.nan
        p_1sd = np.nan

    return {
        "n":
            int(
                len(work)
            ),

        "q4_threshold":
            threshold,

        "q1_q3_mean":
            low_mean,

        "q4_mean":
            high_mean,

        "q4_minus_q1_q3":
            high_mean
            - low_mean,

        "increase_pct":
            increase_pct,

        "hac_beta_q4":
            float(
                q4_model.params[
                    "high_q4"
                ]
            ),

        "hac_se_q4":
            float(
                q4_model.bse[
                    "high_q4"
                ]
            ),

        "hac_t_q4":
            float(
                q4_model.tvalues[
                    "high_q4"
                ]
            ),

        "hac_p_q4":
            float(
                q4_model.pvalues[
                    "high_q4"
                ]
            ),

        "hac_beta_per_1sd_exposure":
            beta_1sd,

        "hac_p_per_1sd_exposure":
            p_1sd,
    }


def summarize_one_configuration(
    daily: pd.DataFrame,
    hac_lags: int,
):
    quartile_frames = []
    test_rows = []

    config_columns = [
        "weight_type",
        "lambda",
        "seed",
        "space",
        "k",
    ]

    config = {
        column:
            daily.iloc[0][
                column
            ]
        for column
        in config_columns
    }

    for exposure_column in [
        "raw_exposure",
        "normalized_exposure",
    ]:

        q = quartile_summary(
            daily=daily,
            exposure_column=(
                exposure_column
            ),
        )

        for column, value in (
            reversed(
                list(
                    config.items()
                )
            )
        ):
            q.insert(
                0,
                column,
                value,
            )

        quartile_frames.append(
            q
        )

        for future_column in [
            "future_variance",
            "future_volatility",
            "future_mdd",
        ]:

            result = hac_test(
                daily=daily,
                exposure_column=(
                    exposure_column
                ),
                future_column=(
                    future_column
                ),
                hac_lags=hac_lags,
            )

            test_rows.append(
                {
                    **config,

                    "exposure_type":
                        exposure_column,

                    "future_metric":
                        future_column,

                    **result,
                }
            )

    return (
        pd.concat(
            quartile_frames,
            ignore_index=True,
        ),
        pd.DataFrame(
            test_rows
        ),
    )


def aggregate_seed_tests(
    tests: pd.DataFrame,
) -> pd.DataFrame:

    dfl = (
        tests.loc[
            tests[
                "weight_type"
            ] == "dfl_mvo"
        ]
        .copy()
    )

    dfl[
        "significant_positive"
    ] = (
        (
            dfl[
                "hac_beta_q4"
            ] > 0
        )
        &
        (
            dfl[
                "hac_p_q4"
            ] < 0.05
        )
    ).astype(int)

    summary = (
        dfl
        .groupby(
            [
                "weight_type",
                "lambda",
                "space",
                "k",
                "exposure_type",
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
                        (
                            x > 0
                        ).sum()
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

    return summary


def print_primary_summary(
    ew_tests: pd.DataFrame,
    dfl_summary: pd.DataFrame,
):
    print()

    print(
        "=" * 110
    )

    print(
        "EQUAL-WEIGHT | "
        "Q4 vs Q1-Q3"
    )

    print(
        "=" * 110
    )

    display = ew_tests[
        [
            "space",
            "k",
            "exposure_type",
            "future_metric",
            "increase_pct",
            "hac_p_q4",
            "hac_beta_per_1sd_exposure",
            "hac_p_per_1sd_exposure",
        ]
    ]

    print(
        display
        .round(6)
        .to_string(
            index=False
        )
    )

    print()

    print(
        "=" * 110
    )

    print(
        "DFL-MVO | "
        "MULTI-SEED SUMMARY"
    )

    print(
        "=" * 110
    )

    display = dfl_summary[
        [
            "lambda",
            "space",
            "k",
            "exposure_type",
            "future_metric",
            "mean_increase_pct",
            "positive_direction_seeds",
            "significant_positive_seeds",
        ]
    ]

    print(
        display
        .round(6)
        .to_string(
            index=False
        )
    )


def main():
    args = parse_args()

    if args.horizon < 2:
        raise ValueError(
            "--horizon은 2 이상이어야 합니다."
        )

    if len(
        set(
            args.ks
        )
    ) != len(
        args.ks
    ):
        raise ValueError(
            "--ks에 중복 값이 있습니다."
        )

    hac_lags = (
        args.horizon - 1
        if args.hac_lags is None
        else args.hac_lags
    )

    if hac_lags < 0:
        raise ValueError(
            "--hac-lags는 0 이상이어야 합니다."
        )

    output_dir = (
        prepare_output_dir(
            args.output_dir,
            args.overwrite,
        )
    )

    # ---------------------------------------------------------
    # Dataset
    #
    # eigen_k 자체는 여기서 분석용으로 쓰지 않는다.
    # residual covariance를 얻기 위해 dataset을 생성한다.
    #
    # scale_mode="none"
    # ---------------------------------------------------------

    dataset = EigenRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type=args.return_type,
        covariance_jitter=(
            args.covariance_jitter
        ),

        market_mode=(
            args.market_mode
        ),
        market_price_csv=(
            args.market_price_csv
        ),
        market_column=(
            args.market_column
        ),

        risk_free_rate=(
            args.risk_free_rate
        ),

        fit_intercept=(
            not args.no_capm_intercept
        ),

        eigen_k=max(
            args.ks
        ),

        scale_mode="none",
    )

    (
        dates,
        residual_covariances,
        returns,
    ) = build_test_arrays(
        dataset=dataset,
        validation_end=(
            args.validation_end
        ),
    )

    matrix_bank = (
        build_eigenspace_matrices(
            residual_covariances=(
                residual_covariances
            ),
            ks=args.ks,
        )
    )

    tickers = [
        str(
            ticker
        )
        for ticker
        in dataset.tickers
    ]

    n_assets = (
        dataset.n_assets
    )

    print(
        "=" * 110
    )

    print(
        "EXPERIMENT A | "
        "TOP / BOTTOM / FULL "
        "RESIDUAL EIGENSPACE "
        "-> FUTURE RISK"
    )

    print(
        "=" * 110
    )

    print(
        f"Assets        : "
        f"{n_assets}"
    )

    print(
        f"Test dates    : "
        f"{len(dates)}"
    )

    print(
        f"Date range    : "
        f"{dates.min().date()} "
        f"~ "
        f"{dates.max().date()}"
    )

    print(
        f"K values      : "
        f"{args.ks}"
    )

    print(
        f"Lambdas       : "
        f"{args.lambdas}"
    )

    print(
        f"Seeds         : "
        f"{args.seeds}"
    )

    print(
        f"Horizon       : "
        f"{args.horizon}"
    )

    print(
        f"HAC lags      : "
        f"{hac_lags}"
    )

    print(
        "Exposure scale: "
        "raw residual covariance "
        "+ trace-normalized"
    )

    print(
        "=" * 110
    )

    daily_frames = []
    quartile_frames = []
    test_frames = []

    # =========================================================
    # A-1
    # Equal Weight
    # =========================================================

    ew_weights = np.full(
        (
            len(dates),
            n_assets,
        ),
        1.0 / n_assets,
        dtype=np.float64,
    )

    ew_daily_all = (
        build_daily_rows(
            weight_type=(
                "equal_weight"
            ),

            risk_aversion=None,
            seed=None,

            dates=dates,
            weights=ew_weights,
            returns=returns,

            matrix_bank=(
                matrix_bank
            ),

            horizon=(
                args.horizon
            ),

            periods_per_year=(
                args.periods_per_year
            ),

            return_type=(
                args.return_type
            ),
        )
    )

    daily_frames.append(
        ew_daily_all
    )

    for (
        _,
        _,
        space,
        k,
    ), group in (
        ew_daily_all
        .groupby(
            [
                "weight_type",
                "lambda",
                "space",
                "k",
            ],
            dropna=False,
            sort=False,
        )
    ):

        q, t = (
            summarize_one_configuration(
                daily=group,
                hac_lags=hac_lags,
            )
        )

        quartile_frames.append(
            q
        )

        test_frames.append(
            t
        )

    print(
        "A-1 Equal Weight: DONE"
    )

    # =========================================================
    # A-2
    # Standardized DFL-MVO weights
    # =========================================================

    baseline_root = Path(
        args.baseline_root
    )

    for risk_aversion in (
        args.lambdas
    ):

        for seed in (
            args.seeds
        ):

            weights = (
                load_dfl_weights(
                    baseline_root=(
                        baseline_root
                    ),

                    risk_aversion=(
                        risk_aversion
                    ),

                    seed=seed,

                    tickers=tickers,

                    reference_dates=(
                        dates
                    ),

                    reference_returns=(
                        returns
                    ),
                )
            )

            daily = (
                build_daily_rows(
                    weight_type=(
                        "dfl_mvo"
                    ),

                    risk_aversion=(
                        risk_aversion
                    ),

                    seed=seed,

                    dates=dates,
                    weights=weights,
                    returns=returns,

                    matrix_bank=(
                        matrix_bank
                    ),

                    horizon=(
                        args.horizon
                    ),

                    periods_per_year=(
                        args.periods_per_year
                    ),

                    return_type=(
                        args.return_type
                    ),
                )
            )

            daily_frames.append(
                daily
            )

            for (
                _,
                _,
                _,
                space,
                k,
            ), group in (
                daily.groupby(
                    [
                        "weight_type",
                        "lambda",
                        "seed",
                        "space",
                        "k",
                    ],
                    dropna=False,
                    sort=False,
                )
            ):

                q, t = (
                    summarize_one_configuration(
                        daily=group,
                        hac_lags=(
                            hac_lags
                        ),
                    )
                )

                quartile_frames.append(
                    q
                )

                test_frames.append(
                    t
                )

            print(
                "A-2 DFL-MVO | "
                f"lambda="
                f"{risk_aversion:g} "
                f"| seed="
                f"{seed} "
                "DONE"
            )

    # =========================================================
    # Save
    # =========================================================

    daily_all = pd.concat(
        daily_frames,
        ignore_index=True,
    )

    quartiles = pd.concat(
        quartile_frames,
        ignore_index=True,
    )

    tests = pd.concat(
        test_frames,
        ignore_index=True,
    )

    ew_tests = (
        tests.loc[
            tests[
                "weight_type"
            ]
            == "equal_weight"
        ]
        .copy()
    )

    dfl_summary = (
        aggregate_seed_tests(
            tests
        )
    )

    daily_path = (
        output_dir
        / "eigen_exposure_future_risk_daily.csv"
    )

    quartile_path = (
        output_dir
        / "eigen_exposure_quartiles.csv"
    )

    test_path = (
        output_dir
        / "eigen_exposure_hac_tests.csv"
    )

    ew_path = (
        output_dir
        / "equal_weight_summary.csv"
    )

    dfl_path = (
        output_dir
        / "dfl_multiseed_summary.csv"
    )

    daily_all.to_csv(
        daily_path,
        index=False,
    )

    quartiles.to_csv(
        quartile_path,
        index=False,
    )

    tests.to_csv(
        test_path,
        index=False,
    )

    ew_tests.to_csv(
        ew_path,
        index=False,
    )

    dfl_summary.to_csv(
        dfl_path,
        index=False,
    )

    print_primary_summary(
        ew_tests=ew_tests,
        dfl_summary=dfl_summary,
    )

    print()

    print(
        "Saved:"
    )

    print(
        daily_path
    )

    print(
        quartile_path
    )

    print(
        test_path
    )

    print(
        ew_path
    )

    print(
        dfl_path
    )


if __name__ == "__main__":
    main()