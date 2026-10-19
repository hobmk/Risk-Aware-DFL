from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

from implementations.eigen_rcr_dfl.src.capm import fit_capm
from implementations.eigen_rcr_dfl.src.dataset import EigenRCRDataset
from implementations.eigen_rcr_dfl.src.eigen_risk import covariance_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Downside vs Upside Rank-Tail placebo validation"
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
        default=[42, 43, 44],
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


def ordinal_rank_percentiles(
    values: torch.Tensor,
) -> torch.Tensor:
    """
    values: [B, T, N]

    asset별 rolling window 내부 ordinal rank:
        1, ..., T
    를 (T + 1)로 나눔.
    """

    B, T, N = values.shape

    order = torch.argsort(
        values,
        dim=1,
    )

    ranks = torch.empty_like(
        order,
    )

    rank_values = (
        torch.arange(
            1,
            T + 1,
            dtype=order.dtype,
            device=order.device,
        )
        .view(1, T, 1)
        .expand(B, T, N)
    )

    ranks.scatter_(
        1,
        order,
        rank_values,
    )

    return ranks.to(
        dtype=values.dtype
    ) / float(T + 1)


def build_tail_matrix(
    residuals: torch.Tensor,
    tail_q: float,
    side: str,
    eps: float = 1e-12,
):
    """
    residuals: [B, T, N]

    Down:
        z = max(0, (q-u)/q)

    Up:
        z = max(0, (u-(1-q))/q)

    centered score covariance
    -> score correlation
    -> residual std scaling.
    """

    if side not in {
        "down",
        "up",
    }:
        raise ValueError(
            "side must be down or up"
        )

    u = ordinal_rank_percentiles(
        residuals
    )

    if side == "down":
        scores = torch.clamp(
            (tail_q - u) / tail_q,
            min=0.0,
        )
    else:
        scores = torch.clamp(
            (
                u
                - (1.0 - tail_q)
            )
            / tail_q,
            min=0.0,
        )

    centered = (
        scores
        - scores.mean(
            dim=1,
            keepdim=True,
        )
    )

    T = residuals.shape[1]

    score_covariance = (
        centered.transpose(
            1,
            2,
        )
        @ centered
    ) / float(T - 1)

    score_variance = torch.diagonal(
        score_covariance,
        dim1=-2,
        dim2=-1,
    )

    inv_std = torch.rsqrt(
        torch.clamp(
            score_variance,
            min=eps,
        )
    )

    score_correlation = (
        score_covariance
        * inv_std.unsqueeze(-1)
        * inv_std.unsqueeze(-2)
    )

    # numerical symmetry
    score_correlation = 0.5 * (
        score_correlation
        + score_correlation.transpose(
            -1,
            -2,
        )
    )

    n_assets = residuals.shape[-1]

    idx = torch.arange(
        n_assets,
        device=residuals.device,
    )

    score_correlation[
        :,
        idx,
        idx,
    ] = 1.0

    residual_covariance = (
        covariance_matrix(
            residuals
        )
    )

    residual_variance = torch.diagonal(
        residual_covariance,
        dim1=-2,
        dim2=-1,
    )

    residual_std = torch.sqrt(
        torch.clamp(
            residual_variance,
            min=0.0,
        )
    )

    risk_matrix = (
        score_correlation
        * residual_std.unsqueeze(-1)
        * residual_std.unsqueeze(-2)
    )

    return risk_matrix


def build_tail_matrix_maps(
    dataset: EigenRCRDataset,
    tail_q: float,
):
    """
    EigenRCRDataset과 같은 rolling CAPM residual을 다시 계산해서
    downside / upside matrix 생성.
    """

    asset_windows = np.lib.stride_tricks.sliding_window_view(
        dataset.returns,
        window_shape=dataset.lookback,
        axis=0,
    ).transpose(
        0,
        2,
        1,
    )[:-1]

    market_windows = np.lib.stride_tricks.sliding_window_view(
        dataset.market_returns,
        window_shape=dataset.lookback,
    )[:-1]

    asset_windows = torch.from_numpy(
        np.ascontiguousarray(
            asset_windows
        )
    ).to(torch.float64)

    market_windows = torch.from_numpy(
        np.ascontiguousarray(
            market_windows
        )
    ).to(torch.float64)

    capm = fit_capm(
        asset_returns=asset_windows,
        market_returns=market_windows,
        risk_free_rate=dataset.risk_free_rate,
        fit_intercept=dataset.fit_intercept,
    )

    downside = build_tail_matrix(
        residuals=capm.residuals,
        tail_q=tail_q,
        side="down",
    )

    upside = build_tail_matrix(
        residuals=capm.residuals,
        tail_q=tail_q,
        side="up",
    )

    down_map = {}
    up_map = {}

    for i, date in enumerate(
        dataset.target_dates
    ):
        date = pd.Timestamp(
            date
        ).normalize()

        down_map[date] = (
            downside[i]
            .detach()
            .cpu()
            .numpy()
        )

        up_map[date] = (
            upside[i]
            .detach()
            .cpu()
            .numpy()
        )

    return (
        down_map,
        up_map,
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
            f"Baseline not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "date",
        "ticker",
        "predicted_weight",
        "true_return",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise KeyError(
            f"Missing columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    df["ticker"] = (
        df["ticker"].astype(str)
    )

    weights = (
        df.pivot(
            index="date",
            columns="ticker",
            values="predicted_weight",
        )
        .sort_index()
        .reindex(columns=tickers)
    )

    returns = (
        df.pivot(
            index="date",
            columns="ticker",
            values="true_return",
        )
        .sort_index()
        .reindex(columns=tickers)
    )

    if (
        weights.isna().any().any()
        or returns.isna().any().any()
    ):
        raise RuntimeError(
            f"NaN after pivot: seed={seed}"
        )

    return (
        weights.index,
        weights.to_numpy(
            dtype=np.float64
        ),
        returns.to_numpy(
            dtype=np.float64
        ),
    )


def split_matrix(
    matrices: np.ndarray,
):
    n_assets = matrices.shape[-1]

    diagonal = np.zeros_like(
        matrices
    )

    diag_values = np.diagonal(
        matrices,
        axis1=1,
        axis2=2,
    )

    idx = np.arange(
        n_assets
    )

    diagonal[
        :,
        idx,
        idx,
    ] = diag_values

    offdiag = (
        matrices
        - diagonal
    )

    positive_offdiag = np.maximum(
        offdiag,
        0.0,
    )

    return (
        offdiag,
        positive_offdiag,
    )


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


def max_drawdown(
    returns: np.ndarray,
):
    wealth = np.cumprod(
        1.0 + returns
    )

    wealth = np.concatenate(
        [
            np.array([1.0]),
            wealth,
        ]
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
):
    n_dates = len(
        weights
    )

    variance = np.full(
        n_dates,
        np.nan,
    )

    volatility = np.full(
        n_dates,
        np.nan,
    )

    mdd = np.full(
        n_dates,
        np.nan,
    )

    n_valid = (
        n_dates
        - horizon
        + 1
    )

    for t in range(
        n_valid
    ):
        future_returns = (
            returns[
                t:t + horizon
            ]
        )

        portfolio_returns = (
            future_returns
            @ weights[t]
        )

        var = (
            np.var(
                portfolio_returns,
                ddof=1,
            )
            * periods_per_year
        )

        variance[t] = var

        volatility[t] = np.sqrt(
            max(
                var,
                0.0,
            )
        )

        mdd[t] = max_drawdown(
            portfolio_returns
        )

    return {
        "future_variance":
            variance,

        "future_volatility":
            volatility,

        "future_mdd":
            mdd,
    }


def hac_q4_test(
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

    increase_pct = (
        (
            high_mean
            / low_mean
            - 1.0
        )
        * 100.0
        if abs(low_mean) > 1e-15
        else np.nan
    )

    X = sm.add_constant(
        work["high_q4"]
    )

    model = sm.OLS(
        work[
            future_column
        ],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": hac_lags,
        },
    )

    return {
        "n":
            len(work),

        "q4_threshold":
            threshold,

        "q1_q3_mean":
            low_mean,

        "q4_mean":
            high_mean,

        "increase_pct":
            increase_pct,

        "hac_beta_q4":
            float(
                model.params[
                    "high_q4"
                ]
            ),

        "hac_t_q4":
            float(
                model.tvalues[
                    "high_q4"
                ]
            ),

        "hac_p_q4":
            float(
                model.pvalues[
                    "high_q4"
                ]
            ),
    }


def joint_hac_test(
    dataframe: pd.DataFrame,
    down_column: str,
    up_column: str,
    future_column: str,
    hac_lags: int,
):
    """
    Future risk ~ standardized Down exposure + standardized Up exposure.

    둘을 동시에 넣었을 때 어느 방향이 incremental information을
    가지는지 확인.
    """

    work = dataframe[
        [
            down_column,
            up_column,
            future_column,
        ]
    ].dropna().copy()

    for column in [
        down_column,
        up_column,
    ]:
        mean = work[column].mean()
        std = work[column].std(
            ddof=1
        )

        work[
            f"z_{column}"
        ] = (
            work[column]
            - mean
        ) / std

    X = sm.add_constant(
        work[
            [
                f"z_{down_column}",
                f"z_{up_column}",
            ]
        ]
    )

    model = sm.OLS(
        work[
            future_column
        ],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": hac_lags,
        },
    )

    return {
        "down_beta":
            float(
                model.params[
                    f"z_{down_column}"
                ]
            ),

        "down_p":
            float(
                model.pvalues[
                    f"z_{down_column}"
                ]
            ),

        "up_beta":
            float(
                model.params[
                    f"z_{up_column}"
                ]
            ),

        "up_p":
            float(
                model.pvalues[
                    f"z_{up_column}"
                ]
            ),
    }


def main():
    args = parse_args()

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
            / "step3_downside_upside_placebo"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 110
    )
    print(
        "RANK-TAIL STEP 3A: "
        "DOWNSIDE vs UPSIDE PLACEBO"
    )
    print(
        "=" * 110
    )

    print(
        f"Tail q   : {args.tail_q:.2f}"
    )
    print(
        f"Horizon  : {args.horizon}"
    )
    print(
        f"HAC lags : {args.hac_lags}"
    )
    print(
        f"Seeds    : {args.seeds}"
    )

    dataset = EigenRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        eigen_k=1,
        scale_mode="none",
        dtype=torch.float64,
    )

    tickers = [
        str(ticker)
        for ticker in dataset.tickers
    ]

    (
        down_map,
        up_map,
    ) = build_tail_matrix_maps(
        dataset=dataset,
        tail_q=args.tail_q,
    )

    baseline_root = Path(
        args.baseline_root
    )

    daily_frames = []
    q4_rows = []
    joint_rows = []

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

        dates = pd.DatetimeIndex(
            dates
        ).normalize()

        down_matrices = np.stack(
            [
                down_map[date]
                for date in dates
            ]
        )

        up_matrices = np.stack(
            [
                up_map[date]
                for date in dates
            ]
        )

        (
            down_offdiag,
            down_positive,
        ) = split_matrix(
            down_matrices
        )

        (
            up_offdiag,
            up_positive,
        ) = split_matrix(
            up_matrices
        )

        daily = pd.DataFrame(
            {
                "seed":
                    seed,

                "date":
                    dates,

                "down_offdiag_exposure":
                    quadratic_exposure(
                        weights,
                        down_offdiag,
                    ),

                "down_positive_offdiag_exposure":
                    quadratic_exposure(
                        weights,
                        down_positive,
                    ),

                "up_offdiag_exposure":
                    quadratic_exposure(
                        weights,
                        up_offdiag,
                    ),

                "up_positive_offdiag_exposure":
                    quadratic_exposure(
                        weights,
                        up_positive,
                    ),
            }
        )

        future = future_fixed_weight_risk(
            weights=weights,
            returns=returns,
            horizon=args.horizon,
            periods_per_year=(
                args.periods_per_year
            ),
        )

        for key, value in (
            future.items()
        ):
            daily[key] = value

        daily_frames.append(
            daily
        )

        exposure_pairs = [
            (
                "down",
                "offdiag",
                "down_offdiag_exposure",
            ),
            (
                "up",
                "offdiag",
                "up_offdiag_exposure",
            ),
            (
                "down",
                "positive_offdiag",
                "down_positive_offdiag_exposure",
            ),
            (
                "up",
                "positive_offdiag",
                "up_positive_offdiag_exposure",
            ),
        ]

        future_metrics = [
            "future_variance",
            "future_volatility",
            "future_mdd",
        ]

        for (
            side,
            exposure_type,
            exposure_column,
        ) in exposure_pairs:

            for future_metric in (
                future_metrics
            ):

                result = hac_q4_test(
                    dataframe=daily,
                    exposure_column=(
                        exposure_column
                    ),
                    future_column=(
                        future_metric
                    ),
                    hac_lags=(
                        args.hac_lags
                    ),
                )

                q4_rows.append(
                    {
                        "seed":
                            seed,

                        "tail_q":
                            args.tail_q,

                        "side":
                            side,

                        "exposure_type":
                            exposure_type,

                        "future_metric":
                            future_metric,

                        **result,
                    }
                )

        # Main joint comparison:
        # positive downside vs positive upside
        for future_metric in (
            future_metrics
        ):
            joint = joint_hac_test(
                dataframe=daily,
                down_column=(
                    "down_positive_offdiag_exposure"
                ),
                up_column=(
                    "up_positive_offdiag_exposure"
                ),
                future_column=(
                    future_metric
                ),
                hac_lags=(
                    args.hac_lags
                ),
            )

            joint_rows.append(
                {
                    "seed":
                        seed,

                    "tail_q":
                        args.tail_q,

                    "future_metric":
                        future_metric,

                    **joint,
                }
            )

        print(
            f"seed {seed} | "
            f"dates={len(dates)} | "
            f"future windows="
            f"{len(dates)-args.horizon+1}"
        )

    daily_all = pd.concat(
        daily_frames,
        ignore_index=True,
    )

    q4_by_seed = pd.DataFrame(
        q4_rows
    )

    joint_by_seed = pd.DataFrame(
        joint_rows
    )

    q4_by_seed[
        "significant_positive"
    ] = (
        (
            q4_by_seed[
                "hac_beta_q4"
            ] > 0
        )
        &
        (
            q4_by_seed[
                "hac_p_q4"
            ] < 0.05
        )
    ).astype(int)

    q4_summary = (
        q4_by_seed
        .groupby(
            [
                "tail_q",
                "side",
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
        )
    )

    joint_summary = (
        joint_by_seed
        .groupby(
            [
                "tail_q",
                "future_metric",
            ],
            as_index=False,
        )
        .agg(
            mean_down_beta=(
                "down_beta",
                "mean",
            ),

            down_positive_beta_seeds=(
                "down_beta",
                lambda x:
                    int(
                        (x > 0).sum()
                    ),
            ),

            down_significant_seeds=(
                "down_p",
                lambda x:
                    int(
                        (
                            (x < 0.05)
                        ).sum()
                    ),
            ),

            mean_up_beta=(
                "up_beta",
                "mean",
            ),

            up_positive_beta_seeds=(
                "up_beta",
                lambda x:
                    int(
                        (x > 0).sum()
                    ),
            ),

            up_significant_seeds=(
                "up_p",
                lambda x:
                    int(
                        (
                            (x < 0.05)
                        ).sum()
                    ),
            ),
        )
    )

    daily_path = (
        output_dir
        / "downside_upside_daily.csv"
    )

    q4_seed_path = (
        output_dir
        / "downside_upside_q4_by_seed.csv"
    )

    q4_summary_path = (
        output_dir
        / "downside_upside_q4_summary.csv"
    )

    joint_seed_path = (
        output_dir
        / "downside_upside_joint_hac_by_seed.csv"
    )

    joint_summary_path = (
        output_dir
        / "downside_upside_joint_hac_summary.csv"
    )

    daily_all.to_csv(
        daily_path,
        index=False,
    )

    q4_by_seed.to_csv(
        q4_seed_path,
        index=False,
    )

    q4_summary.to_csv(
        q4_summary_path,
        index=False,
    )

    joint_by_seed.to_csv(
        joint_seed_path,
        index=False,
    )

    joint_summary.to_csv(
        joint_summary_path,
        index=False,
    )

    primary = q4_summary[
        q4_summary[
            "exposure_type"
        ]
        == "positive_offdiag"
    ]

    print()
    print(
        "=" * 110
    )
    print(
        "PRIMARY DOWNSIDE vs UPSIDE "
        "POSITIVE-OFFDIAGONAL RESULTS"
    )
    print(
        "=" * 110
    )

    print(
        primary[
            [
                "side",
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
    print(
        "=" * 110
    )
    print(
        "JOINT HAC: FUTURE RISK ~ "
        "DOWNSIDE EXPOSURE + UPSIDE EXPOSURE"
    )
    print(
        "=" * 110
    )

    print(
        joint_summary.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(daily_path)
    print(q4_seed_path)
    print(q4_summary_path)
    print(joint_seed_path)
    print(joint_summary_path)


if __name__ == "__main__":
    main()