from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon

from implementations.eigen_rcr_dfl.src.dataset import EigenRCRDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment B: Top-K RCR-DFL exposure reduction "
            "vs DFL-MVO"
        )
    )

    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument(
        "--validation-end",
        default="2022-12-31",
    )

    parser.add_argument("--eigen-k", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.75)

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
        "--hac-lags",
        type=int,
        default=19,
    )

    parser.add_argument(
        "--dfl-root",
        default=(
            "implementations/"
            "dfl_mvo_lee2025/"
            "outputs/"
            "lambda_multiseed_standardized"
        ),
    )

    parser.add_argument(
        "--topk-root",
        default=(
            "implementations/"
            "eigen_rcr_dfl/"
            "outputs/"
            "topk_high_lambda"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "implementations/"
            "eigen_rcr_dfl/"
            "outputs/"
            "mechanism/"
            "experimentB_exposure_reduction"
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
                f"출력 폴더가 이미 존재합니다: {output_dir}\n"
                "--overwrite를 추가하세요."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return output_dir


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


def build_raw_topk(
    residual_covariance: np.ndarray,
    k: int,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        residual_covariance
    )

    eigenvalues = np.clip(
        eigenvalues,
        0.0,
        None,
    )

    top_values = eigenvalues[:, -k:]
    top_vectors = eigenvectors[:, :, -k:]

    top = (
        top_vectors
        * top_values[:, None, :]
    ) @ np.swapaxes(
        top_vectors,
        -1,
        -2,
    )

    return 0.5 * (
        top
        + np.swapaxes(
            top,
            -1,
            -2,
        )
    )


def build_test_matrices(
    dataset: EigenRCRDataset,
    validation_end: str,
    eta: float,
):
    validation_end = pd.Timestamp(
        validation_end
    )

    dates = dataset.target_dates

    indices = np.flatnonzero(
        dates > validation_end
    )

    if len(indices) == 0:
        raise RuntimeError(
            "Test sample이 없습니다."
        )

    test_dates = pd.DatetimeIndex(
        dates[indices]
    ).normalize()

    residual_covariance = (
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

    raw_topk = build_raw_topk(
        residual_covariance,
        dataset.eigen_k,
    )

    preserve_topk = (
        dataset
        .eigen_risk[
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

    model_penalty = (
        eta * preserve_topk
    )

    covariance = (
        dataset
        .covariances[
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

    return (
        test_dates,
        raw_topk,
        model_penalty,
        covariance,
    )


def load_dfl_weights(
    root: Path,
    risk_aversion: float,
    seed: int,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> np.ndarray:
    path = (
        root
        / f"lambda_{lambda_tag(risk_aversion)}"
        / f"seed_{seed}"
        / "asset_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"DFL-MVO result not found: {path}"
        )

    frame = pd.read_csv(path)

    required = {
        "date",
        "ticker",
        "predicted_weight",
    }

    missing = required.difference(
        frame.columns
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
        frame["ticker"].astype(str)
    )

    weights = (
        frame
        .pivot(
            index="date",
            columns="ticker",
            values="predicted_weight",
        )
        .reindex(
            index=dates,
            columns=tickers,
        )
    )

    if weights.isna().any().any():
        raise RuntimeError(
            "DFL-MVO weight 정렬 실패: "
            f"lambda={risk_aversion:g}, "
            f"seed={seed}"
        )

    return weights.to_numpy(
        dtype=np.float64
    )


def load_topk_weights(
    root: Path,
    risk_aversion: float,
    seed: int,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> np.ndarray:
    path = (
        root
        / f"lambda_{lambda_tag(risk_aversion)}"
        / f"seed_{seed}"
        / "portfolio_weights.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Top-K result not found: {path}"
        )

    frame = pd.read_csv(path)

    if "date" not in frame.columns:
        raise KeyError(
            f"{path}: date column이 없습니다."
        )

    frame["date"] = (
        pd.to_datetime(
            frame["date"]
        )
        .dt
        .normalize()
    )

    missing = set(tickers).difference(
        frame.columns
    )

    if missing:
        raise KeyError(
            f"{path} missing tickers: "
            f"{sorted(missing)}"
        )

    weights = (
        frame
        .set_index("date")
        .reindex(
            index=dates,
            columns=tickers,
        )
    )

    if weights.isna().any().any():
        raise RuntimeError(
            "Top-K weight 정렬 실패: "
            f"lambda={risk_aversion:g}, "
            f"seed={seed}"
        )

    return weights.to_numpy(
        dtype=np.float64
    )


def check_weights(
    weights: np.ndarray,
    label: str,
):
    if not np.isfinite(
        weights
    ).all():
        raise RuntimeError(
            f"{label}: non-finite weight"
        )

    sums = weights.sum(
        axis=1
    )

    max_error = float(
        np.max(
            np.abs(
                sums - 1.0
            )
        )
    )

    if max_error > 1e-4:
        raise RuntimeError(
            f"{label}: weight sum error "
            f"{max_error:.3e}"
        )


def assign_dfl_quartiles(
    exposure: np.ndarray,
) -> pd.Categorical:
    """
    Quartile은 반드시 DFL-MVO baseline
    raw Top-K exposure를 기준으로 정한다.
    """

    ranked = pd.Series(
        exposure
    ).rank(
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


def reduction_pct(
    dfl_value: float,
    topk_value: float,
) -> float:
    if abs(dfl_value) <= 1e-15:
        return np.nan

    return (
        (
            dfl_value
            - topk_value
        )
        / dfl_value
        * 100.0
    )


def hac_mean_difference(
    difference: np.ndarray,
    hac_lags: int,
):
    difference = np.asarray(
        difference,
        dtype=np.float64,
    )

    difference = difference[
        np.isfinite(
            difference
        )
    ]

    if len(difference) < 10:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    x = np.ones(
        (
            len(difference),
            1,
        ),
        dtype=np.float64,
    )

    model = sm.OLS(
        difference,
        x,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": hac_lags
        },
    )

    return (
        float(model.params[0]),
        float(model.bse[0]),
        float(model.tvalues[0]),
        float(model.pvalues[0]),
    )


def build_daily_frame(
    *,
    dates: pd.DatetimeIndex,
    risk_aversion: float,
    seed: int,
    dfl_weights: np.ndarray,
    topk_weights: np.ndarray,
    raw_topk: np.ndarray,
    model_penalty: np.ndarray,
    covariance: np.ndarray,
) -> pd.DataFrame:

    dfl_raw = quadratic_exposure(
        dfl_weights,
        raw_topk,
    )

    topk_raw = quadratic_exposure(
        topk_weights,
        raw_topk,
    )

    dfl_penalty = quadratic_exposure(
        dfl_weights,
        model_penalty,
    )

    topk_penalty = quadratic_exposure(
        topk_weights,
        model_penalty,
    )

    dfl_base = quadratic_exposure(
        dfl_weights,
        covariance,
    )

    topk_base = quadratic_exposure(
        topk_weights,
        covariance,
    )

    quartile = assign_dfl_quartiles(
        dfl_raw
    )

    frame = pd.DataFrame(
        {
            "lambda":
                risk_aversion,

            "seed":
                seed,

            "date":
                dates,

            "dfl_quartile":
                quartile,

            "dfl_raw_top5_exposure":
                dfl_raw,

            "topk_raw_top5_exposure":
                topk_raw,

            "raw_delta_topk_minus_dfl":
                topk_raw - dfl_raw,

            "raw_reduction_pct":
                np.divide(
                    dfl_raw - topk_raw,
                    dfl_raw,
                    out=np.full_like(
                        dfl_raw,
                        np.nan,
                    ),
                    where=(
                        np.abs(
                            dfl_raw
                        )
                        > 1e-15
                    ),
                )
                * 100.0,

            "dfl_model_penalty_exposure":
                dfl_penalty,

            "topk_model_penalty_exposure":
                topk_penalty,

            "penalty_delta_topk_minus_dfl":
                topk_penalty
                - dfl_penalty,

            "penalty_reduction_pct":
                np.divide(
                    dfl_penalty
                    - topk_penalty,
                    dfl_penalty,
                    out=np.full_like(
                        dfl_penalty,
                        np.nan,
                    ),
                    where=(
                        np.abs(
                            dfl_penalty
                        )
                        > 1e-15
                    ),
                )
                * 100.0,

            "dfl_base_covariance_risk":
                dfl_base,

            "topk_base_covariance_risk":
                topk_base,
        }
    )

    return frame


def summarize_seed(
    frame: pd.DataFrame,
    hac_lags: int,
) -> pd.DataFrame:
    rows = []

    segments = {
        "Overall": frame,
        "Q1_Low": frame.loc[
            frame["dfl_quartile"]
            == "Q1_Low"
        ],
        "Q2": frame.loc[
            frame["dfl_quartile"]
            == "Q2"
        ],
        "Q3": frame.loc[
            frame["dfl_quartile"]
            == "Q3"
        ],
        "Q4_High": frame.loc[
            frame["dfl_quartile"]
            == "Q4_High"
        ],
    }

    metrics = {
        "raw_top5": (
            "dfl_raw_top5_exposure",
            "topk_raw_top5_exposure",
        ),
        "model_penalty": (
            "dfl_model_penalty_exposure",
            "topk_model_penalty_exposure",
        ),
    }

    for segment_name, segment in (
        segments.items()
    ):
        for metric, (
            dfl_column,
            topk_column,
        ) in metrics.items():

            dfl_mean = float(
                segment[
                    dfl_column
                ].mean()
            )

            topk_mean = float(
                segment[
                    topk_column
                ].mean()
            )

            delta = (
                segment[
                    topk_column
                ].to_numpy()
                - segment[
                    dfl_column
                ].to_numpy()
            )

            (
                hac_beta,
                hac_se,
                hac_t,
                hac_p,
            ) = hac_mean_difference(
                delta,
                hac_lags,
            )

            rows.append(
                {
                    "lambda":
                        float(
                            frame[
                                "lambda"
                            ].iloc[0]
                        ),

                    "seed":
                        int(
                            frame[
                                "seed"
                            ].iloc[0]
                        ),

                    "segment":
                        segment_name,

                    "metric":
                        metric,

                    "n_dates":
                        len(segment),

                    "dfl_mean_exposure":
                        dfl_mean,

                    "topk_mean_exposure":
                        topk_mean,

                    "delta_topk_minus_dfl":
                        topk_mean
                        - dfl_mean,

                    "reduction_pct":
                        reduction_pct(
                            dfl_mean,
                            topk_mean,
                        ),

                    "hac_beta":
                        hac_beta,

                    "hac_se":
                        hac_se,

                    "hac_t":
                        hac_t,

                    "hac_p":
                        hac_p,

                    "hac_significant_reduction":
                        int(
                            np.isfinite(
                                hac_p
                            )
                            and hac_beta < 0
                            and hac_p < 0.05
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def safe_wilcoxon(
    differences: np.ndarray,
):
    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    differences = differences[
        np.isfinite(
            differences
        )
    ]

    if len(differences) == 0:
        return np.nan

    if np.allclose(
        differences,
        0.0,
    ):
        return 1.0

    result = wilcoxon(
        differences,
        alternative="two-sided",
        zero_method="wilcox",
    )

    return float(
        result.pvalue
    )


def aggregate_summary(
    by_seed: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "lambda",
        "segment",
        "metric",
    ]

    for keys, group in (
        by_seed.groupby(
            group_columns,
            sort=True,
        )
    ):
        risk_aversion, segment, metric = (
            keys
        )

        differences = (
            group[
                "delta_topk_minus_dfl"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        reductions = (
            group[
                "reduction_pct"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        rows.append(
            {
                "lambda":
                    risk_aversion,

                "segment":
                    segment,

                "metric":
                    metric,

                "seeds":
                    int(
                        group[
                            "seed"
                        ].nunique()
                    ),

                "dfl_mean_exposure":
                    float(
                        group[
                            "dfl_mean_exposure"
                        ].mean()
                    ),

                "topk_mean_exposure":
                    float(
                        group[
                            "topk_mean_exposure"
                        ].mean()
                    ),

                "mean_reduction_pct":
                    float(
                        np.nanmean(
                            reductions
                        )
                    ),

                "std_reduction_pct":
                    float(
                        np.nanstd(
                            reductions,
                            ddof=1,
                        )
                    ),

                "reduced_seeds":
                    int(
                        (
                            differences < 0
                        ).sum()
                    ),

                "hac_significant_reduction_seeds":
                    int(
                        group[
                            "hac_significant_reduction"
                        ].sum()
                    ),

                "wilcoxon_p_two_sided":
                    safe_wilcoxon(
                        differences
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def print_primary(
    summary: pd.DataFrame,
):
    primary = summary.loc[
        summary[
            "segment"
        ].isin(
            [
                "Overall",
                "Q4_High",
            ]
        )
    ].copy()

    primary = primary[
        [
            "lambda",
            "segment",
            "metric",
            "dfl_mean_exposure",
            "topk_mean_exposure",
            "mean_reduction_pct",
            "reduced_seeds",
            "hac_significant_reduction_seeds",
            "wilcoxon_p_two_sided",
        ]
    ]

    print()
    print(
        "=" * 125
    )
    print(
        "EXPERIMENT B | "
        "TOP-K RCR-DFL EXPOSURE REDUCTION"
    )
    print(
        "=" * 125
    )

    print(
        primary
        .round(8)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "reduction_pct > 0 : "
        "Top-K RCR-DFL exposure가 DFL-MVO보다 낮음"
    )

    print(
        "reduced_seeds     : "
        "10개 seed 중 Top-K exposure가 더 낮은 seed 수"
    )


def main():
    args = parse_args()

    if args.eigen_k != 5:
        print(
            "WARNING: 기존 high-lambda "
            "Top-K 실험은 K=5입니다."
        )

    output_dir = prepare_output_dir(
        args.output_dir,
        args.overwrite,
    )

    # 실제 Top-K high-lambda 모델과 동일:
    #
    # K = 5
    # preserve_residual_trace
    # eta = 0.75
    dataset = EigenRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,

        return_type="simple",
        covariance_jitter=1e-6,

        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,

        eigen_k=args.eigen_k,
        scale_mode=(
            "preserve_residual_trace"
        ),
    )

    (
        dates,
        raw_topk,
        model_penalty,
        covariance,
    ) = build_test_matrices(
        dataset=dataset,
        validation_end=(
            args.validation_end
        ),
        eta=args.eta,
    )

    tickers = [
        str(ticker)
        for ticker
        in dataset.tickers
    ]

    dfl_root = Path(
        args.dfl_root
    )

    topk_root = Path(
        args.topk_root
    )

    print(
        "=" * 100
    )
    print(
        "EXPERIMENT B | "
        "MECHANISM: EXPOSURE REDUCTION"
    )
    print(
        "=" * 100
    )

    print(
        f"Test period : "
        f"{dates.min().date()} "
        f"~ {dates.max().date()}"
    )

    print(
        f"K           : "
        f"{args.eigen_k}"
    )

    print(
        f"Eta         : "
        f"{args.eta}"
    )

    print(
        "Scale       : "
        "preserve_residual_trace"
    )

    print(
        f"Lambdas     : "
        f"{args.lambdas}"
    )

    print(
        f"Seeds       : "
        f"{args.seeds}"
    )

    print(
        "Q1~Q4 기준  : "
        "DFL-MVO raw Top-5 exposure"
    )

    print(
        "=" * 100
    )

    daily_frames = []
    seed_frames = []

    for risk_aversion in (
        args.lambdas
    ):

        for seed in (
            args.seeds
        ):

            dfl_weights = (
                load_dfl_weights(
                    root=dfl_root,
                    risk_aversion=(
                        risk_aversion
                    ),
                    seed=seed,
                    dates=dates,
                    tickers=tickers,
                )
            )

            topk_weights = (
                load_topk_weights(
                    root=topk_root,
                    risk_aversion=(
                        risk_aversion
                    ),
                    seed=seed,
                    dates=dates,
                    tickers=tickers,
                )
            )

            check_weights(
                dfl_weights,
                (
                    f"DFL lambda="
                    f"{risk_aversion:g} "
                    f"seed={seed}"
                ),
            )

            check_weights(
                topk_weights,
                (
                    f"TopK lambda="
                    f"{risk_aversion:g} "
                    f"seed={seed}"
                ),
            )

            daily = build_daily_frame(
                dates=dates,
                risk_aversion=(
                    risk_aversion
                ),
                seed=seed,
                dfl_weights=(
                    dfl_weights
                ),
                topk_weights=(
                    topk_weights
                ),
                raw_topk=raw_topk,
                model_penalty=(
                    model_penalty
                ),
                covariance=covariance,
            )

            daily_frames.append(
                daily
            )

            seed_summary = (
                summarize_seed(
                    frame=daily,
                    hac_lags=(
                        args.hac_lags
                    ),
                )
            )

            seed_frames.append(
                seed_summary
            )

            q4_raw = (
                seed_summary.loc[
                    (
                        seed_summary[
                            "segment"
                        ]
                        == "Q4_High"
                    )
                    &
                    (
                        seed_summary[
                            "metric"
                        ]
                        == "raw_top5"
                    ),
                    "reduction_pct",
                ]
                .iloc[0]
            )

            print(
                f"lambda="
                f"{risk_aversion:g} "
                f"| seed={seed} "
                f"| Q4 raw reduction="
                f"{q4_raw:+.2f}%"
            )

    daily_all = pd.concat(
        daily_frames,
        ignore_index=True,
    )

    by_seed = pd.concat(
        seed_frames,
        ignore_index=True,
    )

    summary = aggregate_summary(
        by_seed
    )

    daily_path = (
        output_dir
        / "experimentB_exposure_reduction_daily.csv"
    )

    seed_path = (
        output_dir
        / "experimentB_exposure_reduction_by_seed.csv"
    )

    summary_path = (
        output_dir
        / "experimentB_exposure_reduction_summary.csv"
    )

    daily_all.to_csv(
        daily_path,
        index=False,
    )

    by_seed.to_csv(
        seed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print_primary(
        summary
    )

    print()
    print(
        "Saved:"
    )
    print(daily_path)
    print(seed_path)
    print(summary_path)


if __name__ == "__main__":
    main()