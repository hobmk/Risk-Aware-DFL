from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset
from implementations.rcr_dfl.src.residual_risk import correlation_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual Collective Risk signal 자체의 구조적·미래위험 관련성을 검증한다."
    )
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--return-type", choices=["simple", "log"], default="simple")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--covariance-jitter", type=float, default=1e-6)
    parser.add_argument("--market-mode", choices=["equal_weight", "external"], default="equal_weight")
    parser.add_argument("--market-price-csv", default=None)
    parser.add_argument("--market-column", default=None)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--no-capm-intercept", action="store_true")
    parser.add_argument("--residual-correlation-shrinkage", type=float, default=0.0)
    parser.add_argument("--correlation-scaling", choices=["none", "trace"], default="trace")
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument(
        "--output-dir",
        default="implementations/rcr_dfl/outputs/rcr_signal_validation",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: str | Path, overwrite: bool) -> Path:
    output_dir = Path(path)

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"출력 폴더가 비어 있지 않습니다: {output_dir}\n"
            "다시 생성하려면 --overwrite를 추가하세요."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def mean_absolute_off_diagonal(
    matrices: torch.Tensor,
) -> np.ndarray:
    n_assets = matrices.size(-1)

    mask = ~torch.eye(
        n_assets,
        dtype=torch.bool,
        device=matrices.device,
    )

    values = matrices[:, mask].abs().mean(dim=1)

    return (
        values.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )


def largest_eigenvalue_share(
    matrices: torch.Tensor,
) -> np.ndarray:
    symmetric = 0.5 * (
        matrices + matrices.transpose(-1, -2)
    )

    largest = torch.linalg.eigvalsh(
        symmetric
    )[..., -1]

    values = largest / matrices.size(-1)

    return (
        values.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )


def build_raw_correlations(
    dataset: RCRRollingMVODataset,
) -> torch.Tensor:
    windows = np.lib.stride_tricks.sliding_window_view(
        dataset.returns,
        window_shape=dataset.lookback,
        axis=0,
    ).transpose(0, 2, 1)[:-1]

    if len(windows) != len(dataset):
        raise RuntimeError(
            "Raw correlation window 수와 "
            "RCR dataset sample 수가 다릅니다: "
            f"raw={len(windows)}, rcr={len(dataset)}"
        )

    windows_tensor = torch.from_numpy(
        np.ascontiguousarray(windows)
    ).to(torch.float64)

    return correlation_matrix(
        windows_tensor,
        eps=dataset.correlation_eps,
    )


def annualized_rms_volatility(
    returns: np.ndarray,
    periods_per_year: int,
) -> float:
    values = np.asarray(
        returns,
        dtype=np.float64,
    )

    if values.size == 0:
        return float("nan")

    return float(
        np.sqrt(
            periods_per_year
            * np.mean(np.square(values))
        )
    )


def maximum_drawdown(
    returns: np.ndarray,
    return_type: str,
) -> float:
    values = np.asarray(
        returns,
        dtype=np.float64,
    )

    if values.size == 0:
        return float("nan")

    if return_type == "simple":
        wealth = np.cumprod(1.0 + values)
    else:
        wealth = np.exp(np.cumsum(values))

    wealth = np.concatenate(
        ([1.0], wealth)
    )

    running_max = np.maximum.accumulate(
        wealth
    )

    drawdown = 1.0 - wealth / running_max

    return float(
        np.max(drawdown)
    )


def standard_covariance_volatility(
    covariances: torch.Tensor,
    periods_per_year: int,
) -> np.ndarray:
    n_assets = covariances.size(-1)

    weights = torch.full(
        (n_assets,),
        1.0 / n_assets,
        dtype=covariances.dtype,
        device=covariances.device,
    )

    variance = torch.einsum(
        "i,bij,j->b",
        weights,
        covariances,
        weights,
    )

    volatility = torch.sqrt(
        torch.clamp(
            variance,
            min=0.0,
        )
        * periods_per_year
    )

    return (
        volatility.detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )


def build_signal_frame(
    dataset: RCRRollingMVODataset,
    raw_correlations: torch.Tensor,
    periods_per_year: int,
) -> pd.DataFrame:
    residual = (
        dataset.residual_correlations_raw
        .to(torch.float64)
    )

    raw = raw_correlations.to(
        torch.float64
    )

    frame = pd.DataFrame(
        {
            "date": dataset.target_dates,

            "residual_marc":
                mean_absolute_off_diagonal(
                    residual
                ),

            "residual_les":
                largest_eigenvalue_share(
                    residual
                ),

            "raw_marc":
                mean_absolute_off_diagonal(
                    raw
                ),

            "raw_les":
                largest_eigenvalue_share(
                    raw
                ),

            "standard_cov_vol":
                standard_covariance_volatility(
                    dataset.covariances.to(
                        torch.float64
                    ),
                    periods_per_year,
                ),
        }
    )

    market_returns = np.asarray(
        dataset.market_returns,
        dtype=np.float64,
    )

    current_vol_20 = []
    future_vol_5 = []
    future_vol_20 = []
    future_mdd_20 = []

    for target_position in dataset.target_positions:
        position = int(
            target_position
        )

        # --------------------------------------------------
        # 현재 이용 가능한 과거 20일 변동성
        # [t-20, ..., t-1]
        # --------------------------------------------------
        current_start = max(
            0,
            position - 20,
        )

        current_vol_20.append(
            annualized_rms_volatility(
                market_returns[
                    current_start:position
                ],
                periods_per_year,
            )
        )

        # --------------------------------------------------
        # Future 5-day volatility
        # RCR은 t 직전까지의 정보만 사용하므로
        # target_date=t부터 미래 구간을 시작한다.
        # --------------------------------------------------
        if position + 5 <= len(
            market_returns
        ):
            future_vol_5.append(
                annualized_rms_volatility(
                    market_returns[
                        position:position + 5
                    ],
                    periods_per_year,
                )
            )
        else:
            future_vol_5.append(
                float("nan")
            )

        # --------------------------------------------------
        # Future 20-day volatility / MDD
        # --------------------------------------------------
        if position + 20 <= len(
            market_returns
        ):
            future_window = (
                market_returns[
                    position:position + 20
                ]
            )

            future_vol_20.append(
                annualized_rms_volatility(
                    future_window,
                    periods_per_year,
                )
            )

            future_mdd_20.append(
                maximum_drawdown(
                    future_window,
                    dataset.return_type,
                )
            )

        else:
            future_vol_20.append(
                float("nan")
            )
            future_mdd_20.append(
                float("nan")
            )

    frame[
        "current_market_vol_20"
    ] = current_vol_20

    frame[
        "future_market_vol_5"
    ] = future_vol_5

    frame[
        "future_market_vol_20"
    ] = future_vol_20

    frame[
        "future_market_mdd_20"
    ] = future_mdd_20

    return frame


def correlation_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    signals = [
        "residual_marc",
        "residual_les",
        "raw_marc",
        "raw_les",
    ]

    targets = [
        "current_market_vol_20",
        "future_market_vol_5",
        "future_market_vol_20",
        "future_market_mdd_20",
        "standard_cov_vol",
    ]

    rows = []

    for signal in signals:
        for target in targets:
            pair = (
                dataframe[
                    [signal, target]
                ]
                .dropna()
            )

            pearson = pearsonr(
                pair[signal],
                pair[target],
            )

            spearman = spearmanr(
                pair[signal],
                pair[target],
            )

            rows.append(
                {
                    "signal": signal,
                    "target": target,
                    "n": len(pair),

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


def save_line_plot(
    dataframe: pd.DataFrame,
    first: str,
    second: str,
    first_label: str,
    second_label: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.plot(
        dataframe["date"],
        dataframe[first],
        label=first_label,
        linewidth=1.5,
    )

    axis.plot(
        dataframe["date"],
        dataframe[second],
        label=second_label,
        linewidth=1.5,
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        y_label
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_scatter_plot(
    dataframe: pd.DataFrame,
    signal: str,
    target: str,
    output_path: Path,
) -> None:
    pair = (
        dataframe[
            [signal, target]
        ]
        .dropna()
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.scatter(
        pair[signal],
        pair[target],
        alpha=0.45,
        s=14,
    )

    slope, intercept = np.polyfit(
        pair[signal],
        pair[target],
        1,
    )

    x_values = np.linspace(
        pair[signal].min(),
        pair[signal].max(),
        200,
    )

    axis.plot(
        x_values,
        intercept
        + slope * x_values,
        linewidth=2,
    )

    axis.set_title(
        f"{signal} vs {target}"
    )

    axis.set_xlabel(
        signal
    )

    axis.set_ylabel(
        target
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    args = parse_args()

    output_dir = prepare_output_dir(
        args.output_dir,
        args.overwrite,
    )

    # ------------------------------------------------------
    # 기존 RCR pipeline과 동일한 dataset 재사용
    # ------------------------------------------------------
    dataset = RCRRollingMVODataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type=args.return_type,
        covariance_jitter=args.covariance_jitter,
        market_mode=args.market_mode,
        market_price_csv=args.market_price_csv,
        market_column=args.market_column,
        risk_free_rate=args.risk_free_rate,
        fit_intercept=not args.no_capm_intercept,
        residual_correlation_shrinkage=
            args.residual_correlation_shrinkage,
        correlation_scaling=
            args.correlation_scaling,

        # 현재 학습 pipeline과 동일하게
        # dataset output은 float32로 구성하고,
        # 분석 시 필요한 matrix는 float64로 변환한다.
        dtype=torch.float32,
    )

    # ------------------------------------------------------
    # Raw return correlation
    # CAPM residual correlation과 동일한 방식으로 계산
    # ------------------------------------------------------
    raw_correlations = (
        build_raw_correlations(
            dataset
        )
    )

    frame = build_signal_frame(
        dataset=dataset,
        raw_correlations=
            raw_correlations,
        periods_per_year=
            args.periods_per_year,
    )

    correlations = correlation_table(
        frame
    )

    # ------------------------------------------------------
    # CSV
    # ------------------------------------------------------
    frame.to_csv(
        output_dir
        / "rcr_signal_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    correlations.to_csv(
        output_dir
        / "correlation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------
    # Figure 1: MARC residual vs raw
    # ------------------------------------------------------
    save_line_plot(
        dataframe=frame,
        first="residual_marc",
        second="raw_marc",
        first_label="Residual MARC",
        second_label="Raw MARC",
        title=(
            "Mean Absolute Correlation: "
            "Residual vs Raw"
        ),
        y_label=(
            "Mean Absolute Correlation"
        ),
        output_path=(
            output_dir
            / "timeseries_marc_residual_vs_raw.png"
        ),
    )

    # ------------------------------------------------------
    # Figure 2: LES residual vs raw
    # ------------------------------------------------------
    save_line_plot(
        dataframe=frame,
        first="residual_les",
        second="raw_les",
        first_label="Residual LES",
        second_label="Raw LES",
        title=(
            "Largest Eigenvalue Share: "
            "Residual vs Raw"
        ),
        y_label=(
            "Largest Eigenvalue Share"
        ),
        output_path=(
            output_dir
            / "timeseries_les_residual_vs_raw.png"
        ),
    )

    # ------------------------------------------------------
    # Figure 3~4: RCR vs future 20-day volatility
    # ------------------------------------------------------
    save_scatter_plot(
        dataframe=frame,
        signal="residual_marc",
        target="future_market_vol_20",
        output_path=(
            output_dir
            / "scatter_residual_marc_vs_future20_vol.png"
        ),
    )

    save_scatter_plot(
        dataframe=frame,
        signal="residual_les",
        target="future_market_vol_20",
        output_path=(
            output_dir
            / "scatter_residual_les_vs_future20_vol.png"
        ),
    )

    # ------------------------------------------------------
    # Config
    # ------------------------------------------------------
    config = vars(
        args
    ).copy()

    config.update(
        {
            "n_assets":
                dataset.n_assets,

            "n_samples":
                len(dataset),

            "market_name":
                dataset.market_name,

            "first_signal_date":
                str(
                    dataset
                    .target_dates[0]
                    .date()
                ),

            "last_signal_date":
                str(
                    dataset
                    .target_dates[-1]
                    .date()
                ),

            "signal_definition":
                (
                    "Residual correlation "
                    "estimated from the "
                    "preceding lookback window"
                ),

            "future_window_definition":
                (
                    "Future window begins "
                    "on target_date"
                ),
        }
    )

    with (
        output_dir / "config.json"
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

    # ------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------
    future_20 = correlations.loc[
        correlations["target"]
        == "future_market_vol_20",
        [
            "signal",
            "n",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p",
        ],
    ]

    current_20 = correlations.loc[
        correlations["target"]
        == "current_market_vol_20",
        [
            "signal",
            "n",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p",
        ],
    ]

    print(
        "\n" + "=" * 100
    )

    print(
        "RCR SIGNAL VALIDATION"
    )

    print(
        "=" * 100
    )

    print(
        f"assets     : "
        f"{dataset.n_assets}"
    )

    print(
        f"samples    : "
        f"{len(dataset)}"
    )

    print(
        f"market     : "
        f"{dataset.market_name}"
    )

    print(
        f"lookback   : "
        f"{dataset.lookback}"
    )

    print(
        f"date range : "
        f"{dataset.target_dates[0].date()} "
        f"~ "
        f"{dataset.target_dates[-1].date()}"
    )

    print(
        "future window starts on "
        "target_date; RCR uses only "
        "the preceding lookback window"
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "CURRENT 20-DAY MARKET "
        "VOLATILITY CORRELATION"
    )

    print(
        "=" * 100
    )

    print(
        current_20
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "FUTURE 20-DAY MARKET "
        "VOLATILITY CORRELATION"
    )

    print(
        "=" * 100
    )

    print(
        future_20
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "MEAN SIGNAL LEVELS"
    )

    print(
        "=" * 100
    )

    print(
        frame[
            [
                "residual_marc",
                "residual_les",
                "raw_marc",
                "raw_les",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .round(6)
        .to_string()
    )

    print(
        "\nSaved:"
    )

    print(
        output_dir
        / "rcr_signal_timeseries.csv"
    )

    print(
        output_dir
        / "correlation_summary.csv"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()