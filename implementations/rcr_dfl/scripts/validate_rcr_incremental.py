from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RCR signal이 기존 covariance risk 및 raw correlation을 "
            "통제한 이후에도 미래 위험에 추가 정보를 제공하는지 검증한다."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=(
            "implementations/rcr_dfl/outputs/"
            "rcr_signal_validation/rcr_signal_timeseries.csv"
        ),
    )
    parser.add_argument(
        "--target",
        default="future_market_vol_20",
    )
    parser.add_argument(
        "--hac-lags",
        type=int,
        default=19,
        help=(
            "Newey-West HAC maximum lag. "
            "20-day overlapping future target의 기본값은 19."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "implementations/rcr_dfl/outputs/"
            "rcr_signal_validation/incremental_validation"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def prepare_output_dir(
    path: str | Path,
    overwrite: bool,
) -> Path:
    output_dir = Path(path)

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not overwrite
    ):
        raise FileExistsError(
            f"출력 폴더가 비어 있지 않습니다: {output_dir}\n"
            "--overwrite를 추가하면 기존 분석 결과를 갱신합니다."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def standardize(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = values.mean()
    std = values.std(ddof=0)

    if not np.isfinite(std) or std <= 0:
        raise ValueError(
            "표준화하려는 변수의 표준편차가 0입니다."
        )

    return (
        values - mean
    ) / std


def build_design_matrix(
    dataframe: pd.DataFrame,
    predictors: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_columns = [
        standardize(
            dataframe[column].to_numpy()
        )
        for column in predictors
    ]

    x = np.column_stack(
        [
            np.ones(
                len(dataframe),
                dtype=np.float64,
            ),
            *x_columns,
        ]
    )

    return x, np.asarray(
        predictors,
        dtype=object,
    )


def ols_fit(
    y: np.ndarray,
    x: np.ndarray,
) -> dict[str, np.ndarray | float]:
    beta = np.linalg.lstsq(
        x,
        y,
        rcond=None,
    )[0]

    fitted = x @ beta
    residuals = y - fitted

    ss_res = float(
        residuals @ residuals
    )

    centered = y - y.mean()

    ss_tot = float(
        centered @ centered
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else float("nan")
    )

    n = len(y)
    k = x.shape[1] - 1

    adj_r2 = (
        1.0
        - (1.0 - r2)
        * (n - 1)
        / (n - k - 1)
        if n > k + 1
        else float("nan")
    )

    return {
        "beta": beta,
        "fitted": fitted,
        "residuals": residuals,
        "r2": r2,
        "adj_r2": adj_r2,
    }


def newey_west_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    n, k = x.shape

    if max_lag < 0:
        raise ValueError(
            "hac-lags는 0 이상이어야 합니다."
        )

    max_lag = min(
        max_lag,
        n - 1,
    )

    xtx_inv = np.linalg.pinv(
        x.T @ x
    )

    xu = (
        x
        * residuals[:, None]
    )

    meat = xu.T @ xu

    for lag in range(
        1,
        max_lag + 1,
    ):
        weight = (
            1.0
            - lag
            / (max_lag + 1.0)
        )

        gamma = (
            xu[lag:].T
            @ xu[:-lag]
        )

        meat += weight * (
            gamma + gamma.T
        )

    covariance = (
        xtx_inv
        @ meat
        @ xtx_inv
    )

    # HC1-style finite-sample correction
    if n > k:
        covariance *= (
            n / (n - k)
        )

    return covariance


def run_regression(
    dataframe: pd.DataFrame,
    target: str,
    predictors: list[str],
    hac_lags: int,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    required = [
        target,
        *predictors,
    ]

    sample = (
        dataframe[
            required
        ]
        .dropna()
        .reset_index(drop=True)
    )

    y = standardize(
        sample[target].to_numpy()
    )

    x, predictor_names = (
        build_design_matrix(
            sample,
            predictors,
        )
    )

    fit = ols_fit(
        y=y,
        x=x,
    )

    covariance = (
        newey_west_covariance(
            x=x,
            residuals=fit["residuals"],
            max_lag=hac_lags,
        )
    )

    standard_errors = np.sqrt(
        np.clip(
            np.diag(covariance),
            a_min=0.0,
            a_max=None,
        )
    )

    beta = fit["beta"]

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        t_values = (
            beta / standard_errors
        )

    p_values = (
        2.0
        * norm.sf(
            np.abs(t_values)
        )
    )

    names = [
        "Intercept",
        *predictor_names.tolist(),
    ]

    coefficient_rows = []

    for index, name in enumerate(
        names
    ):
        coefficient_rows.append(
            {
                "term": name,
                "standardized_beta":
                    float(beta[index]),
                "hac_se":
                    float(
                        standard_errors[index]
                    ),
                "hac_t":
                    float(
                        t_values[index]
                    ),
                "hac_p":
                    float(
                        p_values[index]
                    ),
            }
        )

    model_result = {
        "n": int(
            len(sample)
        ),
        "n_predictors": int(
            len(predictors)
        ),
        "r2": float(
            fit["r2"]
        ),
        "adj_r2": float(
            fit["adj_r2"]
        ),
    }

    return (
        model_result,
        coefficient_rows,
    )


def build_models() -> dict[str, list[str]]:
    return {
        "M0_standard": [
            "standard_cov_vol",
        ],

        "M1_standard_plus_residual_marc": [
            "standard_cov_vol",
            "residual_marc",
        ],

        "M2_standard_plus_residual_les": [
            "standard_cov_vol",
            "residual_les",
        ],

        "M3_standard_plus_raw_marc": [
            "standard_cov_vol",
            "raw_marc",
        ],

        "M4_standard_raw_plus_residual_marc": [
            "standard_cov_vol",
            "raw_marc",
            "residual_marc",
        ],

        "M5_standard_raw_plus_residual_les": [
            "standard_cov_vol",
            "raw_marc",
            "residual_les",
        ],
    }


def run_all_models(
    dataframe: pd.DataFrame,
    target: str,
    hac_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = build_models()

    model_rows = []
    coefficient_rows = []

    for model_name, predictors in (
        models.items()
    ):
        result, coefficients = (
            run_regression(
                dataframe=dataframe,
                target=target,
                predictors=predictors,
                hac_lags=hac_lags,
            )
        )

        model_rows.append(
            {
                "model": model_name,
                "predictors": " + ".join(
                    predictors
                ),
                **result,
            }
        )

        for row in coefficients:
            coefficient_rows.append(
                {
                    "model": model_name,
                    **row,
                }
            )

    models_df = pd.DataFrame(
        model_rows
    )

    coefficients_df = pd.DataFrame(
        coefficient_rows
    )

    baseline_r2 = float(
        models_df.loc[
            models_df["model"]
            == "M0_standard",
            "r2",
        ].iloc[0]
    )

    baseline_adj_r2 = float(
        models_df.loc[
            models_df["model"]
            == "M0_standard",
            "adj_r2",
        ].iloc[0]
    )

    models_df[
        "delta_r2_vs_standard"
    ] = (
        models_df["r2"]
        - baseline_r2
    )

    models_df[
        "delta_adj_r2_vs_standard"
    ] = (
        models_df["adj_r2"]
        - baseline_adj_r2
    )

    # Raw correlation이 들어간 모델 M3와 비교하여
    # residual signal의 추가 설명력을 별도로 계산
    raw_r2 = float(
        models_df.loc[
            models_df["model"]
            == "M3_standard_plus_raw_marc",
            "r2",
        ].iloc[0]
    )

    raw_adj_r2 = float(
        models_df.loc[
            models_df["model"]
            == "M3_standard_plus_raw_marc",
            "adj_r2",
        ].iloc[0]
    )

    models_df[
        "delta_r2_vs_standard_raw"
    ] = np.nan

    models_df[
        "delta_adj_r2_vs_standard_raw"
    ] = np.nan

    mask = models_df[
        "model"
    ].isin(
        [
            "M4_standard_raw_plus_residual_marc",
            "M5_standard_raw_plus_residual_les",
        ]
    )

    models_df.loc[
        mask,
        "delta_r2_vs_standard_raw",
    ] = (
        models_df.loc[
            mask,
            "r2",
        ]
        - raw_r2
    )

    models_df.loc[
        mask,
        "delta_adj_r2_vs_standard_raw",
    ] = (
        models_df.loc[
            mask,
            "adj_r2",
        ]
        - raw_adj_r2
    )

    return (
        models_df,
        coefficients_df,
    )


def build_regime_analysis(
    dataframe: pd.DataFrame,
    target: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    signals = [
        "residual_marc",
        "residual_les",
        "raw_marc",
        "raw_les",
    ]

    required = [
        "date",
        target,
        *signals,
    ]

    sample = (
        dataframe[
            required
        ]
        .dropna()
        .copy()
    )

    q25 = float(
        sample[target].quantile(
            0.25
        )
    )

    q75 = float(
        sample[target].quantile(
            0.75
        )
    )

    sample["risk_regime"] = np.select(
        [
            sample[target] <= q25,
            sample[target] >= q75,
        ],
        [
            "Low",
            "High",
        ],
        default="Normal",
    )

    regime_order = [
        "Low",
        "Normal",
        "High",
    ]

    sample[
        "risk_regime"
    ] = pd.Categorical(
        sample["risk_regime"],
        categories=regime_order,
        ordered=True,
    )

    rows = []

    for regime in regime_order:
        regime_data = sample.loc[
            sample["risk_regime"]
            == regime
        ]

        for signal in signals:
            values = regime_data[
                signal
            ]

            rows.append(
                {
                    "regime": regime,
                    "signal": signal,
                    "n": len(values),
                    "mean":
                        float(
                            values.mean()
                        ),
                    "median":
                        float(
                            values.median()
                        ),
                    "std":
                        float(
                            values.std(ddof=1)
                        ),
                }
            )

    summary = pd.DataFrame(
        rows
    )

    test_rows = []

    for signal in signals:
        low = sample.loc[
            sample["risk_regime"]
            == "Low",
            signal,
        ].to_numpy()

        high = sample.loc[
            sample["risk_regime"]
            == "High",
            signal,
        ].to_numpy()

        test = mannwhitneyu(
            high,
            low,
            alternative="two-sided",
        )

        test_rows.append(
            {
                "signal": signal,

                "low_mean":
                    float(
                        np.mean(low)
                    ),

                "high_mean":
                    float(
                        np.mean(high)
                    ),

                "high_minus_low":
                    float(
                        np.mean(high)
                        - np.mean(low)
                    ),

                "high_vs_low_pct":
                    float(
                        (
                            np.mean(high)
                            / np.mean(low)
                            - 1.0
                        )
                        * 100.0
                    ),

                "mannwhitney_u":
                    float(
                        test.statistic
                    ),

                "mannwhitney_p":
                    float(
                        test.pvalue
                    ),
            }
        )

    tests = pd.DataFrame(
        test_rows
    )

    thresholds = pd.DataFrame(
        [
            {
                "target": target,
                "low_threshold_q25":
                    q25,
                "high_threshold_q75":
                    q75,
            }
        ]
    )

    return (
        summary,
        tests,
        thresholds,
    )


def save_regime_boxplot(
    dataframe: pd.DataFrame,
    target: str,
    signal: str,
    output_path: Path,
) -> None:
    sample = (
        dataframe[
            [target, signal]
        ]
        .dropna()
        .copy()
    )

    q25 = sample[
        target
    ].quantile(
        0.25
    )

    q75 = sample[
        target
    ].quantile(
        0.75
    )

    sample["risk_regime"] = np.select(
        [
            sample[target] <= q25,
            sample[target] >= q75,
        ],
        [
            "Low",
            "High",
        ],
        default="Normal",
    )

    values = [
        sample.loc[
            sample["risk_regime"]
            == regime,
            signal,
        ].to_numpy()
        for regime in [
            "Low",
            "Normal",
            "High",
        ]
    ]

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.boxplot(
        values,
        tick_labels=[
            "Low",
            "Normal",
            "High",
        ],
        showfliers=False,
    )

    axis.set_title(
        f"{signal} by Future Risk Regime"
    )

    axis.set_xlabel(
        "Future 20-Day Volatility Regime"
    )

    axis.set_ylabel(
        signal
    )

    axis.grid(
        True,
        axis="y",
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


def save_incremental_r2_plot(
    models: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_data = (
        models[
            [
                "model",
                "delta_r2_vs_standard",
            ]
        ]
        .copy()
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.bar(
        plot_data["model"],
        plot_data[
            "delta_r2_vs_standard"
        ],
    )

    axis.set_title(
        "Incremental R-squared vs Standard Risk"
    )

    axis.set_xlabel(
        "Regression Model"
    )

    axis.set_ylabel(
        "Delta R-squared"
    )

    axis.tick_params(
        axis="x",
        rotation=35,
    )

    axis.grid(
        True,
        axis="y",
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

    input_path = Path(
        args.input_csv
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"입력 CSV를 찾을 수 없습니다: {input_path}"
        )

    output_dir = prepare_output_dir(
        args.output_dir,
        args.overwrite,
    )

    dataframe = pd.read_csv(
        input_path
    )

    if "date" in dataframe.columns:
        dataframe["date"] = pd.to_datetime(
            dataframe["date"]
        )

    required = {
        args.target,
        "standard_cov_vol",
        "residual_marc",
        "residual_les",
        "raw_marc",
        "raw_les",
    }

    missing = required.difference(
        dataframe.columns
    )

    if missing:
        raise KeyError(
            "필요한 column이 없습니다: "
            f"{sorted(missing)}"
        )

    models, coefficients = (
        run_all_models(
            dataframe=dataframe,
            target=args.target,
            hac_lags=args.hac_lags,
        )
    )

    (
        regime_summary,
        regime_tests,
        regime_thresholds,
    ) = build_regime_analysis(
        dataframe=dataframe,
        target=args.target,
    )

    models.to_csv(
        output_dir
        / "regression_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coefficients.to_csv(
        output_dir
        / "regression_coefficients_hac.csv",
        index=False,
        encoding="utf-8-sig",
    )

    regime_summary.to_csv(
        output_dir
        / "risk_regime_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    regime_tests.to_csv(
        output_dir
        / "risk_regime_high_vs_low_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    regime_thresholds.to_csv(
        output_dir
        / "risk_regime_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_regime_boxplot(
        dataframe=dataframe,
        target=args.target,
        signal="residual_marc",
        output_path=(
            output_dir
            / "boxplot_residual_marc_by_future_risk.png"
        ),
    )

    save_regime_boxplot(
        dataframe=dataframe,
        target=args.target,
        signal="residual_les",
        output_path=(
            output_dir
            / "boxplot_residual_les_by_future_risk.png"
        ),
    )

    save_incremental_r2_plot(
        models=models,
        output_path=(
            output_dir
            / "incremental_r2.png"
        ),
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "RCR INCREMENTAL INFORMATION VALIDATION"
    )

    print(
        "=" * 110
    )

    print(
        f"input     : {input_path}"
    )

    print(
        f"target    : {args.target}"
    )

    print(
        f"HAC lags  : {args.hac_lags}"
    )

    print(
        "\nStandardized regression:"
    )

    print(
        "Target와 predictors를 모두 z-score 표준화한 뒤 "
        "OLS + Newey-West HAC inference를 수행함."
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "MODEL FIT / INCREMENTAL R-SQUARED"
    )

    print(
        "=" * 110
    )

    model_display = models[
        [
            "model",
            "n",
            "r2",
            "adj_r2",
            "delta_r2_vs_standard",
            "delta_adj_r2_vs_standard",
            "delta_r2_vs_standard_raw",
        ]
    ].copy()

    print(
        model_display
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "KEY HAC COEFFICIENTS"
    )

    print(
        "=" * 110
    )

    key_models = {
        "M1_standard_plus_residual_marc":
            "residual_marc",

        "M2_standard_plus_residual_les":
            "residual_les",

        "M3_standard_plus_raw_marc":
            "raw_marc",

        "M4_standard_raw_plus_residual_marc":
            "residual_marc",

        "M5_standard_raw_plus_residual_les":
            "residual_les",
    }

    key_rows = []

    for model_name, term in (
        key_models.items()
    ):
        row = coefficients.loc[
            (
                coefficients["model"]
                == model_name
            )
            & (
                coefficients["term"]
                == term
            )
        ].iloc[0]

        key_rows.append(
            {
                "model": model_name,
                "term": term,
                "std_beta":
                    row[
                        "standardized_beta"
                    ],
                "hac_se":
                    row["hac_se"],
                "hac_t":
                    row["hac_t"],
                "hac_p":
                    row["hac_p"],
            }
        )

    key_df = pd.DataFrame(
        key_rows
    )

    print(
        key_df
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "FUTURE-RISK REGIME SUMMARY"
    )

    print(
        "=" * 110
    )

    regime_display = (
        regime_summary.loc[
            regime_summary[
                "signal"
            ].isin(
                [
                    "residual_marc",
                    "residual_les",
                ]
            )
        ]
        .pivot(
            index="regime",
            columns="signal",
            values="mean",
        )
        .reindex(
            [
                "Low",
                "Normal",
                "High",
            ]
        )
    )

    print(
        regime_display
        .round(6)
        .to_string()
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "HIGH vs LOW FUTURE-RISK REGIME"
    )

    print(
        "=" * 110
    )

    print(
        regime_tests[
            [
                "signal",
                "low_mean",
                "high_mean",
                "high_minus_low",
                "high_vs_low_pct",
                "mannwhitney_p",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "INTERPRETATION CHECKPOINT"
    )

    print(
        "=" * 110
    )

    m0 = models.loc[
        models["model"]
        == "M0_standard"
    ].iloc[0]

    m1 = models.loc[
        models["model"]
        == "M1_standard_plus_residual_marc"
    ].iloc[0]

    m3 = models.loc[
        models["model"]
        == "M3_standard_plus_raw_marc"
    ].iloc[0]

    m4 = models.loc[
        models["model"]
        == "M4_standard_raw_plus_residual_marc"
    ].iloc[0]

    print(
        f"Standard risk R2                    : "
        f"{m0['r2']:.6f}"
    )

    print(
        f"+ Residual MARC R2                 : "
        f"{m1['r2']:.6f}"
    )

    print(
        f"Residual MARC incremental R2       : "
        f"{m1['r2'] - m0['r2']:.6f}"
    )

    print(
        f"Standard + Raw MARC R2             : "
        f"{m3['r2']:.6f}"
    )

    print(
        f"+ Raw MARC + Residual MARC R2      : "
        f"{m4['r2']:.6f}"
    )

    print(
        f"Residual MARC incremental R2 "
        f"after Raw MARC control             : "
        f"{m4['r2'] - m3['r2']:.6f}"
    )

    print(
        "\nSaved:"
    )

    print(
        output_dir
        / "regression_model_summary.csv"
    )

    print(
        output_dir
        / "regression_coefficients_hac.csv"
    )

    print(
        output_dir
        / "risk_regime_summary.csv"
    )

    print(
        output_dir
        / "risk_regime_high_vs_low_tests.csv"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()