from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, pearsonr, spearmanr

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-matrix A_res의 portfolio-level risk exposure를 검증한다."
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
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--hac-lags", type=int, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--output-dir",
        default="implementations/rcr_dfl/outputs/rcr_matrix_exposure_validation",
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


def quadratic_form(
    matrices: torch.Tensor,
    weights: torch.Tensor,
) -> np.ndarray:
    values = torch.einsum(
        "i,bij,j->b",
        weights,
        matrices,
        weights,
    )

    return values.detach().cpu().numpy()


def maximum_drawdown(
    returns: np.ndarray,
    return_type: str,
) -> float:
    if return_type == "simple":
        wealth = np.cumprod(1.0 + returns)
    else:
        wealth = np.exp(np.cumsum(returns))

    wealth = np.concatenate(([1.0], wealth))
    peak = np.maximum.accumulate(wealth)
    drawdown = 1.0 - wealth / peak

    return float(drawdown.max())


def build_exposure_frame(
    dataset: RCRRollingMVODataset,
    eta: float,
    horizon: int,
    periods_per_year: int,
) -> pd.DataFrame:
    n_assets = dataset.n_assets

    weights = torch.full(
        (n_assets,),
        1.0 / n_assets,
        dtype=torch.float64,
    )

    sigma = dataset.covariances.to(torch.float64)
    a_res = dataset.a_res_matrices.to(torch.float64)

    # ------------------------------------------------------
    # Full quadratic-form exposure
    # ------------------------------------------------------
    sigma_daily = quadratic_form(
        sigma,
        weights,
    )

    a_res_total_daily = quadratic_form(
        a_res,
        weights,
    )

    # ------------------------------------------------------
    # A_res diagonal / off-diagonal decomposition
    #
    # total = diagonal + cross-asset dependency
    # ------------------------------------------------------
    diagonal = torch.diagonal(
        a_res,
        dim1=-2,
        dim2=-1,
    )

    a_res_diag_daily = (
        diagonal
        * weights.square().unsqueeze(0)
    ).sum(dim=1).cpu().numpy()

    a_res_offdiag_daily = (
        a_res_total_daily
        - a_res_diag_daily
    )

    effective_daily = (
        sigma_daily
        + eta * a_res_total_daily
    )

    # annualize: variance unit
    sigma_ann = sigma_daily * periods_per_year
    a_res_total_ann = a_res_total_daily * periods_per_year
    a_res_diag_ann = a_res_diag_daily * periods_per_year
    a_res_offdiag_ann = a_res_offdiag_daily * periods_per_year
    effective_ann = effective_daily * periods_per_year

    future_variance = []
    future_volatility = []
    future_mdd = []

    for position in dataset.target_positions:
        position = int(position)

        if position + horizon > len(dataset.returns):
            future_variance.append(np.nan)
            future_volatility.append(np.nan)
            future_mdd.append(np.nan)
            continue

        # Daily-rebalanced Equal-Weight portfolio return
        future_asset_returns = dataset.returns[
            position:position + horizon
        ]

        future_portfolio_returns = (
            future_asset_returns.mean(axis=1)
        )

        realized_variance = (
            np.var(
                future_portfolio_returns,
                ddof=1,
            )
            * periods_per_year
        )

        future_variance.append(
            float(realized_variance)
        )

        future_volatility.append(
            float(
                np.sqrt(
                    max(
                        realized_variance,
                        0.0,
                    )
                )
            )
        )

        future_mdd.append(
            maximum_drawdown(
                future_portfolio_returns,
                dataset.return_type,
            )
        )

    frame = pd.DataFrame(
        {
            "date": dataset.target_dates,
            "sigma_exposure": sigma_ann,
            "a_res_total_exposure": a_res_total_ann,
            "a_res_diag_exposure": a_res_diag_ann,
            "a_res_offdiag_exposure": a_res_offdiag_ann,
            "effective_exposure": effective_ann,
            "future_ew_realized_variance": future_variance,
            "future_ew_realized_volatility": future_volatility,
            "future_ew_mdd": future_mdd,
        }
    )

    frame["a_res_to_sigma_ratio"] = (
        frame["a_res_total_exposure"]
        / frame["sigma_exposure"]
    )

    frame["a_res_offdiag_to_sigma_ratio"] = (
        frame["a_res_offdiag_exposure"]
        / frame["sigma_exposure"]
    )

    return frame


def filter_dates(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    result = frame.copy()

    if start_date is not None:
        result = result.loc[
            result["date"]
            >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        result = result.loc[
            result["date"]
            <= pd.Timestamp(end_date)
        ]

    return result.reset_index(drop=True)


def correlation_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    signals = [
        "sigma_exposure",
        "a_res_total_exposure",
        "a_res_offdiag_exposure",
        "effective_exposure",
        "a_res_to_sigma_ratio",
    ]

    targets = [
        "future_ew_realized_variance",
        "future_ew_realized_volatility",
        "future_ew_mdd",
    ]

    rows = []

    for signal in signals:
        for target in targets:
            sample = frame[
                [signal, target]
            ].dropna()

            pearson = pearsonr(
                sample[signal],
                sample[target],
            )

            spearman = spearmanr(
                sample[signal],
                sample[target],
            )

            rows.append(
                {
                    "signal": signal,
                    "target": target,
                    "n": len(sample),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                }
            )

    return pd.DataFrame(rows)


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    std = values.std(ddof=0)

    if std <= 0:
        raise ValueError(
            "표준편차가 0인 변수가 있습니다."
        )

    return (
        values - values.mean()
    ) / std


def newey_west_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    n, k = x.shape

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

    if n > k:
        covariance *= (
            n / (n - k)
        )

    return covariance


def run_regression(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    hac_lags: int,
) -> tuple[dict, list[dict]]:
    columns = [
        target,
        *predictors,
    ]

    sample = frame[
        columns
    ].dropna()

    y = standardize(
        sample[target].to_numpy()
    )

    standardized_x = [
        standardize(
            sample[column].to_numpy()
        )
        for column in predictors
    ]

    x = np.column_stack(
        [
            np.ones(len(sample)),
            *standardized_x,
        ]
    )

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

    r2 = 1.0 - ss_res / ss_tot

    n = len(sample)
    k = len(predictors)

    adj_r2 = (
        1.0
        - (1.0 - r2)
        * (n - 1)
        / (n - k - 1)
    )

    hac_cov = newey_west_covariance(
        x=x,
        residuals=residuals,
        max_lag=hac_lags,
    )

    se = np.sqrt(
        np.clip(
            np.diag(hac_cov),
            0.0,
            None,
        )
    )

    t_values = beta / se

    p_values = (
        2.0
        * norm.sf(
            np.abs(t_values)
        )
    )

    names = [
        "Intercept",
        *predictors,
    ]

    coefficients = []

    for i, name in enumerate(names):
        coefficients.append(
            {
                "term": name,
                "standardized_beta": float(beta[i]),
                "hac_se": float(se[i]),
                "hac_t": float(t_values[i]),
                "hac_p": float(p_values[i]),
            }
        )

    summary = {
        "n": n,
        "r2": r2,
        "adj_r2": adj_r2,
    }

    return summary, coefficients


def run_regression_models(
    frame: pd.DataFrame,
    target: str,
    hac_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = {
        "M0_standard": [
            "sigma_exposure",
        ],
        "M1_standard_plus_Ares": [
            "sigma_exposure",
            "a_res_total_exposure",
        ],
        "M2_standard_plus_Ares_offdiag": [
            "sigma_exposure",
            "a_res_offdiag_exposure",
        ],
        "M3_effective_eta": [
            "effective_exposure",
        ],
        "M4_Ares_only": [
            "a_res_total_exposure",
        ],
        "M5_Ares_offdiag_only": [
            "a_res_offdiag_exposure",
        ],
    }

    model_rows = []
    coefficient_rows = []

    for model_name, predictors in models.items():
        summary, coefficients = run_regression(
            frame=frame,
            target=target,
            predictors=predictors,
            hac_lags=hac_lags,
        )

        model_rows.append(
            {
                "target": target,
                "model": model_name,
                "predictors": " + ".join(predictors),
                **summary,
            }
        )

        for row in coefficients:
            coefficient_rows.append(
                {
                    "target": target,
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

    models_df["delta_r2_vs_standard"] = (
        models_df["r2"]
        - baseline_r2
    )

    models_df[
        "delta_adj_r2_vs_standard"
    ] = (
        models_df["adj_r2"]
        - baseline_adj_r2
    )

    return (
        models_df,
        coefficients_df,
    )


def exposure_quartile_summary(
    frame: pd.DataFrame,
    signal: str,
) -> pd.DataFrame:
    sample = frame[
        [
            signal,
            "future_ew_realized_variance",
            "future_ew_realized_volatility",
            "future_ew_mdd",
        ]
    ].dropna().copy()

    sample["quartile"] = pd.qcut(
        sample[signal],
        q=4,
        labels=[
            "Q1_Low",
            "Q2",
            "Q3",
            "Q4_High",
        ],
    )

    summary = (
        sample.groupby(
            "quartile",
            observed=True,
        )
        .agg(
            n=(signal, "size"),
            mean_exposure=(signal, "mean"),
            future_variance=(
                "future_ew_realized_variance",
                "mean",
            ),
            future_volatility=(
                "future_ew_realized_volatility",
                "mean",
            ),
            future_mdd=(
                "future_ew_mdd",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.insert(
        0,
        "signal",
        signal,
    )

    return summary


def save_quartile_plot(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.bar(
        summary["quartile"].astype(str),
        summary["future_variance"],
    )

    axis.set_title(
        "Future 20-Day EW Risk by Current RCR Exposure"
    )

    axis.set_xlabel(
        "Current A_res Exposure Quartile"
    )

    axis.set_ylabel(
        "Future Annualized Realized Variance"
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

    plt.close(figure)


def main() -> None:
    args = parse_args()

    if args.horizon < 2:
        raise ValueError(
            "--horizon은 2 이상이어야 합니다."
        )

    if args.eta < 0:
        raise ValueError(
            "--eta는 0 이상이어야 합니다."
        )

    hac_lags = (
        args.horizon - 1
        if args.hac_lags is None
        else args.hac_lags
    )

    output_dir = prepare_output_dir(
        args.output_dir,
        args.overwrite,
    )

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
        dtype=torch.float32,
    )

    frame = build_exposure_frame(
        dataset=dataset,
        eta=args.eta,
        horizon=args.horizon,
        periods_per_year=
            args.periods_per_year,
    )

    frame = filter_dates(
        frame,
        args.start_date,
        args.end_date,
    )

    correlations = correlation_table(
        frame
    )

    variance_models, variance_coefficients = (
        run_regression_models(
            frame=frame,
            target="future_ew_realized_variance",
            hac_lags=hac_lags,
        )
    )

    volatility_models, volatility_coefficients = (
        run_regression_models(
            frame=frame,
            target="future_ew_realized_volatility",
            hac_lags=hac_lags,
        )
    )

    models = pd.concat(
        [
            variance_models,
            volatility_models,
        ],
        ignore_index=True,
    )

    coefficients = pd.concat(
        [
            variance_coefficients,
            volatility_coefficients,
        ],
        ignore_index=True,
    )

    total_quartiles = exposure_quartile_summary(
        frame,
        "a_res_total_exposure",
    )

    offdiag_quartiles = exposure_quartile_summary(
        frame,
        "a_res_offdiag_exposure",
    )

    quartiles = pd.concat(
        [
            total_quartiles,
            offdiag_quartiles,
        ],
        ignore_index=True,
    )

    frame.to_csv(
        output_dir
        / "rcr_matrix_exposure_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    correlations.to_csv(
        output_dir
        / "exposure_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    models.to_csv(
        output_dir
        / "exposure_regression_models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coefficients.to_csv(
        output_dir
        / "exposure_regression_coefficients_hac.csv",
        index=False,
        encoding="utf-8-sig",
    )

    quartiles.to_csv(
        output_dir
        / "exposure_quartile_future_risk.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_quartile_plot(
        total_quartiles,
        output_dir
        / "ares_total_exposure_quartile_future_risk.png",
    )

    config = vars(args).copy()
    config["resolved_hac_lags"] = hac_lags
    config["n_assets"] = dataset.n_assets
    config["n_samples"] = len(frame)
    config["market_name"] = dataset.market_name

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

    print(
        "\n" + "=" * 110
    )
    print(
        "FULL-MATRIX RCR EXPOSURE VALIDATION"
    )
    print(
        "=" * 110
    )

    print(
        f"assets      : {dataset.n_assets}"
    )
    print(
        f"samples     : {len(frame)}"
    )
    print(
        f"lookback    : {args.lookback}"
    )
    print(
        f"horizon     : {args.horizon}"
    )
    print(
        f"eta         : {args.eta}"
    )
    print(
        f"rho         : "
        f"{args.residual_correlation_shrinkage}"
    )
    print(
        f"scaling     : "
        f"{args.correlation_scaling}"
    )
    print(
        f"HAC lags    : {hac_lags}"
    )
    print(
        f"date range  : "
        f"{frame['date'].min().date()} "
        f"~ "
        f"{frame['date'].max().date()}"
    )

    print(
        "\n" + "=" * 110
    )
    print(
        "EXPOSURE CORRELATION WITH FUTURE 20-DAY EW RISK"
    )
    print(
        "=" * 110
    )

    correlation_display = correlations.loc[
        correlations["target"].isin(
            [
                "future_ew_realized_variance",
                "future_ew_realized_volatility",
            ]
        ),
        [
            "signal",
            "target",
            "n",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p",
        ],
    ]

    print(
        correlation_display
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 110
    )
    print(
        "PRIMARY REGRESSION: FUTURE EW REALIZED VARIANCE"
    )
    print(
        "=" * 110
    )

    primary_models = models.loc[
        models["target"]
        == "future_ew_realized_variance",
        [
            "model",
            "n",
            "r2",
            "adj_r2",
            "delta_r2_vs_standard",
            "delta_adj_r2_vs_standard",
        ],
    ]

    print(
        primary_models
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

    key_terms = coefficients.loc[
        (
            coefficients["target"]
            == "future_ew_realized_variance"
        )
        & (
            (
                (
                    coefficients["model"]
                    == "M1_standard_plus_Ares"
                )
                & (
                    coefficients["term"]
                    == "a_res_total_exposure"
                )
            )
            |
            (
                (
                    coefficients["model"]
                    == "M2_standard_plus_Ares_offdiag"
                )
                & (
                    coefficients["term"]
                    == "a_res_offdiag_exposure"
                )
            )
        ),
        [
            "model",
            "term",
            "standardized_beta",
            "hac_se",
            "hac_t",
            "hac_p",
        ],
    ]

    print(
        key_terms
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 110
    )
    print(
        "CURRENT RCR EXPOSURE QUARTILE -> FUTURE RISK"
    )
    print(
        "=" * 110
    )

    print(
        total_quartiles
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n" + "=" * 110
    )
    print(
        "EXPOSURE LEVELS"
    )
    print(
        "=" * 110
    )

    print(
        frame[
            [
                "sigma_exposure",
                "a_res_total_exposure",
                "a_res_diag_exposure",
                "a_res_offdiag_exposure",
                "effective_exposure",
                "a_res_to_sigma_ratio",
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
        / "rcr_matrix_exposure_timeseries.csv"
    )
    print(
        output_dir
        / "exposure_correlations.csv"
    )
    print(
        output_dir
        / "exposure_regression_models.csv"
    )
    print(
        output_dir
        / "exposure_regression_coefficients_hac.csv"
    )
    print(
        output_dir
        / "exposure_quartile_future_risk.csv"
    )


if __name__ == "__main__":
    main()