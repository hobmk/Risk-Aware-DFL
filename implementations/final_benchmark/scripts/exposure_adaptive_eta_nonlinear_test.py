from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from implementations.dfl_mvo_lee2025.src.optimization import (
    build_markowitz_layer,
    covariance_to_risk_factor,
)
from implementations.eigen_rcr_dfl.src.dataset import EigenRCRDataset
from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset


FINAL_ROOT = Path("implementations/final_benchmark/outputs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.2)
    parser.add_argument("--state-lookback", type=int, default=60)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    parser.add_argument(
        "--output-dir",
        default="implementations/final_benchmark/outputs/exposure_adaptive_eta_nonlinear",
    )
    return parser.parse_args()


def lambda_name(value):
    return f"lambda_{value:g}"


def load_dfl(seed, lam, tickers):
    path = (
        FINAL_ROOT
        / "dfl_mvo"
        / lambda_name(lam)
        / f"seed_{seed}"
        / "asset_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)

    required = {
        "date",
        "ticker",
        "predicted_return",
        "predicted_weight",
        "true_return",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")

    def pivot(column):
        result = (
            df.pivot(
                index="date",
                columns="ticker",
                values=column,
            )
            .sort_index()
            .reindex(columns=tickers)
        )

        if result.isna().any().any():
            raise RuntimeError(
                f"NaN after pivot: seed={seed}, column={column}"
            )

        return result

    return {
        "predictions": pivot("predicted_return"),
        "saved_weights": pivot("predicted_weight"),
        "returns": pivot("true_return"),
    }


def build_risk_data(args):
    rank_tail = RankTailRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        tail_q=args.tail_q,
        tail_side="down",
        dtype=torch.float64,
    )

    topk = EigenRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        market_mode="equal_weight",
        risk_free_rate=0.0,
        fit_intercept=True,
        eigen_k=1,
        scale_mode="none",
        dtype=torch.float64,
    )

    if rank_tail.tickers != topk.tickers:
        raise RuntimeError("Ticker mismatch.")

    rt_dates = pd.DatetimeIndex(rank_tail.target_dates).normalize()
    tk_dates = pd.DatetimeIndex(topk.target_dates).normalize()

    if not rt_dates.equals(tk_dates):
        raise RuntimeError("Date mismatch.")

    base_covariances = (
        rank_tail.base_covariances
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    rt_risk = (
        rank_tail.rank_tail_risk
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    topk_risk = (
        topk.eigen_risk
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    date_to_index = {
        pd.Timestamp(date): i
        for i, date in enumerate(rt_dates)
    }

    return (
        list(rank_tail.tickers),
        date_to_index,
        base_covariances,
        {
            "rank_tail": rt_risk,
            "topk": topk_risk,
        },
    )


def quadratic_exposure(weights, matrices):
    return np.einsum(
        "bi,bij,bj->b",
        weights,
        matrices,
        weights,
        optimize=True,
    )


def rolling_percentile(values, lookback):
    """
    미래정보 없이 현재 exposure의 최근 분포 percentile을 계산한다.

    첫 lookback일도 고정 0.5를 주지 않고,
    현재까지 존재하는 과거 데이터로 expanding 방식 사용.
    """
    values = np.asarray(values, dtype=np.float64)

    q = np.full(
        len(values),
        0.5,
        dtype=np.float64,
    )

    for t in range(1, len(values)):
        start = max(
            0,
            t - lookback,
        )

        history = values[
            start:t
        ]

        history = history[
            np.isfinite(history)
        ]

        if len(history) > 0 and np.isfinite(values[t]):
            q[t] = float(
                np.mean(
                    history <= values[t]
                )
            )

    return q


def build_eta_policies(
    percentile,
    eta_max,
    threshold,
):
    q = np.asarray(
        percentile,
        dtype=np.float64,
    )

    return {
        "linear_p1":
            eta_max * q,

        "power_p2":
            eta_max * q ** 2,

        "power_p3":
            eta_max * q ** 3,

        "threshold_80":
            eta_max
            * (
                q >= threshold
            ).astype(np.float64),
    }


def solve_weights(
    layer,
    predictions,
    covariances,
    risk_aversion,
):
    predictions_tensor = torch.tensor(
        predictions,
        dtype=torch.float64,
        device="cpu",
    )

    covariance_tensor = torch.tensor(
        covariances,
        dtype=torch.float64,
        device="cpu",
    )

    risk_factor = covariance_to_risk_factor(
        covariance=covariance_tensor,
        risk_aversion=risk_aversion,
        jitter=0.0,
    )

    with torch.no_grad():
        weights, = layer(
            predictions_tensor,
            risk_factor,
        )

    return (
        weights
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def portfolio_metrics(
    weights,
    returns,
    cost_bps=0.0,
):
    portfolio_returns = np.sum(
        weights * returns,
        axis=1,
    )

    turnover = np.zeros(
        len(weights),
        dtype=np.float64,
    )

    for t in range(
        1,
        len(weights),
    ):
        previous_weight = weights[t - 1]
        previous_return = returns[t - 1]

        denominator = (
            1.0
            + np.dot(
                previous_weight,
                previous_return,
            )
        )

        if abs(denominator) < 1e-12:
            drifted = previous_weight

        else:
            drifted = (
                previous_weight
                * (
                    1.0
                    + previous_return
                )
                / denominator
            )

        turnover[t] = (
            0.5
            * np.abs(
                weights[t]
                - drifted
            ).sum()
        )

    net_returns = (
        portfolio_returns
        - (
            cost_bps
            / 10000.0
        )
        * turnover
    )

    n = len(
        net_returns
    )

    wealth = np.cumprod(
        1.0
        + net_returns
    )

    annual_return = (
        wealth[-1]
        ** (
            252.0
            / n
        )
        - 1.0
        if wealth[-1] > 0
        else np.nan
    )

    volatility = (
        np.std(
            net_returns,
            ddof=1,
        )
        * np.sqrt(
            252.0
        )
    )

    daily_std = np.std(
        net_returns,
        ddof=1,
    )

    sharpe = (
        np.mean(
            net_returns
        )
        / daily_std
        * np.sqrt(
            252.0
        )
        if daily_std > 0
        else np.nan
    )

    running_max = (
        np.maximum.accumulate(
            wealth
        )
    )

    drawdown = (
        wealth
        / running_max
        - 1.0
    )

    mdd = float(
        -np.min(
            drawdown
        )
    )

    return {
        "annual_return":
            float(
                annual_return
            ),

        "volatility":
            float(
                volatility
            ),

        "sharpe":
            float(
                sharpe
            ),

        "mdd":
            mdd,

        "annualized_turnover":
            float(
                turnover.mean()
                * 252.0
            ),
    }


def paired_p(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan

    if np.allclose(
        values,
        0.0,
    ):
        return 1.0

    try:
        return float(
            wilcoxon(
                values,
                alternative="two-sided",
            ).pvalue
        )

    except ValueError:
        return np.nan


def run(args):
    (
        tickers,
        date_to_index,
        base_covariances,
        risk_matrices,
    ) = build_risk_data(
        args
    )

    n_assets = len(
        tickers
    )

    layer = build_markowitz_layer(
        n_assets=n_assets,
        max_weight=args.max_weight,
    )

    result_rows = []
    eta_rows = []

    for seed in args.seeds:
        dfl = load_dfl(
            seed,
            args.lambda_value,
            tickers,
        )

        dates = pd.DatetimeIndex(
            dfl[
                "predictions"
            ].index
        ).normalize()

        indices = np.asarray(
            [
                date_to_index[
                    date
                ]
                for date
                in dates
            ],
            dtype=np.int64,
        )

        predictions = (
            dfl[
                "predictions"
            ].to_numpy(
                dtype=np.float64
            )
        )

        saved_weights = (
            dfl[
                "saved_weights"
            ].to_numpy(
                dtype=np.float64
            )
        )

        returns = (
            dfl[
                "returns"
            ].to_numpy(
                dtype=np.float64
            )
        )

        base = (
            base_covariances[
                indices
            ]
        )

        baseline_weights = solve_weights(
            layer=layer,
            predictions=predictions,
            covariances=base,
            risk_aversion=args.lambda_value,
        )

        max_weight_diff = float(
            np.max(
                np.abs(
                    baseline_weights
                    - saved_weights
                )
            )
        )

        mean_weight_diff = float(
            np.mean(
                np.abs(
                    baseline_weights
                    - saved_weights
                )
            )
        )

        print(
            f"SEED {seed} | "
            f"DFL reproduction "
            f"max={max_weight_diff:.3e}, "
            f"mean={mean_weight_diff:.3e}"
        )

        baseline_metrics = {}

        for cost in [
            0.0,
            5.0,
            10.0,
        ]:
            baseline_metrics[
                cost
            ] = portfolio_metrics(
                baseline_weights,
                returns,
                cost_bps=cost,
            )

        for (
            model_name,
            all_risk,
        ) in risk_matrices.items():

            risk = all_risk[
                indices
            ]

            baseline_exposure = (
                quadratic_exposure(
                    baseline_weights,
                    risk,
                )
            )

            percentile = rolling_percentile(
                baseline_exposure,
                args.state_lookback,
            )

            eta_policies = (
                build_eta_policies(
                    percentile=percentile,
                    eta_max=args.eta_max,
                    threshold=args.threshold,
                )
            )

            result_rows.append({
                "seed":
                    seed,

                "model":
                    model_name,

                "policy":
                    "baseline",

                "control":
                    "baseline",

                "mean_eta":
                    0.0,

                **baseline_metrics[
                    0.0
                ],

                "sharpe_5bp":
                    baseline_metrics[
                        5.0
                    ]["sharpe"],

                "sharpe_10bp":
                    baseline_metrics[
                        10.0
                    ]["sharpe"],

                "mean_target_exposure":
                    float(
                        baseline_exposure.mean()
                    ),

                "exposure_reduction_pct":
                    0.0,
            })

            for (
                policy_name,
                dynamic_eta,
            ) in eta_policies.items():

                matched_eta = float(
                    dynamic_eta.mean()
                )

                dynamic_covariance = (
                    base
                    + dynamic_eta[
                        :,
                        None,
                        None,
                    ]
                    * risk
                )

                static_covariance = (
                    base
                    + matched_eta
                    * risk
                )

                dynamic_weights = solve_weights(
                    layer=layer,
                    predictions=predictions,
                    covariances=dynamic_covariance,
                    risk_aversion=args.lambda_value,
                )

                static_weights = solve_weights(
                    layer=layer,
                    predictions=predictions,
                    covariances=static_covariance,
                    risk_aversion=args.lambda_value,
                )

                dynamic_exposure = (
                    quadratic_exposure(
                        dynamic_weights,
                        risk,
                    )
                )

                static_exposure = (
                    quadratic_exposure(
                        static_weights,
                        risk,
                    )
                )

                for (
                    control,
                    weights,
                    exposure,
                ) in [
                    (
                        "dynamic",
                        dynamic_weights,
                        dynamic_exposure,
                    ),
                    (
                        "matched_static",
                        static_weights,
                        static_exposure,
                    ),
                ]:
                    gross = portfolio_metrics(
                        weights,
                        returns,
                        cost_bps=0.0,
                    )

                    net5 = portfolio_metrics(
                        weights,
                        returns,
                        cost_bps=5.0,
                    )

                    net10 = portfolio_metrics(
                        weights,
                        returns,
                        cost_bps=10.0,
                    )

                    reduction = (
                        (
                            1.0
                            - exposure.mean()
                            / baseline_exposure.mean()
                        )
                        * 100.0
                    )

                    result_rows.append({
                        "seed":
                            seed,

                        "model":
                            model_name,

                        "policy":
                            policy_name,

                        "control":
                            control,

                        "mean_eta":
                            matched_eta,

                        **gross,

                        "sharpe_5bp":
                            net5[
                                "sharpe"
                            ],

                        "sharpe_10bp":
                            net10[
                                "sharpe"
                            ],

                        "mean_target_exposure":
                            float(
                                exposure.mean()
                            ),

                        "exposure_reduction_pct":
                            float(
                                reduction
                            ),
                    })

                for t, date in enumerate(
                    dates
                ):
                    eta_rows.append({
                        "seed":
                            seed,

                        "model":
                            model_name,

                        "policy":
                            policy_name,

                        "date":
                            date,

                        "baseline_exposure":
                            baseline_exposure[t],

                        "percentile":
                            percentile[t],

                        "dynamic_eta":
                            dynamic_eta[t],

                        "matched_static_eta":
                            matched_eta,
                    })

                print(
                    f"  {model_name:10s} | "
                    f"{policy_name:12s} | "
                    f"mean_eta="
                    f"{matched_eta:.4f}"
                )

    return (
        pd.DataFrame(
            result_rows
        ),
        pd.DataFrame(
            eta_rows
        ),
    )


def improvement_diff(
    candidate,
    reference,
    metric,
):
    lower_better = {
        "volatility",
        "mdd",
        "annualized_turnover",
    }

    if metric in lower_better:
        return (
            reference[
                metric
            ].to_numpy()
            - candidate[
                metric
            ].to_numpy()
        )

    return (
        candidate[
            metric
        ].to_numpy()
        - reference[
            metric
        ].to_numpy()
    )


def build_dynamic_vs_static(
    results,
):
    rows = []

    metrics = [
        "annual_return",
        "volatility",
        "sharpe",
        "mdd",
        "annualized_turnover",
        "sharpe_5bp",
        "sharpe_10bp",
        "exposure_reduction_pct",
    ]

    policies = [
        "linear_p1",
        "power_p2",
        "power_p3",
        "threshold_80",
    ]

    for model in [
        "rank_tail",
        "topk",
    ]:
        for policy in policies:
            dynamic = results[
                (
                    results[
                        "model"
                    ] == model
                )
                & (
                    results[
                        "policy"
                    ] == policy
                )
                & (
                    results[
                        "control"
                    ] == "dynamic"
                )
            ].sort_values(
                "seed"
            )

            static = results[
                (
                    results[
                        "model"
                    ] == model
                )
                & (
                    results[
                        "policy"
                    ] == policy
                )
                & (
                    results[
                        "control"
                    ] == "matched_static"
                )
            ].sort_values(
                "seed"
            )

            for metric in metrics:
                diff = improvement_diff(
                    candidate=dynamic,
                    reference=static,
                    metric=metric,
                )

                rows.append({
                    "model":
                        model,

                    "policy":
                        policy,

                    "metric":
                        metric,

                    "mean_improvement":
                        float(
                            diff.mean()
                        ),

                    "median_improvement":
                        float(
                            np.median(
                                diff
                            )
                        ),

                    "candidate_better_seeds":
                        int(
                            (
                                diff > 0
                            ).sum()
                        ),

                    "reference_better_seeds":
                        int(
                            (
                                diff < 0
                            ).sum()
                        ),

                    "p_raw":
                        paired_p(
                            diff
                        ),
                })

    return pd.DataFrame(
        rows
    )


def build_vs_linear(
    results,
):
    rows = []

    metrics = [
        "annual_return",
        "volatility",
        "sharpe",
        "mdd",
        "annualized_turnover",
        "sharpe_5bp",
        "sharpe_10bp",
        "exposure_reduction_pct",
    ]

    candidates = [
        "power_p2",
        "power_p3",
        "threshold_80",
    ]

    for model in [
        "rank_tail",
        "topk",
    ]:
        linear = results[
            (
                results[
                    "model"
                ] == model
            )
            & (
                results[
                    "policy"
                ] == "linear_p1"
            )
            & (
                results[
                    "control"
                ] == "dynamic"
            )
        ].sort_values(
            "seed"
        )

        for policy in candidates:
            candidate = results[
                (
                    results[
                        "model"
                    ] == model
                )
                & (
                    results[
                        "policy"
                    ] == policy
                )
                & (
                    results[
                        "control"
                    ] == "dynamic"
                )
            ].sort_values(
                "seed"
            )

            for metric in metrics:
                diff = improvement_diff(
                    candidate=candidate,
                    reference=linear,
                    metric=metric,
                )

                rows.append({
                    "model":
                        model,

                    "candidate":
                        policy,

                    "reference":
                        "linear_p1",

                    "metric":
                        metric,

                    "mean_improvement":
                        float(
                            diff.mean()
                        ),

                    "median_improvement":
                        float(
                            np.median(
                                diff
                            )
                        ),

                    "candidate_better_seeds":
                        int(
                            (
                                diff > 0
                            ).sum()
                        ),

                    "linear_better_seeds":
                        int(
                            (
                                diff < 0
                            ).sum()
                        ),

                    "p_raw":
                        paired_p(
                            diff
                        ),
                })

    return pd.DataFrame(
        rows
    )


def build_summary(
    results,
):
    active = results[
        results[
            "control"
        ].isin(
            [
                "dynamic",
                "matched_static",
            ]
        )
    ]

    return (
        active.groupby(
            [
                "model",
                "policy",
                "control",
            ],
            as_index=False,
        )
        .agg(
            mean_eta=(
                "mean_eta",
                "mean",
            ),
            annual_return=(
                "annual_return",
                "mean",
            ),
            volatility=(
                "volatility",
                "mean",
            ),
            sharpe=(
                "sharpe",
                "mean",
            ),
            mdd=(
                "mdd",
                "mean",
            ),
            annualized_turnover=(
                "annualized_turnover",
                "mean",
            ),
            sharpe_5bp=(
                "sharpe_5bp",
                "mean",
            ),
            sharpe_10bp=(
                "sharpe_10bp",
                "mean",
            ),
            exposure_reduction_pct=(
                "exposure_reduction_pct",
                "mean",
            ),
        )
    )


def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 110
    )
    print(
        "NONLINEAR EXPOSURE-ADAPTIVE ETA TEST"
    )
    print(
        "=" * 110
    )

    print(
        f"lambda         : "
        f"{args.lambda_value:g}"
    )
    print(
        f"eta_max        : "
        f"{args.eta_max:g}"
    )
    print(
        f"state lookback : "
        f"{args.state_lookback}"
    )
    print(
        f"threshold      : "
        f"{args.threshold:g}"
    )

    results, eta_daily = run(
        args
    )

    summary = build_summary(
        results
    )

    dynamic_vs_static = (
        build_dynamic_vs_static(
            results
        )
    )

    vs_linear = build_vs_linear(
        results
    )

    results.to_csv(
        output_dir
        / "by_seed.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    dynamic_vs_static.to_csv(
        output_dir
        / "dynamic_vs_matched_static.csv",
        index=False,
    )

    vs_linear.to_csv(
        output_dir
        / "nonlinear_vs_linear.csv",
        index=False,
    )

    eta_daily.to_csv(
        output_dir
        / "eta_daily.csv",
        index=False,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "POLICY MEANS"
    )
    print(
        "=" * 110
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print(
        "=" * 110
    )
    print(
        "NONLINEAR VS LINEAR — KEY RESULTS"
    )
    print(
        "=" * 110
    )

    key = vs_linear[
        vs_linear[
            "metric"
        ].isin(
            [
                "annual_return",
                "volatility",
                "sharpe",
                "mdd",
                "sharpe_10bp",
                "exposure_reduction_pct",
            ]
        )
    ]

    print(
        key.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print(
        "=" * 110
    )
    print(
        "DYNAMIC VS MATCHED STATIC — SHARPE"
    )
    print(
        "=" * 110
    )

    key_static = dynamic_vs_static[
        dynamic_vs_static[
            "metric"
        ].isin(
            [
                "sharpe",
                "sharpe_10bp",
                "volatility",
                "exposure_reduction_pct",
            ]
        )
    ]

    print(
        key_static.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print(
        "Saved:"
    )
    print(
        output_dir
        / "by_seed.csv"
    )
    print(
        output_dir
        / "summary.csv"
    )
    print(
        output_dir
        / "dynamic_vs_matched_static.csv"
    )
    print(
        output_dir
        / "nonlinear_vs_linear.csv"
    )
    print(
        output_dir
        / "eta_daily.csv"
    )


if __name__ == "__main__":
    main()