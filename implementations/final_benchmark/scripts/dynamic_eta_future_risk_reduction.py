from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from implementations.dfl_mvo_lee2025.src.optimization import build_markowitz_layer
from implementations.rank_tail_rcr_dfl.scripts.validate_rank_tail_exposure import (
    future_fixed_weight_risk,
    hac_test,
)
from implementations.final_benchmark.scripts.exposure_adaptive_eta_nonlinear_test import (
    load_dfl,
    build_risk_data,
    quadratic_exposure,
    rolling_percentile,
    build_eta_policies,
    solve_weights,
)


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
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    parser.add_argument(
        "--output-dir",
        default="implementations/final_benchmark/outputs/dynamic_eta_future_risk",
    )
    return parser.parse_args()


def paired_p(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    if np.allclose(values, 0.0):
        return 1.0

    try:
        return float(wilcoxon(values).pvalue)
    except ValueError:
        return np.nan


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    result = np.full_like(pvalues, np.nan)

    valid = np.isfinite(pvalues)
    values = pvalues[valid]

    if len(values) == 0:
        return result

    order = np.argsort(values)
    ordered = values[order]

    adjusted = np.empty_like(ordered)
    running = 0.0
    m = len(ordered)

    for i, value in enumerate(ordered):
        current = min(
            1.0,
            (m - i) * value,
        )
        running = max(
            running,
            current,
        )
        adjusted[i] = running

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid] = restored

    return result


def analyze_pair(
    seed,
    model,
    policy,
    reference,
    horizon,
    reference_weights,
    candidate_weights,
    reference_exposure,
    candidate_exposure,
    returns,
):
    (
        ref_variance,
        ref_volatility,
        ref_mdd,
    ) = future_fixed_weight_risk(
        weights=reference_weights,
        returns=returns,
        horizon=horizon,
        periods_per_year=252,
    )

    (
        cand_variance,
        cand_volatility,
        cand_mdd,
    ) = future_fixed_weight_risk(
        weights=candidate_weights,
        returns=returns,
        horizon=horizon,
        periods_per_year=252,
    )

    exposure_reduction = (
        reference_exposure
        - candidate_exposure
    )

    metrics = {
        "future_variance":
            (
                ref_variance,
                cand_variance,
            ),

        "future_volatility":
            (
                ref_volatility,
                cand_volatility,
            ),

        "future_mdd":
            (
                ref_mdd,
                cand_mdd,
            ),
    }

    rows = []

    for metric, (
        reference_risk,
        candidate_risk,
    ) in metrics.items():

        future_reduction = (
            reference_risk
            - candidate_risk
        )

        valid = (
            np.isfinite(exposure_reduction)
            & np.isfinite(reference_risk)
            & np.isfinite(candidate_risk)
        )

        delta_e = exposure_reduction[valid]
        ref_risk = reference_risk[valid]
        cand_risk = candidate_risk[valid]
        delta_risk = future_reduction[valid]

        if len(delta_e) < 10:
            continue

        mean_ref = float(
            np.mean(ref_risk)
        )

        mean_candidate = float(
            np.mean(cand_risk)
        )

        mean_delta = float(
            np.mean(delta_risk)
        )

        overall_reduction_pct = (
            100.0
            * mean_delta
            / mean_ref
            if abs(mean_ref) > 1e-15
            else np.nan
        )

        reduced_mask = (
            delta_e > 0
        )

        n_reduced = int(
            reduced_mask.sum()
        )

        if n_reduced > 0:
            conditional_delta = float(
                np.mean(
                    delta_risk[
                        reduced_mask
                    ]
                )
            )

            conditional_ref = float(
                np.mean(
                    ref_risk[
                        reduced_mask
                    ]
                )
            )

            conditional_reduction_pct = (
                100.0
                * conditional_delta
                / conditional_ref
                if abs(
                    conditional_ref
                ) > 1e-15
                else np.nan
            )

            positive_future_days = float(
                np.mean(
                    delta_risk[
                        reduced_mask
                    ] > 0
                )
                * 100.0
            )

        else:
            conditional_delta = np.nan
            conditional_reduction_pct = np.nan
            positive_future_days = np.nan

        rho = spearmanr(
            delta_e,
            delta_risk,
        )

        dataframe = pd.DataFrame({
            "exposure_reduction":
                delta_e,

            "future_risk_reduction":
                delta_risk,
        })

        hac = hac_test(
            dataframe=dataframe,
            exposure_column="exposure_reduction",
            future_column="future_risk_reduction",
            hac_lags=horizon - 1,
        )

        exposure_reference_mean = float(
            np.mean(
                reference_exposure[
                    valid
                ]
            )
        )

        exposure_candidate_mean = float(
            np.mean(
                candidate_exposure[
                    valid
                ]
            )
        )

        exposure_reduction_pct = (
            100.0
            * (
                exposure_reference_mean
                - exposure_candidate_mean
            )
            / exposure_reference_mean
            if abs(
                exposure_reference_mean
            ) > 1e-15
            else np.nan
        )

        rows.append({
            "seed":
                seed,

            "model":
                model,

            "policy":
                policy,

            "reference":
                reference,

            "horizon":
                horizon,

            "future_metric":
                metric,

            "n":
                len(delta_e),

            "exposure_reduction_pct":
                exposure_reduction_pct,

            "exposure_reduced_days_pct":
                float(
                    np.mean(
                        reduced_mask
                    )
                    * 100.0
                ),

            "reference_future_risk":
                mean_ref,

            "candidate_future_risk":
                mean_candidate,

            "future_risk_reduction_abs":
                mean_delta,

            "future_risk_reduction_pct":
                overall_reduction_pct,

            "conditional_future_reduction_abs":
                conditional_delta,

            "conditional_future_reduction_pct":
                conditional_reduction_pct,

            "positive_future_reduction_days_pct":
                positive_future_days,

            "spearman":
                float(
                    rho.statistic
                ),

            "spearman_p":
                float(
                    rho.pvalue
                ),

            "hac_beta_1sd":
                float(
                    hac[
                        "hac_beta_per_1sd_exposure"
                    ]
                ),

            "hac_p_1sd":
                float(
                    hac[
                        "hac_p_per_1sd_exposure"
                    ]
                ),
        })

    return rows


def run(args):
    (
        tickers,
        date_to_index,
        base_covariances,
        risk_matrices,
    ) = build_risk_data(
        args
    )

    layer = build_markowitz_layer(
        n_assets=len(tickers),
        max_weight=args.max_weight,
    )

    rows = []

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
                for date in dates
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

        print(
            f"SEED {seed}"
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

                for horizon in args.horizons:
                    rows.extend(
                        analyze_pair(
                            seed=seed,
                            model=model_name,
                            policy=policy_name,
                            reference="baseline",
                            horizon=horizon,
                            reference_weights=baseline_weights,
                            candidate_weights=dynamic_weights,
                            reference_exposure=baseline_exposure,
                            candidate_exposure=dynamic_exposure,
                            returns=returns,
                        )
                    )

                    rows.extend(
                        analyze_pair(
                            seed=seed,
                            model=model_name,
                            policy=policy_name,
                            reference="matched_static",
                            horizon=horizon,
                            reference_weights=static_weights,
                            candidate_weights=dynamic_weights,
                            reference_exposure=static_exposure,
                            candidate_exposure=dynamic_exposure,
                            returns=returns,
                        )
                    )

            print(
                f"  DONE | {model_name}"
            )

    return pd.DataFrame(
        rows
    )


def build_summary(by_seed):
    rows = []

    group_columns = [
        "model",
        "policy",
        "reference",
        "horizon",
        "future_metric",
    ]

    for keys, group in by_seed.groupby(
        group_columns
    ):
        (
            model,
            policy,
            reference,
            horizon,
            metric,
        ) = keys

        values = (
            group[
                "future_risk_reduction_pct"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        conditional = (
            group[
                "conditional_future_reduction_pct"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        rho = (
            group[
                "spearman"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        beta = (
            group[
                "hac_beta_1sd"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        exposure = (
            group[
                "exposure_reduction_pct"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        rows.append({
            "model":
                model,

            "policy":
                policy,

            "reference":
                reference,

            "horizon":
                horizon,

            "future_metric":
                metric,

            "mean_exposure_reduction_pct":
                float(
                    np.nanmean(
                        exposure
                    )
                ),

            "mean_future_risk_reduction_pct":
                float(
                    np.nanmean(
                        values
                    )
                ),

            "median_future_risk_reduction_pct":
                float(
                    np.nanmedian(
                        values
                    )
                ),

            "positive_future_risk_seeds":
                int(
                    np.nansum(
                        values > 0
                    )
                ),

            "future_risk_p_raw":
                paired_p(
                    values
                ),

            "mean_conditional_reduction_pct":
                float(
                    np.nanmean(
                        conditional
                    )
                ),

            "positive_conditional_seeds":
                int(
                    np.nansum(
                        conditional > 0
                    )
                ),

            "conditional_p_raw":
                paired_p(
                    conditional
                ),

            "mean_spearman":
                float(
                    np.nanmean(
                        rho
                    )
                ),

            "positive_spearman_seeds":
                int(
                    np.nansum(
                        rho > 0
                    )
                ),

            "spearman_p_raw":
                paired_p(
                    rho
                ),

            "mean_hac_beta_1sd":
                float(
                    np.nanmean(
                        beta
                    )
                ),

            "positive_hac_beta_seeds":
                int(
                    np.nansum(
                        beta > 0
                    )
                ),
        })

    summary = pd.DataFrame(
        rows
    )

    for column in [
        "future_risk_p_raw",
        "conditional_p_raw",
        "spearman_p_raw",
    ]:
        adjusted_column = (
            column.replace(
                "_raw",
                "_holm",
            )
        )

        summary[
            adjusted_column
        ] = np.nan

        for _, group in summary.groupby(
            [
                "model",
                "reference",
                "horizon",
            ]
        ):
            idx = group.index

            summary.loc[
                idx,
                adjusted_column,
            ] = holm_adjust(
                group[
                    column
                ].to_numpy(
                    dtype=np.float64
                )
            )

    return summary


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
        "DYNAMIC ETA -> FUTURE RISK REDUCTION TEST"
    )
    print(
        "=" * 110
    )
    print(
        f"lambda     : "
        f"{args.lambda_value:g}"
    )
    print(
        f"eta_max    : "
        f"{args.eta_max:g}"
    )
    print(
        f"horizons   : "
        f"{args.horizons}"
    )

    by_seed = run(
        args
    )

    summary = build_summary(
        by_seed
    )

    by_seed.to_csv(
        output_dir
        / "by_seed.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "20-DAY FUTURE VOLATILITY — ECONOMIC MAGNITUDE"
    )
    print(
        "=" * 110
    )

    key = summary[
        (
            summary[
                "horizon"
            ] == 20
        )
        & (
            summary[
                "future_metric"
            ]
            == "future_volatility"
        )
    ]

    columns = [
        "model",
        "policy",
        "reference",
        "mean_exposure_reduction_pct",
        "mean_future_risk_reduction_pct",
        "positive_future_risk_seeds",
        "future_risk_p_holm",
        "mean_conditional_reduction_pct",
        "positive_conditional_seeds",
        "mean_spearman",
        "positive_spearman_seeds",
        "spearman_p_holm",
        "mean_hac_beta_1sd",
    ]

    print(
        key[
            columns
        ].to_string(
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
        "20-DAY FUTURE VARIANCE — ECONOMIC MAGNITUDE"
    )
    print(
        "=" * 110
    )

    key_variance = summary[
        (
            summary[
                "horizon"
            ] == 20
        )
        & (
            summary[
                "future_metric"
            ]
            == "future_variance"
        )
    ]

    print(
        key_variance[
            columns
        ].to_string(
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


if __name__ == "__main__":
    main()