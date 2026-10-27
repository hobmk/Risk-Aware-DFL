from __future__ import annotations

import argparse
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import torch

from implementations.dfl_mvo_lee2025.src.optimization import covariance_to_risk_factor
from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset


ROOT = Path("implementations/final_benchmark/outputs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--tail-q", type=float, default=0.35)
    parser.add_argument("--cvar-alpha", type=float, default=0.90)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.2)
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.01, 0.025, 0.05, 0.1, 0.25])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--output-dir",
        default="implementations/final_benchmark/outputs/rcr_collective_cvar_test",
    )
    return parser.parse_args()


def load_dfl(seed, lam, tickers):
    path = ROOT / "dfl_mvo" / f"lambda_{lam:g}" / f"seed_{seed}" / "asset_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    def pivot(column):
        out = df.pivot(index="date", columns="ticker", values=column).sort_index()
        return out.reindex(columns=tickers)

    return {
        "predictions": pivot("predicted_return"),
        "weights": pivot("predicted_weight"),
        "returns": pivot("true_return"),
    }


def rank_tail_scores(residuals, tail_q):
    """
    residuals: [B, T, N]

    각 rolling window 안에서 자산별 residual percentile 계산.
    작은 residual일수록 큰 downside tail score.
    """
    order = np.argsort(residuals, axis=1)
    ranks = np.empty_like(order, dtype=np.float64)

    ordinal = np.arange(
        1,
        residuals.shape[1] + 1,
        dtype=np.float64,
    )[None, :, None]

    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(ordinal, residuals.shape),
        axis=1,
    )

    u = ranks / float(residuals.shape[1] + 1)

    return np.maximum(
        (tail_q - u) / tail_q,
        0.0,
    )


def normalize_scenario_weights(score):
    denominator = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        denominator,
        out=np.full_like(
            score,
            1.0 / score.shape[1],
        ),
        where=denominator > 1e-15,
    )


def rank_tail_scenario_weights(tail_scores):
    """
    기존 RCR weighting.

    c_s = 날짜별 평균 tail severity
    """
    severity = tail_scores.mean(axis=2)

    return normalize_scenario_weights(
        severity
    )


def collective_rcr_scenario_weights(tail_scores):
    """
    Collective RCR weighting.

    breadth  = 동시에 downside tail에 들어간 자산 비율
    severity = 평균 downside severity

    c_s = breadth * severity
    """
    breadth = (
        tail_scores > 0.0
    ).mean(axis=2)

    severity = tail_scores.mean(
        axis=2
    )

    collective = (
        breadth
        * severity
    )

    return normalize_scenario_weights(
        collective
    )


def build_data(args):
    dataset = RankTailRCRDataset(
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

    asset_windows = np.lib.stride_tricks.sliding_window_view(
        dataset.returns,
        window_shape=args.lookback,
        axis=0,
    ).transpose(0, 2, 1)[:-1]

    market_windows = np.lib.stride_tricks.sliding_window_view(
        dataset.market_returns,
        window_shape=args.lookback,
    )[:-1]

    alpha = dataset.capm_alpha.detach().cpu().numpy()
    beta = dataset.capm_beta.detach().cpu().numpy()

    residuals = (
        asset_windows
        - alpha[:, None, :]
        - market_windows[:, :, None] * beta[:, None, :]
    )

    tail_scores = rank_tail_scores(
        residuals,
        args.tail_q,
    )

    rcr_probabilities = rank_tail_scenario_weights(
        tail_scores
    )

    collective_probabilities = collective_rcr_scenario_weights(
        tail_scores
    )

    dates = pd.DatetimeIndex(
        dataset.target_dates
    ).normalize()

    date_to_index = {
        pd.Timestamp(date): i
        for i, date in enumerate(dates)
    }

    base_covariance = (
        dataset.base_covariances
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    return (
        dataset.tickers,
        date_to_index,
        base_covariance,
        residuals.astype(np.float64),
        rcr_probabilities.astype(np.float64),
        collective_probabilities.astype(np.float64),
    )


class CVaRPortfolioSolver:
    def __init__(
        self,
        n_assets,
        n_scenarios,
        gamma,
        alpha,
        max_weight,
    ):
        self.w = cp.Variable(n_assets)
        self.tau = cp.Variable()
        self.u = cp.Variable(n_scenarios)

        self.mu = cp.Parameter(n_assets)
        self.risk_factor = cp.Parameter((n_assets, n_assets))
        self.scenarios = cp.Parameter((n_scenarios, n_assets))
        self.probabilities = cp.Parameter(n_scenarios, nonneg=True)

        losses = (
            -self.scenarios
            @ self.w
        )

        cvar = (
            self.tau
            + self.probabilities @ self.u
            / (1.0 - alpha)
        )

        objective = cp.Minimize(
            cp.sum_squares(
                self.risk_factor
                @ self.w
            )
            - self.mu @ self.w
            + gamma * cvar
        )

        constraints = [
            cp.sum(self.w) == 1,
            self.w >= 0,
            self.w <= max_weight,
            self.u >= losses - self.tau,
            self.u >= 0,
        ]

        self.problem = cp.Problem(
            objective,
            constraints,
        )

    def solve(
        self,
        mu,
        risk_factor,
        scenarios,
        probabilities,
    ):
        self.mu.value = mu
        self.risk_factor.value = risk_factor
        self.scenarios.value = scenarios
        self.probabilities.value = probabilities

        installed = cp.installed_solvers()

        try:
            if "CLARABEL" in installed:
                self.problem.solve(
                    solver=cp.CLARABEL,
                    warm_start=True,
                    verbose=False,
                )
            elif "OSQP" in installed:
                self.problem.solve(
                    solver=cp.OSQP,
                    warm_start=True,
                    verbose=False,
                )
            else:
                self.problem.solve(
                    solver=cp.SCS,
                    warm_start=True,
                    verbose=False,
                )

        except cp.SolverError:
            self.problem.solve(
                solver=cp.SCS,
                warm_start=True,
                verbose=False,
            )

        if self.w.value is None:
            raise RuntimeError(
                f"CVaR optimization failed: {self.problem.status}"
            )

        return np.asarray(
            self.w.value,
            dtype=np.float64,
        )


def solve_candidate(
    predictions,
    base_covariance,
    residuals,
    probabilities,
    gamma,
    args,
):
    n_dates, n_assets = predictions.shape
    n_scenarios = residuals.shape[1]

    risk_factor = (
        covariance_to_risk_factor(
            torch.tensor(
                base_covariance,
                dtype=torch.float64,
            ),
            risk_aversion=args.lambda_value,
            jitter=0.0,
        )
        .detach()
        .cpu()
        .numpy()
    )

    solver = CVaRPortfolioSolver(
        n_assets=n_assets,
        n_scenarios=n_scenarios,
        gamma=gamma,
        alpha=args.cvar_alpha,
        max_weight=args.max_weight,
    )

    weights = np.empty(
        (n_dates, n_assets),
        dtype=np.float64,
    )

    for t in range(n_dates):
        weights[t] = solver.solve(
            mu=predictions[t],
            risk_factor=risk_factor[t],
            scenarios=residuals[t],
            probabilities=probabilities[t],
        )

    return weights


def max_drawdown(returns):
    wealth = np.cumprod(
        1.0 + returns
    )

    peak = np.maximum.accumulate(
        wealth
    )

    return float(
        np.max(
            1.0
            - wealth / peak
        )
    )


def empirical_cvar(returns, alpha):
    losses = -np.asarray(
        returns,
        dtype=np.float64,
    )

    k = max(
        1,
        int(
            np.ceil(
                (1.0 - alpha)
                * len(losses)
            )
        ),
    )

    return float(
        np.mean(
            np.sort(losses)[-k:]
        )
    )


def portfolio_metrics(
    weights,
    returns,
    alpha,
):
    portfolio_returns = np.sum(
        weights * returns,
        axis=1,
    )

    wealth = np.cumprod(
        1.0 + portfolio_returns
    )

    annual_return = (
        wealth[-1]
        ** (
            252.0
            / len(portfolio_returns)
        )
        - 1.0
    )

    volatility = (
        np.std(
            portfolio_returns,
            ddof=1,
        )
        * np.sqrt(252.0)
    )

    sharpe = (
        np.mean(
            portfolio_returns
        )
        / np.std(
            portfolio_returns,
            ddof=1,
        )
        * np.sqrt(252.0)
    )

    return {
        "annual_return":
            float(annual_return),

        "volatility":
            float(volatility),

        "sharpe":
            float(sharpe),

        "mdd":
            max_drawdown(
                portfolio_returns
            ),

        "cvar":
            empirical_cvar(
                portfolio_returns,
                alpha,
            ),
    }


def future_risk(
    weights,
    returns,
    horizon,
    alpha,
):
    n_dates = len(returns)

    volatility = np.full(
        n_dates,
        np.nan,
    )

    mdd = np.full(
        n_dates,
        np.nan,
    )

    cvar = np.full(
        n_dates,
        np.nan,
    )

    for t in range(
        n_dates - horizon + 1
    ):
        future_returns = (
            returns[
                t:t + horizon
            ]
            @ weights[t]
        )

        volatility[t] = (
            np.std(
                future_returns,
                ddof=1,
            )
            * np.sqrt(252.0)
        )

        mdd[t] = max_drawdown(
            future_returns
        )

        cvar[t] = empirical_cvar(
            future_returns,
            alpha,
        )

    return {
        "volatility": volatility,
        "mdd": mdd,
        "cvar": cvar,
    }


def risk_reduction(
    baseline,
    candidate,
):
    valid = (
        np.isfinite(baseline)
        & np.isfinite(candidate)
    )

    baseline = baseline[
        valid
    ]

    candidate = candidate[
        valid
    ]

    reference_mean = float(
        baseline.mean()
    )

    candidate_mean = float(
        candidate.mean()
    )

    if abs(reference_mean) <= 1e-15:
        return np.nan

    return (
        100.0
        * (
            reference_mean
            - candidate_mean
        )
        / reference_mean
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

    (
        tickers,
        date_to_index,
        base_covariance,
        residuals,
        rcr_probabilities,
        collective_probabilities,
    ) = build_data(
        args
    )

    uniform_probabilities = np.full(
        args.lookback,
        1.0 / args.lookback,
        dtype=np.float64,
    )

    rows = []

    print("=" * 120)
    print("RESIDUAL CVaR: UNIFORM vs RCR vs COLLECTIVE RCR")
    print("=" * 120)
    print(f"lambda      : {args.lambda_value:g}")
    print(f"tail_q      : {args.tail_q:g}")
    print(f"CVaR alpha  : {args.cvar_alpha:g}")
    print(f"gammas      : {args.gammas}")
    print(f"seeds       : {args.seeds}")

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
                date_to_index[date]
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

        baseline_weights = (
            dfl[
                "weights"
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

        cov = base_covariance[
            indices
        ]

        scenarios = residuals[
            indices
        ]

        rcr_p = rcr_probabilities[
            indices
        ]

        collective_p = collective_probabilities[
            indices
        ]

        uniform_p = np.broadcast_to(
            uniform_probabilities,
            (
                len(dates),
                args.lookback,
            ),
        )

        baseline_metrics = portfolio_metrics(
            baseline_weights,
            returns,
            args.cvar_alpha,
        )

        baseline_future = {
            horizon: future_risk(
                baseline_weights,
                returns,
                horizon,
                args.cvar_alpha,
            )
            for horizon in [5, 20]
        }

        rows.append({
            "seed": seed,
            "method": "dfl_mvo",
            "gamma": 0.0,
            **baseline_metrics,
            "future_vol_reduction_h5": 0.0,
            "future_mdd_reduction_h5": 0.0,
            "future_cvar_reduction_h5": 0.0,
            "future_vol_reduction_h20": 0.0,
            "future_mdd_reduction_h20": 0.0,
            "future_cvar_reduction_h20": 0.0,
        })

        method_probabilities = {
            "uniform_residual_cvar":
                uniform_p,

            "rcr_weighted_cvar":
                rcr_p,

            "collective_rcr_cvar":
                collective_p,
        }

        for gamma in args.gammas:
            for (
                method,
                probabilities,
            ) in method_probabilities.items():

                weights = solve_candidate(
                    predictions=predictions,
                    base_covariance=cov,
                    residuals=scenarios,
                    probabilities=probabilities,
                    gamma=gamma,
                    args=args,
                )

                metrics = portfolio_metrics(
                    weights,
                    returns,
                    args.cvar_alpha,
                )

                future = {
                    horizon: future_risk(
                        weights,
                        returns,
                        horizon,
                        args.cvar_alpha,
                    )
                    for horizon in [5, 20]
                }

                row = {
                    "seed": seed,
                    "method": method,
                    "gamma": gamma,
                    **metrics,
                }

                for horizon in [5, 20]:
                    row[
                        f"future_vol_reduction_h{horizon}"
                    ] = risk_reduction(
                        baseline_future[
                            horizon
                        ]["volatility"],
                        future[
                            horizon
                        ]["volatility"],
                    )

                    row[
                        f"future_mdd_reduction_h{horizon}"
                    ] = risk_reduction(
                        baseline_future[
                            horizon
                        ]["mdd"],
                        future[
                            horizon
                        ]["mdd"],
                    )

                    row[
                        f"future_cvar_reduction_h{horizon}"
                    ] = risk_reduction(
                        baseline_future[
                            horizon
                        ]["cvar"],
                        future[
                            horizon
                        ]["cvar"],
                    )

                rows.append(
                    row
                )

                print(
                    f"seed={seed} | "
                    f"{method:23s} | "
                    f"gamma={gamma:.3f} | "
                    f"Sharpe={metrics['sharpe']:.4f} | "
                    f"Vol20↓={row['future_vol_reduction_h20']:.3f}% | "
                    f"CVaR20↓={row['future_cvar_reduction_h20']:.3f}% | "
                    f"MDD20↓={row['future_mdd_reduction_h20']:.3f}%"
                )

    by_seed = pd.DataFrame(
        rows
    )

    summary = (
        by_seed.groupby(
            [
                "method",
                "gamma",
            ],
            as_index=False,
        )
        .agg(
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
            cvar=(
                "cvar",
                "mean",
            ),
            future_vol_reduction_h5=(
                "future_vol_reduction_h5",
                "mean",
            ),
            future_mdd_reduction_h5=(
                "future_mdd_reduction_h5",
                "mean",
            ),
            future_cvar_reduction_h5=(
                "future_cvar_reduction_h5",
                "mean",
            ),
            future_vol_reduction_h20=(
                "future_vol_reduction_h20",
                "mean",
            ),
            future_mdd_reduction_h20=(
                "future_mdd_reduction_h20",
                "mean",
            ),
            future_cvar_reduction_h20=(
                "future_cvar_reduction_h20",
                "mean",
            ),
        )
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
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print("=" * 120)
    print("COLLECTIVE RCR vs UNIFORM — SAME GAMMA")
    print("=" * 120)

    uniform = (
        summary[
            summary[
                "method"
            ]
            == "uniform_residual_cvar"
        ]
        .set_index("gamma")
    )

    collective = (
        summary[
            summary[
                "method"
            ]
            == "collective_rcr_cvar"
        ]
        .set_index("gamma")
    )

    comparison_rows = []

    for gamma in args.gammas:
        if (
            gamma not in uniform.index
            or gamma not in collective.index
        ):
            continue

        u = uniform.loc[
            gamma
        ]

        c = collective.loc[
            gamma
        ]

        comparison_rows.append({
            "gamma":
                gamma,

            "sharpe_diff":
                c["sharpe"]
                - u["sharpe"],

            "annual_return_diff":
                c["annual_return"]
                - u["annual_return"],

            "volatility_improvement":
                u["volatility"]
                - c["volatility"],

            "mdd_improvement":
                u["mdd"]
                - c["mdd"],

            "cvar_improvement":
                u["cvar"]
                - c["cvar"],

            "future_vol20_extra_reduction":
                c[
                    "future_vol_reduction_h20"
                ]
                - u[
                    "future_vol_reduction_h20"
                ],

            "future_mdd20_extra_reduction":
                c[
                    "future_mdd_reduction_h20"
                ]
                - u[
                    "future_mdd_reduction_h20"
                ],

            "future_cvar20_extra_reduction":
                c[
                    "future_cvar_reduction_h20"
                ]
                - u[
                    "future_cvar_reduction_h20"
                ],
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        output_dir
        / "collective_vs_uniform.csv",
        index=False,
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print("Saved:")
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
        / "collective_vs_uniform.csv"
    )


if __name__ == "__main__":
    main()