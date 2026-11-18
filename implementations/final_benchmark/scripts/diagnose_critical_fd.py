
"""Validate CVXPYLayers gradients with KKT and CLARABEL finite differences.

Uses outputs saved by diagnose_cross_kkt.py. No retraining is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from implementations.dfl_mvo_lee2025.src.dataset import RollingMVODataset
from implementations.covariance_baselines.src.dataset import CovarianceBaselineDataset
from implementations.dfl_mvo_lee2025.src.optimization import (
    build_markowitz_layer, covariance_to_risk_factor,
)
from implementations.dfl_mvo_lee2025.src.losses import markowitz_cost
from implementations.final_benchmark.scripts.diagnose_cross_kkt import solve_independent_qp


def parse_args():
    p = argparse.ArgumentParser(description="Critical-sample KKT and finite-difference test")
    p.add_argument("--risk", choices=["ridge", "sample"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--batch-index", type=int, required=True)
    p.add_argument("--cross-root", default="implementations/final_benchmark/outputs/diagnostics/cross_kkt")
    p.add_argument("--label", default="fd_v1")
    p.add_argument("--bound-tol", type=float, default=1e-6)
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--tight-eps", type=float, default=1e-8)
    p.add_argument("--tight-max-iters", type=int, default=10000)
    return p.parse_args()


def active_signature(weights, tol, cap):
    lower = tuple(np.flatnonzero(weights <= tol).tolist())
    upper = tuple(np.flatnonzero(weights >= cap - tol).tolist())
    return lower, upper


def kkt_implicit_gradient(mu, y, factor, answer, tol, cap):
    """Gradient through the local fixed-active-set KKT system."""
    w = answer["weights"]
    n = len(w)
    free = np.flatnonzero((w > tol) & (w < cap - tol))

    gradient = np.zeros(n, dtype=np.float64)
    condition = np.nan

    if len(free) >= 2:
        q = factor.T @ factor
        m = len(free)

        system = np.zeros((m + 1, m + 1), dtype=np.float64)
        system[:m, :m] = 2 * q[np.ix_(free, free)]
        system[:m, m] = 1.0
        system[m, :m] = 1.0

        rhs = np.zeros((m + 1, m), dtype=np.float64)
        rhs[:m, :] = np.eye(m)

        derivative = np.linalg.solve(system, rhs)[:m, :]
        cost_gradient_w = 2 * (q @ w) - y

        gradient[free] = derivative.T @ cost_gradient_w[free]
        condition = float(np.linalg.cond(system))

    # For 0 or 1 free asset, fixed bounds and sum(w)=1 locally fix w.
    return gradient, free, condition


def true_cost(weights, y, factor):
    rw = factor @ weights
    return float(rw @ rw - y @ weights)


def cvxpy_single_gradient(layer, mu, y, factor, solver_args):
    # Match the original Predictor's float32 outputs.
    prediction = torch.tensor(mu, dtype=torch.float32, requires_grad=True)
    risk_factor = torch.tensor(factor, dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.float64)

    (weights,) = layer(
        prediction.to(torch.float64),
        risk_factor,
        solver_args=solver_args,
    )

    cost = markowitz_cost(weights, target, risk_factor)
    (gradient,) = torch.autograd.grad(cost, prediction)

    return (
        gradient.detach().cpu().numpy().astype(np.float64),
        weights.detach().cpu().numpy(),
    )


def independent_solution(mu, factor, cap):
    return solve_independent_qp(
        np.asarray(mu, dtype=np.float64),
        factor,
        max_weight=cap,
    )


def main():
    args = parse_args()

    if args.max_weight != 0.2:
        raise ValueError("This study uses max_weight=0.2")
    if args.batch_index < 0 or args.bound_tol <= 0:
        raise ValueError("Invalid batch-index or bound-tol")

    torch.set_num_threads(1)

    cross = (
        Path(args.cross_root)
        / args.risk
        / f"seed_{args.seed}"
        / f"step_{args.step}"
        / "first"
    )

    if not cross.is_dir():
        raise FileNotFoundError(f"Missing cross_kkt folder: {cross}")

    out = cross / args.label

    if out.exists():
        raise FileExistsError(
            f"Output exists: {out}. Use another --label."
        )

    config = json.loads(
        (cross / "config.json").read_text(encoding="utf-8")
    )

    previous = pd.read_csv(
        cross / "independent_kkt_by_sample.csv"
    )

    records = previous[
        (previous["key"] == "P_base__S_base")
        & (previous["batch_index"] == args.batch_index)
    ]

    if len(records) != 1:
        raise ValueError(f"Unexpected batch-index: {args.batch_index}")

    record = records.iloc[0]
    sample_index = int(record["sample_index"])
    selected_date = str(record["date"])

    common = dict(
        price_csv=config["price_csv"],
        lookback=int(config["lookback"]),
        covariance_jitter=1e-6,
        dtype=torch.float64,
    )

    sample = RollingMVODataset(
        **common,
        risk_cache_dir="implementations/dfl_mvo_lee2025/cache/risk_matrix",
    )

    if selected_date != str(sample.target_dates[sample_index].date()):
        raise RuntimeError("Saved sample date does not match dataset")

    sample_cov = sample.covariances[
        sample_index
    ].detach().cpu().numpy()

    base_cov = sample_cov

    if args.risk == "ridge":
        ridge = CovarianceBaselineDataset(
            **common,
            method="ridge",
            ridge_shrinkage=float(config["rho"]),
        )
        base_cov = ridge.covariances[
            sample_index
        ].detach().cpu().numpy()

    rng = np.random.default_rng(
        int(config["direction_seed"])
    )

    v = rng.normal(size=sample.n_assets)
    v /= np.linalg.norm(v)

    perturbation = (
        float(config["perturbation"])
        * np.linalg.norm(sample_cov)
        * np.outer(v, v)
    )

    factors = {
        "base": covariance_to_risk_factor(
            torch.tensor(base_cov, dtype=torch.float64),
            float(config["risk_aversion"]),
        ).numpy(),
        "perturbed": covariance_to_risk_factor(
            torch.tensor(base_cov + perturbation, dtype=torch.float64),
            float(config["risk_aversion"]),
        ).numpy(),
    }

    y = (
        sample[sample_index]["target"]
        .float()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    stored = np.load(
        cross / "cross_arrays.npz"
    )

    batch_size = int(
        (previous["key"] == "P_base__S_base").sum()
    )

    mu_base = stored[
        "P_base__S_base__mu"
    ][args.batch_index].astype(np.float64)

    mu_pert = stored[
        "P_perturbed__S_base__mu"
    ][args.batch_index].astype(np.float64)

    delta_mu = mu_pert - mu_base
    delta_norm = float(np.linalg.norm(delta_mu))

    if delta_norm == 0:
        raise RuntimeError("Predictor outputs are identical")

    direction_predictor = delta_mu / delta_norm

    base_answer = independent_solution(
        mu_base, factors["base"], args.max_weight,
    )

    _, free_base, _ = kkt_implicit_gradient(
        mu_base,
        y,
        factors["base"],
        base_answer,
        args.bound_tol,
        args.max_weight,
    )

    if len(free_base) >= 2:
        i, j = int(free_base[0]), int(free_base[1])
    else:
        i, j = np.argsort(mu_base)[-2:].tolist()

    direction_free = (
        np.eye(sample.n_assets)[i]
        - np.eye(sample.n_assets)[j]
    )
    direction_free /= np.linalg.norm(direction_free)

    directions = {
        "predictor_delta": direction_predictor,
        "two_asset_free_axis": direction_free,
    }

    layer = build_markowitz_layer(
        sample.n_assets,
        max_weight=args.max_weight,
    )

    solver_args = {
        "eps": args.tight_eps,
        "max_iters": args.tight_max_iters,
        "acceleration_lookback": 0,
    }

    labels = [
        "P_base__S_base",
        "P_base__S_perturbed",
        "P_perturbed__S_base",
        "P_perturbed__S_perturbed",
    ]

    rows = []
    fd_rows = []

    for key in labels:
        predictor_name = (
            "base"
            if key.startswith("P_base__")
            else "perturbed"
        )
        cov_name = (
            "base"
            if key.endswith("__S_base")
            else "perturbed"
        )

        mu = mu_base if predictor_name == "base" else mu_pert
        factor = factors[cov_name]

        answer = independent_solution(
            mu, factor, args.max_weight,
        )

        weights = answer["weights"]

        gradient_kkt, free, kkt_condition = kkt_implicit_gradient(
            mu, y, factor, answer,
            args.bound_tol, args.max_weight,
        )

        gradient_cvx, weights_cvx = cvxpy_single_gradient(
            layer, mu, y, factor, solver_args,
        )

        # Previously saved gradient is from mean(batch costs).
        # Convert it to the single-sample gradient before comparison.
        gradient_saved = (
            stored[f"{key}__mu_gradient_cost"][args.batch_index]
            .astype(np.float64)
            * batch_size
        )

        signature = active_signature(
            weights, args.bound_tol, args.max_weight,
        )

        lower_idx, upper_idx = signature

        active_duals = np.concatenate([
            answer["lower_dual"][list(lower_idx)],
            answer["upper_dual"][list(upper_idx)],
        ])

        rows.append({
            "key": key,
            "batch_index": args.batch_index,
            "sample_index": sample_index,
            "date": selected_date,
            "predictor_delta_l2": delta_norm,
            "n_free": len(free),
            "n_lower": len(lower_idx),
            "n_upper": len(upper_idx),
            "free_assets": ",".join(sample.tickers[k] for k in free),
            "kkt_system_condition": kkt_condition,
            "cvx_gradient_l2": float(np.linalg.norm(gradient_cvx)),
            "implicit_gradient_l2": float(np.linalg.norm(gradient_kkt)),
            "saved_batch_vs_single_gradient_l2": float(
                np.linalg.norm(gradient_saved - gradient_cvx)
            ),
            "cvx_vs_implicit_gradient_l2": float(
                np.linalg.norm(gradient_cvx - gradient_kkt)
            ),
            "cvx_vs_clarabel_weight_max": float(
                np.abs(weights_cvx - weights).max()
            ),
            "kkt_stationarity_inf": answer["kkt_stationarity_inf"],
            "kkt_complementarity_inf": answer["kkt_complementarity_inf"],
            "min_active_bound_dual": (
                float(active_duals.min())
                if len(active_duals) else np.nan
            ),
            "max_free_dual": (
                float(np.max(np.maximum(
                    answer["lower_dual"][free],
                    answer["upper_dual"][free],
                )))
                if len(free) else np.nan
            ),
            "min_free_slack": (
                float(np.min(np.minimum(
                    weights[free],
                    args.max_weight - weights[free],
                )))
                if len(free) else np.nan
            ),
        })

        for direction_name, direction in directions.items():
            directional_cvx = float(
                gradient_cvx @ direction
            )
            directional_kkt = float(
                gradient_kkt @ direction
            )

            cost_base = true_cost(
                weights, y, factor,
            )

            step_sizes = sorted(set([
                1e-3, 1e-4, 1e-5,
                1e-6, 1e-7, 1e-8,
                delta_norm,
                delta_norm / 4,
            ]), reverse=True)

            for h in step_sizes:
                plus = independent_solution(
                    mu + h * direction,
                    factor,
                    args.max_weight,
                )
                minus = independent_solution(
                    mu - h * direction,
                    factor,
                    args.max_weight,
                )

                cost_plus = true_cost(
                    plus["weights"], y, factor,
                )
                cost_minus = true_cost(
                    minus["weights"], y, factor,
                )

                fd_forward = (cost_plus - cost_base) / h
                fd_backward = (cost_base - cost_minus) / h
                fd_central = (cost_plus - cost_minus) / (2 * h)

                signature_plus = active_signature(
                    plus["weights"], args.bound_tol, args.max_weight,
                )
                signature_minus = active_signature(
                    minus["weights"], args.bound_tol, args.max_weight,
                )

                fd_rows.append({
                    "key": key,
                    "direction": direction_name,
                    "h": h,
                    "h_over_predictor_delta": h / delta_norm,
                    "cvx_directional": directional_cvx,
                    "implicit_directional": directional_kkt,
                    "fd_forward": fd_forward,
                    "fd_backward": fd_backward,
                    "fd_central": fd_central,
                    "fd_forward_minus_implicit": (
                        fd_forward - directional_kkt
                    ),
                    "fd_backward_minus_implicit": (
                        fd_backward - directional_kkt
                    ),
                    "fd_central_minus_implicit": (
                        fd_central - directional_kkt
                    ),
                    "fd_central_minus_cvx": (
                        fd_central - directional_cvx
                    ),
                    "same_active_plus": (
                        signature_plus == signature
                    ),
                    "same_active_minus": (
                        signature_minus == signature
                    ),
                    "plus_weight_change_max": float(
                        np.abs(plus["weights"] - weights).max()
                    ),
                    "minus_weight_change_max": float(
                        np.abs(minus["weights"] - weights).max()
                    ),
                    "plus_kkt_residual": (
                        plus["kkt_stationarity_inf"]
                    ),
                    "minus_kkt_residual": (
                        minus["kkt_stationarity_inf"]
                    ),
                })

    out.mkdir(parents=True)

    summary = pd.DataFrame(rows)
    fd = pd.DataFrame(fd_rows)

    summary.to_csv(
        out / "gradient_validation.csv",
        index=False,
    )
    fd.to_csv(
        out / "finite_difference_directional.csv",
        index=False,
    )

    print(
        f"\n[CASE] {args.risk} seed={args.seed} "
        f"step={args.step} date={selected_date} "
        f"batch_index={args.batch_index}"
    )
    print(
        f"[PREDICTOR] Delta_mu L2={delta_norm:.9g} | "
        f"free axis={sample.tickers[i]}/{sample.tickers[j]}"
    )

    columns = [
        "key",
        "n_free",
        "cvx_gradient_l2",
        "implicit_gradient_l2",
        "cvx_vs_implicit_gradient_l2",
        "saved_batch_vs_single_gradient_l2",
        "cvx_vs_clarabel_weight_max",
    ]

    print(
        "\n[GRADIENT VALIDATION]\n"
        + summary[columns].to_string(index=False)
    )

    at_delta = fd[
        np.isclose(fd["h"], delta_norm, atol=0, rtol=1e-10)
    ]

    fd_columns = [
        "key",
        "direction",
        "h",
        "fd_forward",
        "fd_backward",
        "fd_central",
        "implicit_directional",
        "cvx_directional",
        "same_active_plus",
        "same_active_minus",
    ]

    print(
        "\n[FINITE DIFFERENCE AT h = ||Predictor Delta||]\n"
        + at_delta[fd_columns].to_string(index=False)
    )

    stable_count = int(
        (fd["same_active_plus"] & fd["same_active_minus"]).sum()
    )

    print(
        f"[ACTIVE-SET STABLE PROBES] "
        f"{stable_count}/{len(fd)}"
    )
    print(f"\n[DONE] {out.resolve()}")


if __name__ == "__main__":
    main()
