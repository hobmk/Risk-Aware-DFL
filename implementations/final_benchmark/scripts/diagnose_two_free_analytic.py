
"""Analytical two-free-asset KKT diagnosis.

Uses cross_kkt results. No retraining is required.
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
    p = argparse.ArgumentParser(description="Analytical two-free-asset KKT validation")
    p.add_argument("--risk", choices=["sample", "ridge"], default="sample")
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--step", type=int, default=852)
    p.add_argument("--batch-index", type=int, default=4)
    p.add_argument("--covariance", choices=["base", "perturbed"], default="base")
    p.add_argument("--asset-i", default="CRM")
    p.add_argument("--asset-j", default="MMM")
    p.add_argument("--root", default="implementations/final_benchmark/outputs/diagnostics/cross_kkt")
    p.add_argument("--label", default="analytic_v1")
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--boundary-tol", type=float, default=1e-6)
    p.add_argument("--tight-eps", type=float, default=1e-8)
    p.add_argument("--ultra-eps", type=float, default=1e-10)
    p.add_argument("--ultra-max-iters", type=int, default=100000)
    return p.parse_args()


def analytical_two_free(mu, y, q, fixed, i, j, cap):
    n = len(mu)
    d = np.zeros(n)
    d[i], d[j] = 1.0, -1.0

    w0 = fixed.copy()
    remaining = 1.0 - fixed.sum()
    w0[i], w0[j] = 0.0, remaining

    lo = max(0.0, remaining - cap)
    hi = min(cap, remaining)

    a = float(d @ q @ d)

    if a <= 0 or lo > hi + 1e-12:
        raise ValueError("Invalid analytical subproblem")

    z_raw = float(
        (d @ mu - 2.0 * d @ q @ w0) / (2.0 * a)
    )
    z = float(np.clip(z_raw, lo, hi))
    w = w0 + z * d

    cost_slope = float(2.0 * d @ q @ w - y @ d)

    if lo < z_raw < hi:
        gradient = d * (cost_slope / (2.0 * a))
    else:
        gradient = np.zeros(n)

    # Verify KKT conditions for the full 30-asset problem.
    g = 2.0 * q @ w - mu

    lower = w <= 1e-12
    upper = w >= cap - 1e-12
    free = ~(lower | upper)

    if free.any():
        nu = -float(g[free].mean())
    else:
        lower_limit = max((-g[lower]).tolist(), default=-np.inf)
        upper_limit = min((-g[upper]).tolist(), default=np.inf)
        nu = (lower_limit + upper_limit) / 2.0

    free_residual = (
        float(np.abs(g[free] + nu).max())
        if free.any() else 0.0
    )
    lower_violation = (
        max(0.0, -float((g[lower] + nu).min()))
        if lower.any() else 0.0
    )
    upper_violation = (
        max(0.0, float((g[upper] + nu).max()))
        if upper.any() else 0.0
    )

    dual_violation = max(
        free_residual, lower_violation, upper_violation
    )

    return {
        "w": w,
        "z": z,
        "z_unclipped": z_raw,
        "lo": lo,
        "hi": hi,
        "a": a,
        "direction": d,
        "gradient": gradient,
        "dual_violation": dual_violation,
        "n_lower": int(lower.sum()),
        "n_upper": int(upper.sum()),
        "n_free": int(free.sum()),
        "value": float(w @ q @ w - y @ w),
    }


def cvx_layer_probe(layer, mu, y, factor, solver_args):
    prediction = torch.tensor(
        mu, dtype=torch.float32, requires_grad=True
    )
    r = torch.tensor(factor, dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.float64)

    (weights,) = layer(
        prediction.to(torch.float64),
        r,
        solver_args=solver_args,
    )

    cost = markowitz_cost(weights, target, r)
    (gradient,) = torch.autograd.grad(cost, prediction)

    return (
        gradient.detach().numpy().astype(np.float64),
        weights.detach().numpy(),
    )


def main():
    args = parse_args()

    if args.batch_index < 0 or args.max_weight != 0.2:
        raise ValueError("Invalid batch-index or max-weight")

    torch.set_num_threads(1)

    cross = (
        Path(args.root)
        / args.risk
        / f"seed_{args.seed}"
        / f"step_{args.step}"
        / "first"
    )
    out = cross / args.label / args.covariance

    if out.exists():
        raise FileExistsError(
            f"Output exists: {out}. Change --label."
        )

    config = json.loads(
        (cross / "config.json").read_text(encoding="utf-8")
    )
    records = pd.read_csv(
        cross / "independent_kkt_by_sample.csv"
    )

    selected = records[
        (records["key"] == "P_base__S_base")
        & (records["batch_index"] == args.batch_index)
    ]

    if len(selected) != 1:
        raise ValueError("Requested sample not found")

    sample_index = int(selected.iloc[0]["sample_index"])

    common = dict(
        price_csv=config["price_csv"],
        lookback=int(config["lookback"]),
        covariance_jitter=1e-6,
        dtype=torch.float64,
    )

    dataset = RollingMVODataset(
        **common,
        risk_cache_dir="implementations/dfl_mvo_lee2025/cache/risk_matrix",
    )

    selected_date = str(dataset.target_dates[sample_index].date())

    if selected_date != str(selected.iloc[0]["date"]):
        raise RuntimeError("Sample index/date mismatch")

    sample_cov = dataset.covariances[
        sample_index
    ].detach().numpy()

    base_cov = sample_cov

    if args.risk == "ridge":
        ridge = CovarianceBaselineDataset(
            **common,
            method="ridge",
            ridge_shrinkage=float(config["rho"]),
        )
        base_cov = ridge.covariances[
            sample_index
        ].detach().numpy()

    rng = np.random.default_rng(
        int(config["direction_seed"])
    )
    v = rng.normal(size=dataset.n_assets)
    v /= np.linalg.norm(v)

    delta = (
        float(config["perturbation"])
        * np.linalg.norm(sample_cov)
        * np.outer(v, v)
    )

    covariance = (
        base_cov if args.covariance == "base"
        else base_cov + delta
    )

    factor = covariance_to_risk_factor(
        torch.tensor(covariance, dtype=torch.float64),
        float(config["risk_aversion"]),
    ).detach().numpy()

    q = factor.T @ factor

    y = dataset[sample_index]["target"].float().detach().numpy()
    y = y.astype(np.float64)

    saved = np.load(cross / "cross_arrays.npz")

    mu0 = saved[
        "P_base__S_base__mu"
    ][args.batch_index].astype(np.float64)

    mu1 = saved[
        "P_perturbed__S_base__mu"
    ][args.batch_index].astype(np.float64)

    delta_mu = mu1 - mu0
    delta_norm = float(np.linalg.norm(delta_mu))

    i = dataset.tickers.index(args.asset_i)
    j = dataset.tickers.index(args.asset_j)

    clarabel0 = solve_independent_qp(
        mu0, factor, max_weight=args.max_weight
    )

    # Fix the other 28 assets to their active bounds.
    fixed = np.zeros(dataset.n_assets)

    for k, weight in enumerate(clarabel0["weights"]):
        if k in (i, j):
            continue

        if weight >= args.max_weight - args.boundary_tol:
            fixed[k] = args.max_weight
        elif abs(weight) > args.boundary_tol:
            raise ValueError(
                f"Unexpected free asset: {dataset.tickers[k]}"
            )

    remaining = 1.0 - fixed.sum()

    print(
        f"[SETUP] date={selected_date} | "
        f"free={args.asset_i},{args.asset_j} | "
        f"remaining={remaining:.10g}",
        flush=True,
    )

    def exact(mu):
        return analytical_two_free(
            mu, y, q, fixed, i, j, args.max_weight
        )

    base = exact(mu0)
    pert = exact(mu1)

    d = base["direction"]
    dz_dt = float(d @ delta_mu / (2.0 * base["a"]))

    thresholds = {}

    if abs(dz_dt) > 1e-18:
        thresholds = {
            "z_lower": float(
                (base["lo"] - base["z_unclipped"]) / dz_dt
            ),
            "z_upper": float(
                (base["hi"] - base["z_unclipped"]) / dz_dt
            ),
        }

    print(
        f"[PREDICTOR] ||delta_mu||={delta_norm:.9e} "
        f"| dz/dt={dz_dt:.9e}",
        flush=True,
    )
    print(f"[THRESHOLDS] {thresholds}", flush=True)

    for name, answer in [
        ("P_base", base),
        ("P_perturbed", pert),
    ]:
        print(
            f"[{name}] "
            f"z_raw={answer['z_unclipped']:.12g} "
            f"z={answer['z']:.12g} "
            f"n_free={answer['n_free']} "
            f"KKT_violation={answer['dual_violation']:.3e} "
            f"gradient_norm={np.linalg.norm(answer['gradient']):.6g}",
            flush=True,
        )

    path_rows = []

    for t in np.linspace(-0.25, 1.25, 151):
        result = exact(mu0 + float(t) * delta_mu)

        path_rows.append({
            "t": float(t),
            "z_unclipped": result["z_unclipped"],
            "z": result["z"],
            "n_free": result["n_free"],
            "global_kkt_violation": result["dual_violation"],
            "analytic_grad_norm": float(
                np.linalg.norm(result["gradient"])
            ),
            "analytical_cost": result["value"],
        })

    probe_ts = [0.0, 0.25, 0.5, 0.75, 1.0]

    for critical_t in thresholds.values():
        if -0.25 <= critical_t <= 1.25:
            probe_ts.extend([
                critical_t - 0.01,
                critical_t + 0.01,
            ])

    probe_ts = sorted(set(
        round(float(t), 10) for t in probe_ts
    ))

    layer = build_markowitz_layer(
        dataset.n_assets,
        max_weight=args.max_weight,
    )

    solver_configs = {
        "tight": {
            "eps": args.tight_eps,
            "max_iters": 10000,
            "acceleration_lookback": 0,
        },
        "ultra": {
            "eps": args.ultra_eps,
            "max_iters": args.ultra_max_iters,
            "acceleration_lookback": 0,
        },
    }

    probes = []

    for t in probe_ts:
        mu = mu0 + t * delta_mu
        analytic = exact(mu)

        independent = solve_independent_qp(
            mu, factor, max_weight=args.max_weight
        )

        row = {
            "t": t,
            "z_analytic": analytic["z"],
            "z_raw": analytic["z_unclipped"],
            "analytic_n_free": analytic["n_free"],
            "analytic_global_KKT_violation": (
                analytic["dual_violation"]
            ),
            "analytic_gradient_norm": float(
                np.linalg.norm(analytic["gradient"])
            ),
            "clarabel_vs_analytic_wmax": float(
                np.max(np.abs(
                    independent["weights"] - analytic["w"]
                ))
            ),
            "clarabel_complementarity": (
                independent["kkt_complementarity_inf"]
            ),
        }

        for solver, solver_args in solver_configs.items():
            row[f"{solver}_gradient_norm"] = np.nan
            row[f"{solver}_gradient_diff_vs_analytic"] = np.nan
            row[f"{solver}_wmax_vs_analytic"] = np.nan
            row[f"{solver}_error"] = ""

            try:
                grad, weights = cvx_layer_probe(
                    layer, mu, y, factor, solver_args
                )

                row[f"{solver}_gradient_norm"] = float(
                    np.linalg.norm(grad)
                )
                row[f"{solver}_gradient_diff_vs_analytic"] = float(
                    np.linalg.norm(
                        grad - analytic["gradient"]
                    )
                )
                row[f"{solver}_wmax_vs_analytic"] = float(
                    np.max(np.abs(
                        weights - analytic["w"]
                    ))
                )

                if t in (0.0, 1.0) and solver == "tight":
                    predictor = (
                        "P_base" if t == 0.0 else "P_perturbed"
                    )
                    key = (
                        f"{predictor}__S_{args.covariance}"
                        "__mu_gradient_cost"
                    )
                    mu_key = (
                        f"{predictor}__S_{args.covariance}__mu"
                    )
                    saved_grad = (
                        saved[key][args.batch_index]
                        * len(saved[mu_key])
                    )
                    row["saved_vs_rerun_gradient_l2"] = float(
                        np.linalg.norm(saved_grad - grad)
                    )

            except Exception as exc:
                row[f"{solver}_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )[:200]

        probes.append(row)

    # Finite difference using the exact piecewise quadratic solution.
    fd_rows = []
    unit = delta_mu / delta_norm

    step_sizes = sorted(set([
        1e-8,
        1e-7,
        delta_norm / 100,
        delta_norm / 10,
        delta_norm,
    ]))

    for name, mu in [
        ("base", mu0),
        ("perturbed", mu1),
    ]:
        center = exact(mu)

        for h in step_sizes:
            plus = exact(mu + h * unit)
            minus = exact(mu - h * unit)

            fd_rows.append({
                "predictor": name,
                "h": h,
                "analytic_directional": float(
                    center["gradient"] @ unit
                ),
                "fd_forward": (
                    plus["value"] - center["value"]
                ) / h,
                "fd_backward": (
                    center["value"] - minus["value"]
                ) / h,
                "fd_central": (
                    plus["value"] - minus["value"]
                ) / (2 * h),
                "plus_global_kkt_violation": (
                    plus["dual_violation"]
                ),
                "minus_global_kkt_violation": (
                    minus["dual_violation"]
                ),
                "plus_n_free": plus["n_free"],
                "minus_n_free": minus["n_free"],
            })

    out.mkdir(parents=True)

    pd.DataFrame(path_rows).to_csv(
        out / "analytic_path.csv", index=False
    )
    pd.DataFrame(probes).to_csv(
        out / "solver_vs_analytic.csv", index=False
    )
    pd.DataFrame(fd_rows).to_csv(
        out / "analytic_finite_difference.csv", index=False
    )

    (out / "thresholds.json").write_text(
        json.dumps({
            "risk": args.risk,
            "seed": args.seed,
            "step": args.step,
            "sample_index": sample_index,
            "date": selected_date,
            "covariance": args.covariance,
            "assets": [args.asset_i, args.asset_j],
            "delta_mu_l2": delta_norm,
            "thresholds_t": thresholds,
            "kkt_violation_base": base["dual_violation"],
            "kkt_violation_perturbed": pert["dual_violation"],
        }, indent=2),
        encoding="utf-8",
    )

    summary = pd.DataFrame(probes)

    columns = [
        "t",
        "z_analytic",
        "analytic_n_free",
        "analytic_global_KKT_violation",
        "tight_gradient_norm",
        "tight_gradient_diff_vs_analytic",
        "ultra_gradient_norm",
        "ultra_gradient_diff_vs_analytic",
    ]

    print(
        "\n[PROBE SUMMARY]\n"
        + summary[columns].to_string(index=False),
        flush=True,
    )

    print(f"\n[DONE] {out.resolve()}", flush=True)
    print(
        "Interpret gradients only where the full KKT "
        "residual is small and the solution is differentiable."
    )


if __name__ == "__main__":
    main()
