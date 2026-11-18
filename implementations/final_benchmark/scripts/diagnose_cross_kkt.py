
"""Replay critical DFL training steps and diagnose frozen predictor x covariance.

Run from repository root. Existing checkpoints and training outputs are preserved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset, chronological_split, fit_feature_standardizer,
)
from implementations.covariance_baselines.src.dataset import CovarianceBaselineDataset
from implementations.dfl_mvo_lee2025.src.optimization import covariance_to_risk_factor
from implementations.dfl_mvo_lee2025.src.losses import markowitz_cost
from implementations.final_benchmark.scripts.run_solver_precision_2x2 import (
    make_arms, forward_objective, oracle_costs, evaluate,
)


def parse_args():
    p = argparse.ArgumentParser(description="Frozen Predictor x Covariance KKT diagnosis")
    p.add_argument("--risk", choices=["ridge", "sample"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--target-step", type=int, required=True)
    p.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    p.add_argument("--out-dir", default="implementations/final_benchmark/outputs/diagnostics/cross_kkt")
    p.add_argument("--label", default="first")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--risk-aversion", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mse-scale", type=float, default=30.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--perturbation", type=float, default=1e-6)
    p.add_argument("--direction-seed", type=int, default=20261110)
    p.add_argument("--tight-eps", type=float, default=1e-8)
    p.add_argument("--tight-max-iters", type=int, default=10000)
    p.add_argument("--boundary-tol", type=float, default=1e-6)
    p.add_argument("--dual-tol", type=float, default=1e-7)
    p.add_argument("--reference-csv", default="")
    return p.parse_args()


def solve_independent_qp(mu, factor, max_weight=0.2):
    """Solve QP independently using CLARABEL and compute KKT residuals."""
    n = len(mu)
    w = cp.Variable(n)
    p_mu = cp.Parameter(n)
    p_r = cp.Parameter((n, n))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(p_r @ w) - p_mu @ w),
        constraints,
    )

    p_mu.value = np.asarray(mu, dtype=np.float64)
    p_r.value = np.asarray(factor, dtype=np.float64)

    try:
        problem.solve(
            solver=cp.CLARABEL,
            warm_start=False,
            verbose=False,
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
            max_iter=1000,
        )
    except cp.SolverError:
        problem.solve(
            solver=cp.CLARABEL,
            warm_start=False,
            verbose=False,
        )

    if w.value is None or any(c.dual_value is None for c in constraints):
        raise RuntimeError(f"Independent CLARABEL failed: {problem.status}")

    solution = np.asarray(w.value, dtype=np.float64).reshape(-1)
    equality_dual = float(np.asarray(constraints[0].dual_value).item())
    lower_dual = np.asarray(constraints[1].dual_value, dtype=np.float64).reshape(-1)
    upper_dual = np.asarray(constraints[2].dual_value, dtype=np.float64).reshape(-1)

    sigma = factor.T @ factor
    stationarity = (
        2 * (sigma @ solution)
        - mu
        + equality_dual
        - lower_dual
        + upper_dual
    )

    return {
        "weights": solution,
        "lower_dual": lower_dual,
        "upper_dual": upper_dual,
        "status": problem.status,
        "objective": float(problem.value),
        "kkt_stationarity_inf": float(np.abs(stationarity).max()),
        "kkt_eq_abs": float(abs(solution.sum() - 1)),
        "kkt_box_violation": float(max(
            0.0, -solution.min(), solution.max() - max_weight,
        )),
        "kkt_complementarity_inf": float(max(
            np.abs(lower_dual * solution).max(),
            np.abs(upper_dual * (max_weight - solution)).max(),
        )),
        "kkt_dual_violation": float(max(
            0.0, -lower_dual.min(), -upper_dual.min(),
        )),
    }


def evaluate_cross(arms, x, y, ix, args):
    """Evaluate four combinations without updating Predictor parameters."""
    covariances = {
        "base": arms["tight_base"]["covariance"][ix],
        "perturbed": arms["tight_perturbed"]["covariance"][ix],
    }
    predictors = {
        "base": arms["tight_base"]["model"],
        "perturbed": arms["tight_perturbed"]["model"],
    }
    solver_args = arms["tight_base"]["solver_args"]

    results, sample_rows = {}, []

    for predictor_name, model in predictors.items():
        model.train()

        for covariance_name, covariance in covariances.items():
            key = f"P_{predictor_name}__S_{covariance_name}"
            model.zero_grad(set_to_none=True)

            mu = model.return_model(x[ix])
            factor = covariance_to_risk_factor(
                covariance, args.risk_aversion,
            )

            (weights,) = model.markowitz_layer(
                mu.to(torch.float64),
                factor,
                solver_args=solver_args,
            )

            costs = markowitz_cost(
                weights, y[ix].to(torch.float64), factor,
            )
            mse = F.mse_loss(mu, y[ix])
            objective = (
                args.alpha * costs.mean()
                + (1 - args.alpha) * args.mse_scale * mse.to(torch.float64)
            )

            # Gradient of only the decision-focused cost.
            g_mu_cost = torch.autograd.grad(
                costs.mean(), mu, retain_graph=True,
            )[0]

            # Gradient of the combined objective w.r.t. predictions.
            g_mu_total = torch.autograd.grad(
                objective, mu, retain_graph=True,
            )[0]

            # Gradient of the combined objective w.r.t. MLP parameters.
            theta_grads = torch.autograd.grad(
                objective, tuple(model.return_model.parameters()),
            )
            g_theta = torch.cat([
                g.reshape(-1) for g in theta_grads
            ]).detach().cpu().numpy().copy()

            results[key] = {
                "theta_gradient": g_theta,
                "mu_gradient_total": g_mu_total.detach().cpu().numpy().copy(),
                "mu_gradient_cost": g_mu_cost.detach().cpu().numpy().copy(),
                "mu": mu.detach().cpu().numpy().copy(),
                "weights": weights.detach().cpu().numpy().copy(),
                "objective": float(objective.detach()),
            }

            for j, row_ix in enumerate(ix):
                independent = solve_independent_qp(
                    results[key]["mu"][j].astype(np.float64),
                    factor[j].detach().cpu().numpy(),
                )

                w_ind = independent["weights"]
                lower = w_ind <= args.boundary_tol
                upper = w_ind >= 0.2 - args.boundary_tol
                dual_lower = independent["lower_dual"] > args.dual_tol
                dual_upper = independent["upper_dual"] > args.dual_tol

                sample_rows.append({
                    "key": key,
                    "sample_index": int(row_ix),
                    "batch_index": j,
                    "date": str(args.target_dates[int(row_ix)]),
                    "cvxpylayer_vs_clarabel_wmax": float(
                        np.abs(results[key]["weights"][j] - w_ind).max()
                    ),
                    "mu_cost_grad_l2": float(
                        np.linalg.norm(results[key]["mu_gradient_cost"][j])
                    ),
                    "mu_total_grad_l2": float(
                        np.linalg.norm(results[key]["mu_gradient_total"][j])
                    ),
                    "n_lower": int(lower.sum()),
                    "n_upper": int(upper.sum()),
                    "n_free": int((~(lower | upper)).sum()),
                    "box_simplex_licq_risk": bool((lower | upper).all()),
                    "n_lower_dual_active": int(dual_lower.sum()),
                    "n_upper_dual_active": int(dual_upper.sum()),
                    "lower_idx": ",".join(np.asarray(args.tickers)[lower]),
                    "upper_idx": ",".join(np.asarray(args.tickers)[upper]),
                    "lower_dual_idx": ",".join(np.asarray(args.tickers)[dual_lower]),
                    "upper_dual_idx": ",".join(np.asarray(args.tickers)[dual_upper]),
                    **{
                        k: v for k, v in independent.items()
                        if k not in {"weights", "lower_dual", "upper_dual"}
                    },
                })

    return results, pd.DataFrame(sample_rows)


def make_comparisons(results):
    pairs = [
        ("cov_effect_at_P_base", "P_base__S_base", "P_base__S_perturbed"),
        ("cov_effect_at_P_pert", "P_perturbed__S_base", "P_perturbed__S_perturbed"),
        ("predictor_effect_at_S_base", "P_base__S_base", "P_perturbed__S_base"),
        ("predictor_effect_at_S_pert", "P_base__S_perturbed", "P_perturbed__S_perturbed"),
        ("on_path_difference", "P_base__S_base", "P_perturbed__S_perturbed"),
    ]

    rows, per_sample = [], []

    for name, akey, bkey in pairs:
        a, b = results[akey], results[bkey]
        ga, gb = a["theta_gradient"], b["theta_gradient"]

        na, nb = np.linalg.norm(ga), np.linalg.norm(gb)
        cosine = float(np.dot(ga, gb) / (na * nb)) if na > 0 and nb > 0 else np.nan

        d_cost = b["mu_gradient_cost"] - a["mu_gradient_cost"]
        d_total = b["mu_gradient_total"] - a["mu_gradient_total"]
        d_weight = b["weights"] - a["weights"]

        rows.append({
            "contrast": name,
            "theta_grad_diff_l2": float(np.linalg.norm(gb - ga)),
            "theta_grad_cosine": cosine,
            "mu_cost_grad_diff_l2": float(np.linalg.norm(d_cost)),
            "mu_total_grad_diff_l2": float(np.linalg.norm(d_total)),
            "mu_prediction_max_abs_diff": float(
                np.abs(b["mu"] - a["mu"]).max()
            ),
            "weight_max_abs_diff": float(np.abs(d_weight).max()),
            "proxy_objective_diff": b["objective"] - a["objective"],
        })

        per_sample.append({
            "contrast": name,
            "mu_cost_grad_diff_per_sample": np.linalg.norm(
                d_cost, axis=1,
            ).tolist(),
            "weight_diff_per_sample": np.max(
                np.abs(d_weight), axis=1,
            ).tolist(),
        })

    return pd.DataFrame(rows), per_sample


def replay(args, x, y, train_ix, val_ix, arms):
    """Reproduce original training up to the update before target-step."""
    for arm in arms.values():
        arm["val_oracle"] = oracle_costs(
            arm["model"],
            y,
            arm["covariance"],
            val_ix,
            arm["solver_args"],
            args,
        )

    generator = torch.Generator().manual_seed(args.seed)
    step = 0
    per_epoch = math.ceil(len(train_ix) / args.batch_size)

    for epoch in range(1, math.ceil(args.target_step / per_epoch) + 1):
        order = train_ix[
            torch.randperm(
                len(train_ix),
                generator=generator,
            ).numpy()
        ]

        for start in range(0, len(order), args.batch_size):
            step += 1
            ix = order[start:start + args.batch_size]

            # Do not update on the target step.
            if step == args.target_step:
                print(
                    f"[SNAPSHOT] Epoch={epoch} Step={step} B={len(ix)}",
                    flush=True,
                )
                return ix

            for arm in arms.values():
                model = arm["model"]
                model.train()
                arm["optimizer"].zero_grad(set_to_none=True)

                loss, _, _, _, _ = forward_objective(
                    model,
                    x[ix],
                    y[ix],
                    arm["covariance"][ix],
                    arm["solver_args"],
                    args,
                )

                loss.backward()
                arm["optimizer"].step()

            if step == 1 or step % 100 == 0:
                print(
                    f"[REPLAY] Step {step} / {args.target_step - 1}",
                    flush=True,
                )

        # Match the validation solver calls in the original training.
        for arm in arms.values():
            evaluate(
                arm["model"],
                x,
                y,
                arm["covariance"],
                val_ix,
                arm["solver_args"],
                args,
                reference=arm["val_oracle"],
            )

    raise RuntimeError("Target-step was not reached")


def main():
    args = parse_args()

    if args.target_step <= 1 or args.batch_size < 1 or args.perturbation <= 0:
        raise ValueError("Invalid target-step, batch-size, or perturbation")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    out = (
        Path(args.out_dir)
        / args.risk
        / f"seed_{args.seed}"
        / f"step_{args.target_step}"
        / args.label
    )

    if out.exists():
        raise FileExistsError(
            f"Output exists: {out}. Use another --label."
        )

    out.mkdir(parents=True, exist_ok=False)

    common = dict(
        price_csv=args.price_csv,
        lookback=args.lookback,
        covariance_jitter=1e-6,
        dtype=torch.float64,
    )

    sample = RollingMVODataset(
        **common,
        risk_cache_dir="implementations/dfl_mvo_lee2025/cache/risk_matrix",
    )
    ridge = CovarianceBaselineDataset(
        **common,
        method="ridge",
        ridge_shrinkage=args.rho,
    )

    if (
        sample.tickers != ridge.tickers
        or not sample.target_dates.equals(ridge.target_dates)
    ):
        raise RuntimeError("Sample/Ridge alignment mismatch")

    train, val, _ = chronological_split(
        sample,
        args.train_end,
        args.validation_end,
    )

    train_ix = np.asarray(train.indices, dtype=int)
    val_ix = np.asarray(val.indices, dtype=int)

    mean, std = fit_feature_standardizer(sample, train)

    features = torch.stack([
        sample[i]["features"].float()
        for i in range(len(sample))
    ])
    x = (features - mean.float()) / std.float()

    y = torch.stack([
        sample[i]["target"].float()
        for i in range(len(sample))
    ])

    sample_cov = sample.covariances.detach().cpu().numpy()
    ridge_cov = ridge.covariances.detach().cpu().numpy()
    base = sample_cov if args.risk == "sample" else ridge_cov

    rng = np.random.default_rng(args.direction_seed)
    direction = rng.normal(size=sample.n_assets)
    direction /= np.linalg.norm(direction)

    delta = (
        args.perturbation
        * np.linalg.norm(sample_cov, axis=(1, 2))[:, None, None]
        * np.outer(direction, direction)[None]
    )

    covariances = {
        "base": torch.from_numpy(
            np.ascontiguousarray(base)
        ),
        "perturbed": torch.from_numpy(
            np.ascontiguousarray(base + delta)
        ),
    }

    args.target_dates = sample.target_dates.strftime("%Y-%m-%d")
    args.tickers = sample.tickers

    arms = make_arms(
        args,
        sample.n_assets,
        covariances,
    )

    ix = replay(
        args,
        x,
        y,
        train_ix,
        val_ix,
        arms,
    )

    results, independent = evaluate_cross(
        arms, x, y, ix, args,
    )
    comparisons, per_sample = make_comparisons(results)

    independent.to_csv(
        out / "independent_kkt_by_sample.csv",
        index=False,
    )
    comparisons.to_csv(
        out / "cross_gradient_comparison.csv",
        index=False,
    )
    pd.DataFrame(per_sample).to_json(
        out / "cross_by_sample.json",
        orient="records",
        indent=2,
    )

    np.savez_compressed(
        out / "cross_arrays.npz",
        **{
            f"{key}__{field}": values
            for key, item in results.items()
            for field, values in item.items()
            if isinstance(values, np.ndarray)
        },
    )

    saved = vars(args).copy()
    saved["target_dates"] = None
    saved["tickers"] = sample.tickers
    saved["batch_indices"] = ix.tolist()
    saved["batch_dates"] = [
        str(args.target_dates[i]) for i in ix
    ]

    (out / "config.json").write_text(
        json.dumps(saved, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    reference = (
        Path(args.reference_csv)
        if args.reference_csv
        else (
            Path("implementations/final_benchmark/outputs/diagnostics/step_replay")
            / args.risk
            / f"seed_{args.seed}"
            / "aa_focus"
            / "divergence_trace.csv"
        )
    )

    if reference.is_file():
        history = pd.read_csv(reference)
        match = history[
            (history["step"] == args.target_step)
            & (history["contrast"] == "perturb_tight")
        ]

        if not match.empty:
            observed = match.iloc[0]
            cross = comparisons.set_index("contrast").loc[
                "on_path_difference"
            ]

            for old_name, new_name in [
                ("gradient_diff_l2", "theta_grad_diff_l2"),
                ("pre_step_prediction_max_abs_diff", "mu_prediction_max_abs_diff"),
                ("weight_max_abs_diff", "weight_max_abs_diff"),
            ]:
                error = abs(
                    float(observed[old_name])
                    - float(cross[new_name])
                )
                print(
                    f"[REPLAY CHECK] {old_name}: abs_error={error:.3e}"
                )
    else:
        print(
            f"[REPLAY CHECK] Reference not found: {reference}"
        )

    print(
        "\n[CROSS GRADIENT]\n"
        + comparisons.to_string(index=False)
    )
    print(
        "\n[KKT] Maximum stationarity residual:",
        independent.kkt_stationarity_inf.max(),
    )
    print(
        "[KKT] Maximum CVXPYLayer-Clarabel weight difference:",
        independent.cvxpylayer_vs_clarabel_wmax.max(),
    )
    print(
        "[KKT] All-bound samples (potential LICQ failure):",
        independent.groupby("key").box_simplex_licq_risk.sum().to_dict(),
    )
    print(f"\n[DONE] {out.resolve()}")


if __name__ == "__main__":
    main()
