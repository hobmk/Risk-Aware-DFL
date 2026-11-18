
"""KKT and finite-difference diagnostics for DFL covariance sensitivity.

Run from repository root:
python -m implementations.final_benchmark.scripts.diagnose_kkt_fd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset,
    chronological_split,
    fit_feature_standardizer,
)
from implementations.covariance_baselines.src.dataset import (
    CovarianceBaselineDataset,
)
from implementations.dfl_mvo_lee2025.scripts.train_mlp_markowitz_combined import (
    MLPWithMarkowitz,
)
from implementations.dfl_mvo_lee2025.src.optimization import covariance_to_risk_factor
from implementations.dfl_mvo_lee2025.src.losses import markowitz_cost


def parse_args():
    p = argparse.ArgumentParser(description="DFL KKT and finite-difference diagnostics")
    p.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    p.add_argument("--checkpoint-root", default="implementations/final_benchmark/outputs")
    p.add_argument("--out-dir", default="implementations/final_benchmark/outputs/diagnostics/kkt_fd")
    p.add_argument("--cases", nargs="+", default=["dfl_mvo:43:ridge:0.90", "ridge:48:sample:0.99"])
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--risk-aversion", type=float, default=1.0)
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--ridge-shrinkage", type=float, default=0.1)
    p.add_argument("--covariance-jitter", type=float, default=1e-6)
    p.add_argument("--perturbation", type=float, default=1e-6)
    p.add_argument("--direction-seed", type=int, default=20261110)
    p.add_argument("--active-tol", type=float, default=1e-5)
    p.add_argument("--fd-steps", nargs="+", type=float, default=[1e-3, 1e-4, 1e-5])
    p.add_argument("--tight-eps", type=float, default=1e-8)
    p.add_argument("--tight-max-iters", type=int, default=10000)
    p.add_argument("--no-tight", action="store_true")
    return p.parse_args()


def call_layer(layer, mu, factor, options):
    kw = {"solver_args": options} if options else {}
    (weights,) = layer(mu, factor, **kw)
    return weights


def solve_batch(model, mu, target, covariance, risk_aversion, options):
    mu_var = mu.detach().clone().requires_grad_(True)
    factor = covariance_to_risk_factor(
        covariance, risk_aversion=risk_aversion
    )
    weights = call_layer(model.markowitz_layer, mu_var, factor, options)
    costs = markowitz_cost(weights, target, factor)
    gradient = torch.autograd.grad(costs.sum(), mu_var)[0]
    return weights.detach().numpy(), gradient.detach().numpy()


def solve_one(model, mu_np, target_np, covariance_np, risk_aversion, options, grad=False):
    mu = torch.tensor(mu_np, dtype=torch.float64, requires_grad=grad)
    target = torch.tensor(target_np, dtype=torch.float64)
    covariance = torch.tensor(covariance_np, dtype=torch.float64)
    factor = covariance_to_risk_factor(
        covariance, risk_aversion=risk_aversion
    )

    weights = call_layer(model.markowitz_layer, mu, factor, options)
    cost = markowitz_cost(weights, target, factor)

    gradient = (
        torch.autograd.grad(cost, mu)[0].detach().numpy()
        if grad else None
    )

    return float(cost.detach()), weights.detach().numpy(), gradient


def kkt_from_cvxpy(mu, covariance, risk_aversion, max_weight, tol, reference_weights):
    """Independent CVXPY solve and primal-dual KKT diagnostics."""
    import cvxpy as cp

    n = len(mu)
    w = cp.Variable(n)

    eq = cp.sum(w) == 1
    lower = w >= 0
    upper = w <= max_weight

    objective = cp.Minimize(
        risk_aversion * cp.quad_form(w, cp.psd_wrap(covariance))
        - mu @ w
    )
    problem = cp.Problem(objective, [eq, lower, upper])

    installed = cp.installed_solvers()
    solver = "CLARABEL" if "CLARABEL" in installed else "OSQP"

    if solver not in installed:
        return {
            "independent_solver": "unavailable",
            "solver_status": "not_installed",
        }

    try:
        problem.solve(solver=solver, verbose=False)

        if w.value is None:
            return {
                "independent_solver": solver,
                "solver_status": str(problem.status),
            }

        x = np.asarray(w.value, dtype=float).reshape(-1)
        lo = np.asarray(lower.dual_value, dtype=float).reshape(-1)
        hi = np.asarray(upper.dual_value, dtype=float).reshape(-1)
        nu = float(np.asarray(eq.dual_value).item())

        residual = (
            2 * risk_aversion * covariance @ x
            - mu + nu - lo + hi
        )

        mask_lo = x <= tol
        mask_hi = x >= max_weight - tol
        mask_free = ~(mask_lo | mask_hi)

        active_duals = np.r_[lo[mask_lo], hi[mask_hi]]
        min_active_dual = (
            float(active_duals.min())
            if len(active_duals) else float("nan")
        )

        if mask_free.any():
            h = 2 * risk_aversion * covariance[
                np.ix_(mask_free, mask_free)
            ]
            ones = np.ones((len(h), 1))
            restricted = np.block([
                [h, ones],
                [ones.T, np.zeros((1, 1))],
            ])
            restricted_condition = float(
                np.linalg.cond(restricted)
            )
        else:
            restricted_condition = float("nan")

        return {
            "independent_solver": solver,
            "solver_status": str(problem.status),
            "independent_weight_diff_max": float(
                np.max(np.abs(x - reference_weights))
            ),
            "kkt_stationarity_max": float(
                np.max(np.abs(residual))
            ),
            "kkt_sum_error": float(
                abs(x.sum() - 1)
            ),
            "kkt_bound_violation_max": float(
                max(np.max(-x), np.max(x - max_weight), 0)
            ),
            "kkt_complementarity_max": float(
                max(
                    np.max(np.abs(lo * x)),
                    np.max(np.abs(hi * (max_weight - x))),
                )
            ),
            "n_lower": int(mask_lo.sum()),
            "n_upper": int(mask_hi.sum()),
            "n_free": int(mask_free.sum()),
            "min_active_dual": min_active_dual,
            "min_lower_dual": (
                float(np.min(lo[mask_lo]))
                if mask_lo.any() else float("nan")
            ),
            "min_upper_dual": (
                float(np.min(hi[mask_hi]))
                if mask_hi.any() else float("nan")
            ),
            "restricted_kkt_condition": restricted_condition,
            "all_bound_vertex": bool(mask_free.sum() == 0),
        }

    except Exception as error:
        return {
            "independent_solver": solver,
            "solver_status": f"ERROR: {error}",
        }


def load_predictor(args, n_assets, predictor, seed):
    model = MLPWithMarkowitz(
        n_assets=n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        dropout=0,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
    ).to("cpu")

    path = (
        Path(args.checkpoint_root)
        / predictor
        / f"lambda_{args.risk_aversion:g}"
        / f"seed_{seed}"
        / "best_model.pt"
    )

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )
    state = checkpoint["model_state_dict"]

    if any(key.startswith("return_model.") for key in state):
        state = {
            key[len("return_model."):]: value
            for key, value in state.items()
            if key.startswith("return_model.")
        }

    model.return_model.load_state_dict(state, strict=True)
    model.eval()
    return model, path


def nearest_batch(indices, sample_covariances, quantile, batch_size):
    eig = np.linalg.eigvalsh(sample_covariances[indices])
    conditions = eig[:, -1] / eig[:, 0]
    loc = int(
        np.argmin(
            abs(conditions - np.quantile(conditions, quantile))
        )
    )
    start = int(
        np.clip(
            loc - batch_size // 2,
            0,
            len(indices) - batch_size,
        )
    )
    return indices[start:start + batch_size]


def main():
    args = parse_args()

    if (
        args.batch_size < 1
        or args.perturbation <= 0
        or not args.fd_steps
        or any(h <= 0 for h in args.fd_steps)
    ):
        raise ValueError(
            "Check batch-size, perturbation and fd-steps"
        )

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    common = dict(
        price_csv=args.price_csv,
        lookback=args.lookback,
        covariance_jitter=args.covariance_jitter,
        dtype=torch.float64,
    )

    sample = RollingMVODataset(
        **common,
        risk_cache_dir="implementations/dfl_mvo_lee2025/cache/risk_matrix",
    )
    ridge = CovarianceBaselineDataset(
        **common,
        method="ridge",
        ridge_shrinkage=args.ridge_shrinkage,
    )

    if (
        sample.tickers != ridge.tickers
        or not sample.target_dates.equals(ridge.target_dates)
    ):
        raise RuntimeError(
            "Sample/Ridge dates or tickers do not match"
        )

    train, _, _ = chronological_split(
        sample, args.train_end, args.validation_end
    )
    indices = np.array(train.indices, dtype=int)
    mean, std = fit_feature_standardizer(sample, train)

    covariance_arrays = {
        "sample": sample.covariances.detach().cpu().numpy(),
        "ridge": ridge.covariances.detach().cpu().numpy(),
    }

    rng = np.random.default_rng(args.direction_seed)
    v = rng.normal(size=sample.n_assets)
    v /= np.linalg.norm(v)
    projector = np.outer(v, v)

    solver_configs = [("default", {})]

    if not args.no_tight:
        solver_configs.append((
            "tight_scs",
            {
                "eps": args.tight_eps,
                "max_iters": args.tight_max_iters,
                "acceleration_lookback": 0,
            },
        ))

    case_rows, fd_rows = [], []

    for item in args.cases:
        predictor, seed_str, risk, q_str = item.split(":")
        seed, q = int(seed_str), float(q_str)

        if (
            predictor not in {"dfl_mvo", "ridge"}
            or risk not in covariance_arrays
        ):
            raise ValueError(f"Invalid case: {item}")

        torch.manual_seed(seed)

        model, checkpoint = load_predictor(
            args, sample.n_assets, predictor, seed
        )

        ix = nearest_batch(
            indices,
            covariance_arrays["sample"],
            q,
            args.batch_size,
        )

        features = torch.stack([
            sample[int(i)]["features"] for i in ix
        ])
        features = (
            (features - mean) / std
        ).to(torch.float32)

        targets = torch.stack([
            sample[int(i)]["target"] for i in ix
        ]).to(torch.float64)

        with torch.no_grad():
            predicted = model.return_model(
                features
            ).to(torch.float64)

        cov_before_np = covariance_arrays[risk][ix]

        delta_np = (
            args.perturbation
            * np.linalg.norm(
                covariance_arrays["sample"][ix],
                axis=(1, 2),
            )[:, None, None]
            * projector[None]
        )

        cov_after_np = cov_before_np + delta_np

        base_cov = torch.from_numpy(
            np.ascontiguousarray(cov_before_np)
        )
        new_cov = torch.from_numpy(
            np.ascontiguousarray(cov_after_np)
        )

        print(
            f"\n[CASE] {item} | checkpoint={checkpoint}",
            flush=True,
        )

        batch_by_solver = {}

        for solver_name, options in solver_configs:
            before_w, before_g = solve_batch(
                model, predicted, targets, base_cov,
                args.risk_aversion, options,
            )
            after_w, after_g = solve_batch(
                model, predicted, targets, new_cov,
                args.risk_aversion, options,
            )

            delta_grad = np.linalg.norm(
                after_g - before_g, axis=1
            )

            batch_by_solver[solver_name] = (
                before_w, before_g,
                after_w, after_g,
                delta_grad,
            )

        # Choose the most sensitive observation under the default solver.
        worst = int(
            np.argmax(batch_by_solver["default"][4])
        )
        selected_index = int(ix[worst])
        date = str(
            pd.Timestamp(
                sample.target_dates[selected_index]
            ).date()
        )

        mu_np = predicted[worst].detach().numpy()
        target_np = targets[worst].detach().numpy()

        direction = (
            batch_by_solver["default"][3][worst]
            - batch_by_solver["default"][1][worst]
        )

        if np.linalg.norm(direction) < 1e-15:
            direction = (
                batch_by_solver["default"][1][worst].copy()
            )

        if np.linalg.norm(direction) < 1e-15:
            direction = np.ones(sample.n_assets)

        direction = direction / np.linalg.norm(direction)

        print(
            f"  Worst sample: {date}, "
            f"batch row={worst}, index={selected_index}",
            flush=True,
        )

        for solver_name, options in solver_configs:
            w0, g0, w1, g1, dg = batch_by_solver[solver_name]

            relative = (
                np.linalg.norm(g1 - g0)
                / max(np.linalg.norm(g0), 1e-30)
            )

            row = {
                "case": item,
                "predictor": predictor,
                "seed": seed,
                "risk": risk,
                "quantile": q,
                "solver": solver_name,
                "worst_date": date,
                "batch_regret_mu_gradient_rel_change": float(relative),
                "batch_regret_mu_gradient_abs_change": float(
                    np.linalg.norm(g1 - g0)
                ),
                "batch_weight_max_abs_change": float(
                    np.max(abs(w1 - w0))
                ),
                "worst_sample_mu_gradient_abs_change": float(
                    dg[worst]
                ),
                "covariance_condition_at_sample": float(
                    np.linalg.cond(cov_before_np[worst])
                ),
                "relative_covariance_change_at_sample": float(
                    np.linalg.norm(delta_np[worst])
                    / np.linalg.norm(cov_before_np[worst])
                ),
            }

            for name, cov_np, batch_w, batch_grad in [
                ("base", cov_before_np[worst], w0[worst], g0[worst]),
                ("perturbed", cov_after_np[worst], w1[worst], g1[worst]),
            ]:
                f0, single_w, autograd = solve_one(
                    model,
                    mu_np,
                    target_np,
                    cov_np,
                    args.risk_aversion,
                    options,
                    grad=True,
                )

                row[f"{name}_batch_single_grad_diff"] = float(
                    np.linalg.norm(autograd - batch_grad)
                )
                row[f"{name}_batch_single_weight_diff"] = float(
                    np.max(abs(single_w - batch_w))
                )

                check = kkt_from_cvxpy(
                    mu_np,
                    cov_np,
                    args.risk_aversion,
                    args.max_weight,
                    args.active_tol,
                    single_w,
                )

                for key, value in check.items():
                    row[f"{name}_{key}"] = value

                for h in args.fd_steps:
                    fplus, wp, _ = solve_one(
                        model,
                        mu_np + h * direction,
                        target_np,
                        cov_np,
                        args.risk_aversion,
                        options,
                    )
                    fminus, wm, _ = solve_one(
                        model,
                        mu_np - h * direction,
                        target_np,
                        cov_np,
                        args.risk_aversion,
                        options,
                    )

                    finite_difference = (
                        fplus - fminus
                    ) / (2 * h)

                    autodiff = float(
                        np.dot(autograd, direction)
                    )

                    fd_rows.append({
                        "case": item,
                        "solver": solver_name,
                        "covariance": name,
                        "date": date,
                        "fd_step": h,
                        "autograd_directional": autodiff,
                        "finite_difference_directional": float(
                            finite_difference
                        ),
                        "absolute_error": float(
                            abs(autodiff - finite_difference)
                        ),
                        "relative_error": float(
                            abs(autodiff - finite_difference)
                            / max(
                                abs(autodiff),
                                abs(finite_difference),
                                1e-10,
                            )
                        ),
                        "fd_weight_change_l2": float(
                            np.linalg.norm(wp - wm)
                        ),
                        "single_cost": f0,
                    })

            case_rows.append(row)

            print(
                f"  {solver_name:9s}: "
                f"batch |Δgrad_mu|="
                f"{row['batch_regret_mu_gradient_abs_change']:.3e}, "
                f"|Δw|max="
                f"{row['batch_weight_max_abs_change']:.3e}",
                flush=True,
            )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(case_rows).to_csv(
        out / "kkt_case_summary.csv",
        index=False,
    )
    pd.DataFrame(fd_rows).to_csv(
        out / "finite_difference.csv",
        index=False,
    )

    (out / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    print(f"\n[DONE] {out.resolve()}")
    print(
        "Send kkt_case_summary.csv and finite_difference.csv "
        "for interpretation."
    )


if __name__ == "__main__":
    main()
