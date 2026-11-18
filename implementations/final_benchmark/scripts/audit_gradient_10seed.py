
"""Audit 10-seed DFL gradient sensitivity and solver precision (no retraining)."""
from __future__ import annotations

import argparse
import json
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
from implementations.dfl_mvo_lee2025.scripts.train_mlp_markowitz_combined import MLPWithMarkowitz
from implementations.dfl_mvo_lee2025.src.optimization import covariance_to_risk_factor
from implementations.dfl_mvo_lee2025.src.losses import markowitz_cost


def parse_args():
    p = argparse.ArgumentParser(description="10-seed gradient sensitivity and solver precision audit")
    p.add_argument("--predictor-source", choices=["dfl_mvo", "ridge"], required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    p.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    p.add_argument("--checkpoint-root", default="implementations/final_benchmark/outputs")
    p.add_argument("--out-root", default="implementations/final_benchmark/outputs/diagnostics/gradient_audit")
    p.add_argument("--label", default="q50_q90_q99")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batches-per-quantile", type=int, default=1)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--risk-aversion", type=float, default=1.0)
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mse-scale", type=float, default=30.0)
    p.add_argument("--perturbation", type=float, default=1e-6)
    p.add_argument("--direction-seed", type=int, default=20261110)
    p.add_argument("--tight-eps", type=float, default=1e-8)
    p.add_argument("--ultra-eps", type=float, default=1e-10)
    p.add_argument("--tight-max-iters", type=int, default=10000)
    p.add_argument("--ultra-max-iters", type=int, default=100000)
    p.add_argument("--grad-threshold", type=float, default=0.01)
    p.add_argument("--solver-threshold", type=float, default=0.001)
    p.add_argument("--bound-tol", type=float, default=1e-6)
    p.add_argument("--boundary-warning", type=float, default=1e-4)
    return p.parse_args()


class IndependentQP:
    def __init__(self, n, cap):
        w = cp.Variable(n)
        self.mu = cp.Parameter(n)
        self.r = cp.Parameter((n, n))
        self.constraints = [cp.sum(w) == 1, w >= 0, w <= cap]
        self.problem = cp.Problem(
            cp.Minimize(cp.sum_squares(self.r @ w) - self.mu @ w),
            self.constraints,
        )
        self.w, self.cap = w, cap

    def solve(self, mu, factor, y, tol, near_tol):
        self.mu.value = mu.astype(np.float64)
        self.r.value = factor.astype(np.float64)

        try:
            self.problem.solve(
                solver=cp.CLARABEL,
                warm_start=False,
                tol_gap_abs=1e-11,
                tol_gap_rel=1e-11,
                tol_feas=1e-11,
                max_iter=1000,
            )
        except cp.SolverError:
            self.problem.solve(solver=cp.CLARABEL, warm_start=False)

        if self.w.value is None or any(c.dual_value is None for c in self.constraints):
            raise RuntimeError(f"CLARABEL status={self.problem.status}")

        w = np.asarray(self.w.value, dtype=np.float64).reshape(-1)
        lower_dual = np.asarray(self.constraints[1].dual_value).reshape(-1)
        upper_dual = np.asarray(self.constraints[2].dual_value).reshape(-1)
        nu = float(np.asarray(self.constraints[0].dual_value).item())

        q = factor.T @ factor
        stationarity = 2 * q @ w - mu + nu - lower_dual + upper_dual
        comp = max(
            np.abs(lower_dual * w).max(),
            np.abs(upper_dual * (self.cap - w)).max(),
        )

        state = np.where(
            w <= tol, 0,
            np.where(w >= self.cap - tol, 2, 1),
        )

        free = np.flatnonzero(state == 1)
        active_duals = np.concatenate((
            lower_dual[state == 0],
            upper_dual[state == 2],
        ))

        min_active_dual = (
            float(active_duals.min()) if len(active_duals) else np.nan
        )

        free_slack = np.minimum(w[free], self.cap - w[free])
        min_free_slack = (
            float(free_slack.min()) if len(free) else np.nan
        )

        near_boundary = bool(np.any(
            ((w > tol) & (w < near_tol))
            | ((w > self.cap - near_tol) & (w < self.cap - tol))
        ))

        reliable_kkt = (
            np.max(np.abs(stationarity)) < 1e-7
            and comp < 1e-8
        )

        candidate = np.zeros(len(mu), dtype=np.float64)

        if len(free) >= 2:
            f = len(free)
            system = np.zeros((f + 1, f + 1), dtype=np.float64)
            system[:f, :f] = 2 * q[np.ix_(free, free)]
            system[:f, f] = 1
            system[f, :f] = 1

            rhs = np.zeros((f + 1, f), dtype=np.float64)
            rhs[:f] = np.eye(f)

            dw = np.linalg.solve(system, rhs)[:f]
            candidate[free] = dw.T @ (2 * q @ w - y)[free]

        # Conservative diagnostic, not a global derivative certificate.
        clear = bool(
            reliable_kkt
            and not near_boundary
            and len(free) >= 2
            and min_free_slack > 1e-4
            and (not len(active_duals) or min_active_dual > 1e-5)
        )

        return {
            "weights": w,
            "state": state,
            "grad_kkt_candidate": candidate,
            "kkt_clear_interior": clear,
            "near_boundary": near_boundary,
            "n_free": len(free),
            "min_free_slack": min_free_slack,
            "min_active_dual": min_active_dual,
            "kkt_stationarity": float(np.max(np.abs(stationarity))),
            "kkt_complementarity": float(comp),
        }


def checkpoint_state(root, source, seed, risk_aversion):
    path = (
        Path(root) / source
        / f"lambda_{risk_aversion:g}"
        / f"seed_{seed}"
        / "best_model.pt"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    try:
        payload = torch.load(
            path, map_location="cpu", weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot safely load {path}: {exc}"
        ) from exc

    state = payload["model_state_dict"]

    if any(k.startswith("return_model.") for k in state):
        state = {
            k.removeprefix("return_model."): value
            for k, value in state.items()
            if k.startswith("return_model.")
        }

    return state


def select_stratified_batches(train_ix, conditions, batch_size, count):
    ranked = train_ix[np.argsort(conditions[train_ix])]
    window = batch_size * count

    if window >= len(ranked):
        raise ValueError("Requested batches exceed training samples")

    selected = []

    for quantile in (0.50, 0.90, 0.99):
        center = round(quantile * (len(ranked) - 1))
        start = int(np.clip(
            center - window // 2,
            0,
            len(ranked) - window,
        ))

        for b in range(count):
            ix = ranked[
                start + b * batch_size:
                start + (b + 1) * batch_size
            ]
            selected.append((f"q{int(quantile * 100)}", b, ix))

    combined = np.concatenate([v[2] for v in selected])

    if len(combined) != len(np.unique(combined)):
        raise ValueError("Overlapping condition-number windows")

    return selected


def run_solver(model, features, targets, factor, solver_args, alpha, mse_scale):
    mu = model.return_model(features)

    (weights,) = model.markowitz_layer(
        mu.to(torch.float64),
        factor,
        solver_args=solver_args,
    )

    costs = markowitz_cost(
        weights, targets.to(torch.float64), factor,
    )

    # costs.sum() yields individual-sample gradients,
    # rather than batch-mean gradients scaled by 1/B.
    (grad_mu,) = torch.autograd.grad(
        costs.sum(), mu, retain_graph=True,
    )

    mse = F.mse_loss(mu, targets)
    objective = (
        alpha * costs.mean()
        + (1 - alpha) * mse_scale * mse.to(torch.float64)
    )

    gradients = torch.autograd.grad(
        objective,
        tuple(model.return_model.parameters()),
    )

    grad_theta = torch.cat([
        g.reshape(-1) for g in gradients
    ]).detach().numpy().copy()

    return {
        "weights": weights.detach().numpy().copy(),
        "grad_mu": grad_mu.detach().numpy().copy(),
        "grad_theta": grad_theta,
        "prediction": mu.detach().numpy().copy(),
    }


def summary_tables(samples, pairs, batches, out):
    if not samples.empty:
        samples.to_csv(out / "solver_sample.csv", index=False)

    if not pairs.empty:
        pairs.to_csv(out / "pair_sensitivity.csv", index=False)

        grouped = pairs.groupby(
            ["predictor_source", "risk", "solver", "quantile"],
            dropna=False,
        )

        agg = grouped.agg(
            n=("grad_diff_l2", "size"),
            unique_dates=("sample_index", "nunique"),
            median_grad_diff=("grad_diff_l2", "median"),
            p95_grad_diff=(
                "grad_diff_l2", lambda z: z.quantile(.95),
            ),
            median_relative_grad_diff=("relative_grad_diff", "median"),
            fraction_grad_gt_threshold=("large_grad_change", "mean"),
            fraction_active_change_candidate=("active_change_candidate", "mean"),
            fraction_near_boundary=("near_boundary_either", "mean"),
        ).reset_index()

        agg.to_csv(
            out / "summary_by_quantile.csv", index=False,
        )

        by_seed = pairs.groupby(
            ["predictor_source", "seed", "risk", "solver"],
        ).agg(
            n=("grad_diff_l2", "size"),
            median_grad_diff=("grad_diff_l2", "median"),
            fraction_grad_gt_threshold=("large_grad_change", "mean"),
            fraction_active_change_candidate=("active_change_candidate", "mean"),
        ).reset_index()

        by_seed.to_csv(
            out / "summary_by_seed.csv", index=False,
        )

    if not batches.empty:
        batches.to_csv(
            out / "batch_parameter_grad.csv", index=False,
        )


def main():
    args = parse_args()

    if args.max_weight != 0.2:
        raise ValueError("This experiment fixes max_weight at 0.2")

    if min(args.batch_size, args.batches_per_quantile) <= 0:
        raise ValueError("Batch size/count must be positive")

    if args.perturbation <= 0:
        raise ValueError("Perturbation must be positive")

    if not 0 < args.bound_tol < args.boundary_warning < args.max_weight:
        raise ValueError(
            "Require bound-tol < boundary-warning < max-weight"
        )

    torch.set_num_threads(1)

    out = (
        Path(args.out_root)
        / args.predictor_source
        / args.label
    )

    if out.exists():
        raise FileExistsError(
            f"Output exists: {out}. Choose another --label"
        )

    out.mkdir(parents=True)

    (out / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

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
        raise RuntimeError(
            "Sample/Ridge tickers or target dates differ"
        )

    train, _, _ = chronological_split(
        sample, args.train_end, args.validation_end,
    )

    mean, std = fit_feature_standardizer(sample, train)

    x = torch.stack([
        sample[i]["features"].float()
        for i in range(len(sample))
    ])
    x = (x - mean.float()) / std.float()

    y = torch.stack([
        sample[i]["target"].float()
        for i in range(len(sample))
    ])

    sample_cov = (
        sample.covariances.detach().cpu().numpy()
    )
    ridge_cov = (
        ridge.covariances.detach().cpu().numpy()
    )

    train_ix = np.asarray(train.indices, dtype=int)
    eig = np.linalg.eigvalsh(sample_cov[train_ix])
    cond_train = eig[:, -1] / eig[:, 0]

    sample_condition = np.full(len(sample), np.nan)
    sample_condition[train_ix] = cond_train

    selected = select_stratified_batches(
        train_ix,
        sample_condition,
        args.batch_size,
        args.batches_per_quantile,
    )

    rng = np.random.default_rng(args.direction_seed)
    v = rng.normal(size=sample.n_assets)
    v /= np.linalg.norm(v)

    delta = (
        args.perturbation
        * np.linalg.norm(sample_cov, axis=(1, 2))[:, None, None]
        * np.outer(v, v)[None]
    )

    matrices = {
        "sample": sample_cov,
        "ridge": ridge_cov,
    }

    solver_configs = {
        "default": {},
        "tight": {
            "eps": args.tight_eps,
            "max_iters": args.tight_max_iters,
            "acceleration_lookback": 0,
        },
        "ultra": {
            "eps": args.ultra_eps,
            "max_iters": args.ultra_max_iters,
            "acceleration_lookback": 0,
        },
    }

    qp = IndependentQP(sample.n_assets, args.max_weight)

    sample_rows = []
    pair_rows = []
    batch_rows = []
    failures = []

    for seed in args.seeds:
        model = MLPWithMarkowitz(
            n_assets=sample.n_assets,
            lookback=args.lookback,
            hidden_dim=args.hidden_dim,
            dropout=0.0,
            risk_aversion=args.risk_aversion,
            max_weight=args.max_weight,
        ).float().cpu()

        model.return_model.load_state_dict(
            checkpoint_state(
                args.checkpoint_root,
                args.predictor_source,
                seed,
                args.risk_aversion,
            ),
            strict=True,
        )

        model.eval()

        for risk, all_cov in matrices.items():
            for quantile, batch_number, ix in selected:
                features, targets = x[ix], y[ix]
                data = {}
                independent = {}

                for change in ("base", "perturbed"):
                    cov = all_cov[ix] + (
                        delta[ix]
                        if change == "perturbed" else 0
                    )

                    factor = covariance_to_risk_factor(
                        torch.from_numpy(
                            np.ascontiguousarray(cov)
                        ),
                        args.risk_aversion,
                    )

                    with torch.no_grad():
                        mu_np = (
                            model.return_model(features)
                            .detach().numpy().astype(np.float64)
                        )

                    independent[change] = []

                    for j in range(len(ix)):
                        try:
                            independent[change].append(
                                qp.solve(
                                    mu_np[j],
                                    factor[j].numpy(),
                                    targets[j].double().numpy(),
                                    args.bound_tol,
                                    args.boundary_warning,
                                )
                            )
                        except Exception as exc:
                            independent[change].append(None)
                            failures.append({
                                "seed": seed,
                                "risk": risk,
                                "quantile": quantile,
                                "batch": batch_number,
                                "change": change,
                                "sample_index": int(ix[j]),
                                "solver": "CLARABEL",
                                "error": str(exc)[:250],
                            })

                    for solver, options in solver_configs.items():
                        try:
                            data[(change, solver)] = run_solver(
                                model, features, targets,
                                factor, options,
                                args.alpha, args.mse_scale,
                            )
                        except Exception as exc:
                            failures.append({
                                "seed": seed,
                                "risk": risk,
                                "quantile": quantile,
                                "batch": batch_number,
                                "change": change,
                                "sample_index": -1,
                                "solver": solver,
                                "error": str(exc)[:250],
                            })

                for change in ("base", "perturbed"):
                    ultra = data.get((change, "ultra"))

                    for solver in solver_configs:
                        result = data.get((change, solver))

                        if result is None:
                            continue

                        for j, idx in enumerate(ix):
                            ref = independent[change][j]
                            g = result["grad_mu"][j]

                            uw = (
                                ultra["grad_mu"][j]
                                if ultra is not None
                                else None
                            )

                            sample_rows.append({
                                "predictor_source": args.predictor_source,
                                "seed": seed,
                                "risk": risk,
                                "quantile": quantile,
                                "batch": batch_number,
                                "sample_index": int(idx),
                                "date": str(
                                    sample.target_dates[int(idx)].date()
                                ),
                                "sample_condition": float(
                                    sample_condition[int(idx)]
                                ),
                                "risk_condition": float(
                                    np.linalg.cond(all_cov[int(idx)])
                                ),
                                "change": change,
                                "solver": solver,
                                "grad_l2": float(np.linalg.norm(g)),
                                "grad_vs_ultra_l2": (
                                    float(np.linalg.norm(g - uw))
                                    if uw is not None else np.nan
                                ),
                                "solver_sensitive": (
                                    bool(
                                        np.linalg.norm(g - uw)
                                        > args.solver_threshold
                                    )
                                    if uw is not None else np.nan
                                ),
                                "weight_vs_clarabel_max": (
                                    float(np.max(np.abs(
                                        result["weights"][j]
                                        - ref["weights"]
                                    )))
                                    if ref else np.nan
                                ),
                                "grad_vs_kkt_candidate_l2": (
                                    float(np.linalg.norm(
                                        g - ref["grad_kkt_candidate"]
                                    ))
                                    if ref else np.nan
                                ),
                                "kkt_clear_interior": (
                                    ref["kkt_clear_interior"]
                                    if ref else False
                                ),
                                "n_free": (
                                    ref["n_free"] if ref else np.nan
                                ),
                                "near_boundary": (
                                    ref["near_boundary"]
                                    if ref else np.nan
                                ),
                                "min_free_slack": (
                                    ref["min_free_slack"]
                                    if ref else np.nan
                                ),
                                "min_active_dual": (
                                    ref["min_active_dual"]
                                    if ref else np.nan
                                ),
                                "kkt_stationarity": (
                                    ref["kkt_stationarity"]
                                    if ref else np.nan
                                ),
                            })

                for solver in solver_configs:
                    a = data.get(("base", solver))
                    b = data.get(("perturbed", solver))

                    if a is None or b is None:
                        continue

                    t0, t1 = a["grad_theta"], b["grad_theta"]

                    batch_rows.append({
                        "predictor_source": args.predictor_source,
                        "seed": seed,
                        "risk": risk,
                        "quantile": quantile,
                        "batch": batch_number,
                        "solver": solver,
                        "theta_grad_diff_l2": float(
                            np.linalg.norm(t1 - t0)
                        ),
                        "theta_grad_relative": float(
                            np.linalg.norm(t1 - t0)
                            / max(
                                np.linalg.norm(t0),
                                np.linalg.norm(t1),
                                1e-8,
                            )
                        ),
                        "prediction_diff_max": float(
                            np.abs(
                                a["prediction"] - b["prediction"]
                            ).max()
                        ),
                    })

                    for j, idx in enumerate(ix):
                        g0 = a["grad_mu"][j]
                        g1 = b["grad_mu"][j]

                        diff = float(
                            np.linalg.norm(g1 - g0)
                        )

                        ref0 = independent["base"][j]
                        ref1 = independent["perturbed"][j]

                        candidate_flip = (
                            bool(np.any(
                                ref0["state"] != ref1["state"]
                            ))
                            if ref0 and ref1 else np.nan
                        )

                        boundary_warning = (
                            bool(
                                ref0["near_boundary"]
                                or ref1["near_boundary"]
                            )
                            if ref0 and ref1 else np.nan
                        )

                        pair_rows.append({
                            "predictor_source": args.predictor_source,
                            "seed": seed,
                            "risk": risk,
                            "quantile": quantile,
                            "batch": batch_number,
                            "sample_index": int(idx),
                            "date": str(
                                sample.target_dates[int(idx)].date()
                            ),
                            "sample_condition": float(
                                sample_condition[int(idx)]
                            ),
                            "risk_condition": float(
                                np.linalg.cond(all_cov[int(idx)])
                            ),
                            "solver": solver,
                            "grad_diff_l2": diff,
                            "relative_grad_diff": (
                                diff / max(
                                    np.linalg.norm(g0),
                                    np.linalg.norm(g1),
                                    1e-8,
                                )
                            ),
                            "large_grad_change": bool(
                                diff > args.grad_threshold
                            ),
                            "weight_diff_max": float(
                                np.abs(
                                    a["weights"][j]
                                    - b["weights"][j]
                                ).max()
                            ),
                            "active_change_candidate": candidate_flip,
                            "near_boundary_either": boundary_warning,
                            "n_free_base": (
                                ref0["n_free"] if ref0 else np.nan
                            ),
                            "n_free_perturbed": (
                                ref1["n_free"] if ref1 else np.nan
                            ),
                        })

                print(
                    f"[{args.predictor_source} seed={seed}] "
                    f"{risk} {quantile} "
                    f"batch={batch_number + 1} done",
                    flush=True,
                )

        summary_tables(
            pd.DataFrame(sample_rows),
            pd.DataFrame(pair_rows),
            pd.DataFrame(batch_rows),
            out,
        )

        if failures:
            pd.DataFrame(failures).to_csv(
                out / "errors.csv", index=False,
            )

        print(
            f"[SEED {seed}] Saved: {out}",
            flush=True,
        )

    if pair_rows:
        summary = pd.read_csv(
            out / "summary_by_quantile.csv"
        )
        print(
            "\n[SUMMARY BY QUANTILE]\n"
            + summary.to_string(index=False)
        )

    if failures:
        print(
            f"[WARNING] {len(failures)} solver/QP failures "
            "logged in errors.csv"
        )

    print(f"\n[DONE] {out.resolve()}")


if __name__ == "__main__":
    main()
