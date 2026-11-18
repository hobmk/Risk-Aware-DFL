
"""Controlled local fine-tuning of anomalous DFL gradients.

This is a diagnostic experiment, not a full training reproduction.
AdamW states start fresh and one selected training batch is repeated.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

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
    p = argparse.ArgumentParser(description="Controlled DFL training divergence")
    p.add_argument("--case", default="dfl_mvo:43:ridge:0.90")
    p.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    p.add_argument("--checkpoint-root", default="implementations/final_benchmark/outputs")
    p.add_argument("--out-dir", default="implementations/final_benchmark/outputs/diagnostics/training_divergence")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--risk-aversion", type=float, default=1.0)
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mse-scale", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--direction-seed", type=int, default=20261110)
    p.add_argument("--tight-eps", type=float, default=1e-8)
    p.add_argument("--tight-max-iters", type=int, default=10000)
    return p.parse_args()


def vector(model, gradient=False):
    arrays = [
        (p.grad if gradient else p.detach()).reshape(-1)
        for p in model.return_model.parameters()
        if not gradient or p.grad is not None
    ]
    return torch.cat(arrays).detach().cpu().numpy().copy()


def compare(a, b):
    diff = float(np.linalg.norm(a - b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = (
        float(np.dot(a, b) / (na * nb))
        if na > 0 and nb > 0 else float("nan")
    )
    return diff, diff / max(na, 1e-30), cosine


def make_model(args, n_assets, checkpoint):
    model = MLPWithMarkowitz(
        n_assets=n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        dropout=0.0,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
    ).float().cpu()

    model.return_model.load_state_dict(
        copy.deepcopy(checkpoint), strict=True
    )
    model.train()
    return model


def train_step(model, optimizer, x, y, covariance, solver_args, args):
    optimizer.zero_grad(set_to_none=True)

    prediction = model.return_model(x)
    factor = covariance_to_risk_factor(
        covariance, args.risk_aversion, jitter=0.0
    )

    (weights,) = model.markowitz_layer(
        prediction.to(torch.float64),
        factor,
        solver_args=solver_args,
    )

    predicted_cost = markowitz_cost(
        weights, y.to(torch.float64), factor
    ).mean()
    mse = F.mse_loss(prediction, y)

    # Regret = predicted_cost - oracle_cost.
    # Oracle cost is independent of Predictor parameters.
    # Omitting it preserves the parameter gradient.
    objective = (
        args.alpha * predicted_cost
        + (1 - args.alpha) * args.mse_scale * mse.to(torch.float64)
    )
    objective.backward()

    gradient = vector(model, gradient=True)

    result = {
        "gradient": gradient,
        "weights": weights.detach().cpu().numpy().copy(),
        "train_prediction": prediction.detach().cpu().numpy().copy(),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "predicted_cost": float(predicted_cost.detach()),
        "mse": float(mse.detach()),
    }

    optimizer.step()
    return result


def nearest_batch(indices, cov, quantile, batch_size):
    eigenvalues = np.linalg.eigvalsh(cov[indices])
    conditions = eigenvalues[:, -1] / eigenvalues[:, 0]

    pos = int(np.argmin(
        np.abs(conditions - np.quantile(conditions, quantile))
    ))
    start = int(np.clip(
        pos - batch_size // 2, 0, len(indices) - batch_size
    ))
    return indices[start:start + batch_size]


def main():
    args = parse_args()
    predictor, seed_text, risk, quantile_text = args.case.split(":")
    seed, quantile = int(seed_text), float(quantile_text)

    if predictor not in {"dfl_mvo", "ridge"} or risk not in {"sample", "ridge"}:
        raise ValueError("Invalid --case. Example: dfl_mvo:43:ridge:0.90")
    if args.steps < 1 or args.batch_size < 1 or args.eps <= 0:
        raise ValueError("steps, batch-size, eps must be positive")
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between 0 and 1")

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

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
        raise RuntimeError("Sample/Ridge data alignment failed")

    train, val, _ = chronological_split(
        sample, args.train_end, args.validation_end
    )
    indices = np.asarray(train.indices, dtype=int)

    if len(indices) < args.batch_size or len(val) < args.batch_size:
        raise RuntimeError("Insufficient training/validation samples")

    mean, std = fit_feature_standardizer(sample, train)

    sample_cov = sample.covariances.detach().cpu().numpy()
    ridge_cov = ridge.covariances.detach().cpu().numpy()

    ix = nearest_batch(
        indices, sample_cov, quantile, args.batch_size
    )

    x = torch.stack([
        sample[int(i)]["features"] for i in ix
    ])
    x = ((x - mean) / std).float()

    y = torch.stack([
        sample[int(i)]["target"] for i in ix
    ]).float()

    # Fixed validation inputs are used only to monitor prediction drift.
    probe_ix = np.asarray(val.indices[:args.batch_size], dtype=int)
    probe_x = torch.stack([
        sample[int(i)]["features"] for i in probe_ix
    ])
    probe_x = ((probe_x - mean) / std).float()

    # Identical PSD perturbation direction for all conditions.
    v = np.random.default_rng(
        args.direction_seed
    ).normal(size=sample.n_assets)
    v /= np.linalg.norm(v)

    delta = (
        args.eps
        * np.linalg.norm(sample_cov[ix], axis=(1, 2))[:, None, None]
        * np.outer(v, v)[None]
    )

    base = (
        sample_cov[ix]
        if risk == "sample"
        else ridge_cov[ix]
    )

    risk_matrices = {
        "base": torch.tensor(
            np.array(base, copy=True), dtype=torch.float64
        ),
        "perturbed": torch.tensor(
            base + delta, dtype=torch.float64
        ),
    }

    checkpoint_path = (
        Path(args.checkpoint_root)
        / predictor
        / f"lambda_{args.risk_aversion:g}"
        / f"seed_{seed}"
        / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    state = checkpoint["model_state_dict"]

    if any(k.startswith("return_model.") for k in state):
        state = {
            k.removeprefix("return_model."): value
            for k, value in state.items()
            if k.startswith("return_model.")
        }

    solver_configs = {
        "default": {},
        "tight": {
            "eps": args.tight_eps,
            "max_iters": args.tight_max_iters,
            "acceleration_lookback": 0,
        },
    }

    # Four identical Predictor initializations, separate fresh AdamW states.
    torch.manual_seed(seed)
    models, optimizers = {}, {}

    for solver in solver_configs:
        for cov_name in risk_matrices:
            key = (solver, cov_name)

            models[key] = make_model(
                args, sample.n_assets, state
            )
            optimizers[key] = torch.optim.AdamW(
                models[key].return_model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

    rows = []

    print(
        f"CASE: {args.case} | "
        f"Batch: {sample.target_dates[ix[0]].date()} ~ "
        f"{sample.target_dates[ix[-1]].date()} | "
        f"Steps: {args.steps}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        outputs = {}

        for solver, options in solver_configs.items():
            for cov_name, cov in risk_matrices.items():
                key = (solver, cov_name)

                outputs[key] = train_step(
                    model=models[key],
                    optimizer=optimizers[key],
                    x=x,
                    y=y,
                    covariance=cov,
                    solver_args=options,
                    args=args,
                )

        for solver in solver_configs:
            base_out = outputs[(solver, "base")]
            pert_out = outputs[(solver, "perturbed")]

            grad_abs, grad_rel, grad_cos = compare(
                base_out["gradient"],
                pert_out["gradient"],
            )

            param_abs, _, _ = compare(
                vector(models[(solver, "base")]),
                vector(models[(solver, "perturbed")]),
            )

            with torch.no_grad():
                probe_base = (
                    models[(solver, "base")]
                    .return_model(probe_x)
                )
                probe_pert = (
                    models[(solver, "perturbed")]
                    .return_model(probe_x)
                )

            rows.append({
                "step": step,
                "case": args.case,
                "solver": solver,
                "gradient_diff_l2": grad_abs,
                "gradient_diff_relative": grad_rel,
                "gradient_cosine": grad_cos,
                "gradient_base_norm": base_out["gradient_norm"],
                "gradient_pert_norm": pert_out["gradient_norm"],
                "weight_max_abs_diff": float(np.max(np.abs(
                    base_out["weights"] - pert_out["weights"]
                ))),
                "train_prediction_max_abs_diff_pre_step": float(
                    np.max(np.abs(
                        base_out["train_prediction"]
                        - pert_out["train_prediction"]
                    ))
                ),
                "parameter_diff_l2_post_step": param_abs,
                "probe_prediction_max_abs_diff_post_step": float(
                    (probe_base - probe_pert).abs().max()
                ),
                "probe_prediction_rmse_post_step": float(
                    torch.sqrt(
                        ((probe_base - probe_pert) ** 2).mean()
                    )
                ),
                "mse_base": base_out["mse"],
                "mse_pert": pert_out["mse"],
            })

        if step == 1 or step % 5 == 0 or step == args.steps:
            default_row, tight_row = rows[-2], rows[-1]

            print(
                f"Step {step:03d} | "
                f"default: Δgrad="
                f"{default_row['gradient_diff_l2']:.3e}, "
                f"Δpred="
                f"{default_row['probe_prediction_max_abs_diff_post_step']:.3e} | "
                f"tight: Δgrad="
                f"{tight_row['gradient_diff_l2']:.3e}, "
                f"Δpred="
                f"{tight_row['probe_prediction_max_abs_diff_post_step']:.3e}",
                flush=True,
            )

    output = (
        Path(args.out_dir)
        / args.case.replace(":", "_")
    )
    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        output / "step_trace.csv", index=False
    )

    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    print(f"\n[DONE] {output.resolve()}")


if __name__ == "__main__":
    main()
