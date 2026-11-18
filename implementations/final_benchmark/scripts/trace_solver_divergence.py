
"""A/A replay and dense step diagnostics for Solver x Covariance 2x2 training.

Run from project root. Does not modify previously trained checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset, chronological_split, fit_feature_standardizer,
)
from implementations.covariance_baselines.src.dataset import CovarianceBaselineDataset
from implementations.final_benchmark.scripts.run_solver_precision_2x2 import (
    ARM_NAMES, make_arms, forward_objective, oracle_costs, evaluate, flat,
)


def parse_args():
    p = argparse.ArgumentParser(description="2x2 DFL A/A replay and dense divergence trace")
    p.add_argument("--risk", choices=["sample", "ridge"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--focus-start", type=int, required=True)
    p.add_argument("--focus-end", type=int, required=True)
    p.add_argument("--label", default="aa_focus")
    p.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    p.add_argument("--out-dir", default="implementations/final_benchmark/outputs/diagnostics/step_replay")
    p.add_argument("--reference-root", default="implementations/final_benchmark/outputs/diagnostics/solver_2x2")
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
    p.add_argument("--boundary-tol", type=float, default=1e-5)
    p.add_argument("--save-every", type=int, default=25)
    return p.parse_args()


def maxdiff(a, b):
    return float(np.max(np.abs(a - b)))


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def train_step(arm, x, y, ix, args):
    model = arm["model"]
    model.train()
    arm["optimizer"].zero_grad(set_to_none=True)

    loss, pred, weights, costs, mse = forward_objective(
        model, x[ix], y[ix], arm["covariance"][ix], arm["solver_args"], args,
    )
    loss.backward()

    result = {
        "gradient": flat(model, grads=True),
        "prediction_pre": pred.detach().cpu().numpy().copy(),
        "weights": weights.detach().cpu().numpy().copy(),
        "loss": float(loss.detach().item()),
        "cost": float(costs.detach().mean().item()),
        "mse": float(mse.detach().item()),
    }

    arm["optimizer"].step()
    result["parameters_post"] = flat(model)

    with torch.no_grad():
        result["prediction_post"] = (
            model.return_model(x[ix]).detach().cpu().numpy().copy()
        )

    return result


def aa_record(step, epoch, arm_name, first, second):
    record = {"step": step, "epoch": epoch, "arm": arm_name}

    for name in (
        "gradient", "parameters_post", "prediction_pre",
        "prediction_post", "weights",
    ):
        a, b = first[name], second[name]
        record[f"aa_{name}_max_abs"] = maxdiff(a, b)
        record[f"aa_{name}_l2"] = float(np.linalg.norm(a - b))
        record[f"aa_{name}_exact"] = bool(np.array_equal(a, b))

    return record


def contrast_record(step, epoch, name, first, second, max_weight, tol):
    ga, gb = first["gradient"], second["gradient"]
    wa, wb = first["weights"], second["weights"]

    lower_a, lower_b = wa <= tol, wb <= tol
    upper_a = wa >= max_weight - tol
    upper_b = wb >= max_weight - tol

    return {
        "step": step,
        "epoch": epoch,
        "contrast": name,
        "gradient_diff_l2": float(np.linalg.norm(ga - gb)),
        "gradient_cosine": cosine(ga, gb),
        "gradient_ref_l2": float(np.linalg.norm(ga)),
        "gradient_other_l2": float(np.linalg.norm(gb)),
        "gradient_relative": float(
            np.linalg.norm(ga - gb) / max(np.linalg.norm(ga), 1e-30)
        ),
        "pre_step_prediction_max_abs_diff": maxdiff(
            first["prediction_pre"], second["prediction_pre"]
        ),
        "post_step_prediction_max_abs_diff": maxdiff(
            first["prediction_post"], second["prediction_post"]
        ),
        "post_step_parameter_diff_l2": float(
            np.linalg.norm(first["parameters_post"] - second["parameters_post"])
        ),
        "weight_max_abs_diff": maxdiff(wa, wb),
        "bound_lower_ref": int(lower_a.sum()),
        "bound_upper_ref": int(upper_a.sum()),
        "bound_lower_other": int(lower_b.sum()),
        "bound_upper_other": int(upper_b.sum()),
        "bound_lower_changes": int(np.count_nonzero(lower_a != lower_b)),
        "bound_upper_changes": int(np.count_nonzero(upper_a != upper_b)),
        "loss_ref": first["loss"],
        "loss_other": second["loss"],
    }


def compare_historical(new_trace, args, out):
    old_path = (
        Path(args.reference_root)
        / args.risk
        / f"seed_{args.seed}"
        / "full"
        / "step_trace.csv"
    )

    if not old_path.exists():
        print(f"[REFERENCE] Not found: {old_path}")
        return

    old = pd.read_csv(old_path)
    keys = ["step", "contrast"]

    columns = [
        "gradient_diff_l2",
        "pre_step_prediction_max_abs_diff",
        "post_step_parameter_diff_l2",
    ]

    joint = pd.merge(
        new_trace,
        old[keys + columns],
        on=keys,
        suffixes=("_new", "_old"),
    )

    if joint.empty:
        print("[REFERENCE] No overlapping comparison rows")
        return

    for column in columns:
        joint[f"{column}_abs_error"] = np.abs(
            joint[f"{column}_new"] - joint[f"{column}_old"]
        )
        joint[f"{column}_close"] = np.isclose(
            joint[f"{column}_new"],
            joint[f"{column}_old"],
            atol=1e-6,
            rtol=1e-4,
        )

    joint.to_csv(
        out / "historical_replay_comparison.csv",
        index=False,
    )

    print(f"[REFERENCE] Compared {len(joint)} rows with old step_trace.csv")

    for column in columns:
        print(
            f"  {column}: "
            f"max abs error={joint[f'{column}_abs_error'].max():.4e}, "
            f"close={int(joint[f'{column}_close'].sum())}/{len(joint)}"
        )


def main():
    args = parse_args()

    if not (1 <= args.focus_start <= args.focus_end <= args.steps):
        raise ValueError("Require 1 <= focus-start <= focus-end <= steps")

    if (
        args.steps < 1
        or args.batch_size < 1
        or args.save_every < 1
        or args.perturbation <= 0
    ):
        raise ValueError("Invalid positive argument")

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
        / args.label
    )

    if out.exists():
        raise FileExistsError(
            f"Output exists: {out}. "
            "Use a different --label to preserve data."
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
        raise RuntimeError("Sample/Ridge target alignment failed")

    train, val, _ = chronological_split(
        sample,
        args.train_end,
        args.validation_end,
    )

    train_ix = np.asarray(train.indices, dtype=int)
    val_ix = np.asarray(val.indices, dtype=int)

    n_per_epoch = math.ceil(len(train_ix) / args.batch_size)

    if args.steps > n_per_epoch * 18:
        raise ValueError(
            "This diagnostic is for the early training phase (< 19 epochs)."
        )

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
    v = rng.normal(size=sample.n_assets)
    v /= np.linalg.norm(v)

    delta = (
        args.perturbation
        * np.linalg.norm(sample_cov, axis=(1, 2))[:, None, None]
        * np.outer(v, v)[None]
    )

    covariances = {
        "base": torch.from_numpy(
            np.ascontiguousarray(base)
        ),
        "perturbed": torch.from_numpy(
            np.ascontiguousarray(base + delta)
        ),
    }

    # Two independent copies with identical initial model parameters.
    replicas = [
        make_arms(args, sample.n_assets, covariances)
        for _ in range(2)
    ]

    generator = torch.Generator().manual_seed(args.seed)

    # Mirror the original validation/oracle solver calls.
    for rep in replicas:
        for name in ARM_NAMES:
            arm = rep[name]
            arm["val_oracle"] = oracle_costs(
                arm["model"],
                y,
                arm["covariance"],
                val_ix,
                arm["solver_args"],
                args,
            )

    aa_rows, contrast_rows = [], []

    pairs = [
        ("default_base", "default_perturbed", "perturb_default"),
        ("tight_base", "tight_perturbed", "perturb_tight"),
        ("default_base", "tight_base", "solver_only_base"),
        (
            "default_perturbed",
            "tight_perturbed",
            "solver_only_perturbed",
        ),
    ]

    step = 0

    for epoch in range(1, math.ceil(args.steps / n_per_epoch) + 1):
        order = train_ix[
            torch.randperm(
                len(train_ix),
                generator=generator,
            ).numpy()
        ]

        for start in range(0, len(order), args.batch_size):
            if step >= args.steps:
                break

            step += 1
            ix = order[start:start + args.batch_size]
            outputs = []

            for rep in replicas:
                d = {}

                for name in ARM_NAMES:
                    d[name] = train_step(
                        rep[name], x, y, ix, args,
                    )

                outputs.append(d)

            for name in ARM_NAMES:
                aa_rows.append(
                    aa_record(
                        step,
                        epoch,
                        name,
                        outputs[0][name],
                        outputs[1][name],
                    )
                )

            if (
                step == 1
                or step % 25 == 0
                or args.focus_start <= step <= args.focus_end
            ):
                for first, second, contrast in pairs:
                    contrast_rows.append(
                        contrast_record(
                            step,
                            epoch,
                            contrast,
                            outputs[0][first],
                            outputs[0][second],
                            max_weight=0.2,
                            tol=args.boundary_tol,
                        )
                    )

            if (
                step == 1
                or step % 25 == 0
                or step in (args.focus_start, args.focus_end)
            ):
                t = next(
                    r for r in contrast_rows[::-1]
                    if r["step"] == step
                    and r["contrast"] == "perturb_tight"
                )

                current_aa = aa_rows[-4:]

                print(
                    f"Step {step:04d} | "
                    f"tight Δgrad={t['gradient_diff_l2']:.3e} "
                    f"Δpred={t['pre_step_prediction_max_abs_diff']:.3e} "
                    f"Δparam={t['post_step_parameter_diff_l2']:.3e} "
                    f"A/A grad max="
                    f"{max(r['aa_gradient_max_abs'] for r in current_aa):.2e}",
                    flush=True,
                )

            if step % args.save_every == 0:
                pd.DataFrame(aa_rows).to_csv(
                    out / "aa_per_step.csv",
                    index=False,
                )
                pd.DataFrame(contrast_rows).to_csv(
                    out / "divergence_trace.csv",
                    index=False,
                )

        # Match original validation evaluation at completed epoch boundaries.
        if step % n_per_epoch == 0:
            for rep in replicas:
                for name in ARM_NAMES:
                    arm = rep[name]

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

    aa = pd.DataFrame(aa_rows)
    traced = pd.DataFrame(contrast_rows)

    aa.to_csv(
        out / "aa_per_step.csv",
        index=False,
    )

    traced.to_csv(
        out / "divergence_trace.csv",
        index=False,
    )

    aa_summary = (
        aa.groupby("arm")[[
            "aa_gradient_max_abs",
            "aa_parameters_post_max_abs",
            "aa_prediction_pre_max_abs",
            "aa_weights_max_abs",
        ]]
        .max()
        .reset_index()
    )

    aa_summary.to_csv(
        out / "aa_summary.csv",
        index=False,
    )

    compare_historical(traced, args, out)

    print(
        "\nA/A max errors:\n"
        + aa_summary.to_string(index=False)
    )

    print(f"\n[DONE] {out.resolve()}")

    print(
        "Note: bound counts are tolerance-based proxies, "
        "not KKT dual active-set certificates."
    )


if __name__ == "__main__":
    main()
