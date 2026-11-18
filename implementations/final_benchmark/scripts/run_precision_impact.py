
"""Paired full-training comparison of default and high-precision SCS.

Run as: python -m implementations.final_benchmark.scripts.run_precision_impact ...
Training uses realized Markowitz cost rather than regret; subtracting the oracle
cost changes loss values but not training gradients or within-arm best epochs.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset, chronological_split, fit_feature_standardizer,
)
from implementations.covariance_baselines.src.dataset import CovarianceBaselineDataset
from implementations.dfl_mvo_lee2025.src.losses import markowitz_cost
from implementations.dfl_mvo_lee2025.src.metrics import calculate_portfolio_metrics
from implementations.dfl_mvo_lee2025.src.model import ReturnMLP
from implementations.dfl_mvo_lee2025.src.optimization import (
    build_markowitz_layer, covariance_to_risk_factor,
)


def arguments():
    p = argparse.ArgumentParser(description="Paired default vs precise-SCS full training")
    p.add_argument("--risk", choices=["sample", "ridge"], required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[43])
    p.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    p.add_argument("--out-root", default="implementations/final_benchmark/outputs/diagnostics/precision_impact")
    p.add_argument("--label", default="ultra_1e10")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mse-scale", type=float, default=30.0)
    p.add_argument("--lambda-risk", type=float, default=1.0)
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--ridge-shrinkage", type=float, default=0.1)
    p.add_argument("--perturbation", type=float, default=1e-6)
    p.add_argument("--direction-seed", type=int, default=20261110)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--val-end", default="2022-12-31")
    p.add_argument("--eps", type=float, default=1e-10)
    p.add_argument("--max-iters", type=int, default=100000)
    p.add_argument("--pair-grad-threshold", type=float, default=0.01)
    p.add_argument("--spike-ratio", type=float, default=5.0)
    p.add_argument("--spike-min", type=float, default=0.01)
    p.add_argument("--spike-window", type=int, default=20)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def norm(parameters):
    return float(sum(p.grad.detach().square().sum().item() for p in parameters if p.grad is not None) ** 0.5)


def flat_grad(parameters):
    return np.concatenate([
        (p.grad.detach().cpu().numpy().reshape(-1) if p.grad is not None
         else np.zeros(p.numel(), dtype=np.float32)) for p in parameters
    ])


def state(weights, cap, tol=1e-6):
    return torch.where(weights <= tol, 0, torch.where(weights >= cap - tol, 2, 1))


def run_forward(net, layer, features, target, covariance, args, solver, train=False):
    mu = net(features)
    r = covariance_to_risk_factor(covariance, args.lambda_risk)
    (w,) = layer(mu.to(device="cpu", dtype=torch.float64), r, solver_args=solver)
    cost = markowitz_cost(w, target.to(device="cpu", dtype=torch.float64), r).mean()
    mse = (mu - target).square().mean()
    objective = args.alpha * cost + (1 - args.alpha) * args.mse_scale * mse.double()

    if not train:
        return {"mu": mu, "w": w, "cost": cost, "mse": mse, "objective": objective}

    # Oracle cost is independent of the predictor.
    # Therefore, regret and realized Markowitz cost have the same gradient.
    (gc,) = torch.autograd.grad(cost, mu, retain_graph=True)
    gm = 2 * (mu.detach() - target) / mu.numel()
    gtotal = args.alpha * gc + (1 - args.alpha) * args.mse_scale * gm

    params = tuple(net.parameters())
    g_regret = torch.autograd.grad(
        mu, params, grad_outputs=args.alpha * gc, retain_graph=True
    )
    regret_norm = float(sum(g.square().sum().item() for g in g_regret) ** 0.5)

    mu.backward(gtotal)

    return {
        "mu": mu.detach(),
        "w": w.detach(),
        "cost": float(cost.detach()),
        "mse": float(mse.detach()),
        "objective": float(objective.detach()),
        "regret_grad_l2": regret_norm,
        "mu_cost_grad_l2": float(gc.detach().norm()),
        "grad_l2": norm(params),
        "flat_grad": flat_grad(params),
    }


def evaluate(net, layer, x, y, cov, indices, args, solver, device,
             chunk=32, save_daily=False, dates=None):
    net.eval()
    returns, costs, mses, dates_out, weights_out, mu_out = [], [], [], [], [], []

    with torch.no_grad():
        for begin in range(0, len(indices), chunk):
            ix = indices[begin:begin + chunk]
            out = run_forward(
                net, layer, x[ix].to(device), y[ix].to(device),
                cov[ix], args, solver
            )

            w = out["w"].detach().cpu()
            mu = out["mu"].detach().cpu()

            returns.extend((w * y[ix].double()).sum(dim=1).numpy().tolist())
            costs.append(float(out["cost"]) * len(ix))
            mses.append(float(out["mse"]) * len(ix))

            if save_daily:
                dates_out.extend([str(dates[k].date()) for k in ix])
                weights_out.append(w.numpy())
                mu_out.append(mu.numpy())

    metrics, _, _ = calculate_portfolio_metrics(
        torch.tensor(returns, dtype=torch.float64)
    )

    result = dict(metrics)
    result.update({
        "test_cost": sum(costs) / len(indices),
        "test_mse": sum(mses) / len(indices),
    })

    if save_daily:
        return (
            result,
            pd.DataFrame({"date": dates_out, "portfolio_return": returns}),
            np.concatenate(weights_out),
            np.concatenate(mu_out),
        )

    return result


def build_data(args):
    sample = RollingMVODataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float64,
        risk_cache_dir="implementations/dfl_mvo_lee2025/cache/risk_matrix",
    )

    ridge = CovarianceBaselineDataset(
        price_csv=args.price_csv,
        method="ridge",
        ridge_shrinkage=args.ridge_shrinkage,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float64,
    )

    if sample.tickers != ridge.tickers or not sample.target_dates.equals(ridge.target_dates):
        raise RuntimeError("Sample and Ridge datasets do not align")

    train, val, test = chronological_split(
        sample, args.train_end, args.val_end
    )

    mean, std = fit_feature_standardizer(sample, train)
    positions = sample.target_positions

    features = np.stack([
        sample.returns[p - args.lookback:p] for p in positions
    ])
    targets = sample.returns[positions]

    x = (
        (torch.tensor(features, dtype=torch.float32) - mean.float())
        / std.float()
    ).contiguous()

    y = torch.tensor(targets, dtype=torch.float32)
    sample_cov = sample.covariances.double()

    base_cov = (
        sample_cov if args.risk == "sample"
        else ridge.covariances.double()
    )

    rng = np.random.default_rng(args.direction_seed)
    v = rng.normal(size=sample.n_assets)
    v /= np.linalg.norm(v)

    scale = torch.linalg.matrix_norm(
        sample_cov, ord="fro", dim=(-2, -1)
    )

    delta = (
        args.perturbation
        * scale[:, None, None]
        * torch.from_numpy(np.outer(v, v))
    )

    return (
        sample, x, y, base_cov, base_cov + delta,
        np.asarray(train.indices),
        np.asarray(val.indices),
        np.asarray(test.indices),
    )


def one_seed(args, seed, dataset):
    sample, x, y, base_cov, pert_cov, train_ix, val_ix, test_ix = dataset

    out_dir = (
        Path(args.out_root)
        / args.risk
        / f"seed_{seed}"
        / args.label
    )

    if out_dir.exists():
        raise FileExistsError(f"Existing output directory: {out_dir}")

    out_dir.mkdir(parents=True)

    (out_dir / "config.json").write_text(
        json.dumps({**vars(args), "seed": seed}, indent=2),
        encoding="utf-8",
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    template = ReturnMLP(
        sample.n_assets,
        args.lookback,
        args.hidden_dim,
        dropout=0.0,
    ).float()

    init_state = copy.deepcopy(template.state_dict())
    layer = build_markowitz_layer(sample.n_assets, args.max_weight)

    solvers = {
        "default": {},
        "precise": {
            "eps": args.eps,
            "max_iters": args.max_iters,
            "acceleration_lookback": 0,
        },
    }

    covs = {
        "base": base_cov,
        "perturbed": pert_cov,
    }

    arms = {}

    for solver in solvers:
        for change in covs:
            name = f"{solver}_{change}"

            net = ReturnMLP(
                sample.n_assets,
                args.lookback,
                args.hidden_dim,
                dropout=0.0,
            ).float().to(device)

            net.load_state_dict(init_state)

            arms[name] = {
                "net": net,
                "optim": torch.optim.AdamW(
                    net.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                ),
                "solver": solvers[solver],
                "cov": covs[change],
                "best": np.inf,
                "best_epoch": 0,
                "bad_epochs": 0,
                "finished": False,
                "history": deque(maxlen=args.spike_window),
                "best_state": None,
            }

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        torch.tensor(train_ix, dtype=torch.long),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    step_rows, pair_rows, epoch_rows = [], [], []
    steps_per_epoch = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        for batch_no, indices in enumerate(train_loader, 1):
            ix = indices.numpy()
            outputs = {}

            for name, arm in arms.items():
                if arm["finished"]:
                    continue

                arm["net"].train()
                arm["optim"].zero_grad(set_to_none=True)

                result = run_forward(
                    arm["net"],
                    layer,
                    x[ix].to(device),
                    y[ix].to(device),
                    arm["cov"][ix],
                    args,
                    arm["solver"],
                    train=True,
                )

                past = (
                    np.median(arm["history"])
                    if len(arm["history"]) >= 5 else np.nan
                )

                ratio = (
                    result["grad_l2"] / max(float(past), 1e-8)
                    if np.isfinite(past) else np.nan
                )

                spike = bool(
                    np.isfinite(ratio)
                    and ratio >= args.spike_ratio
                    and result["grad_l2"] >= args.spike_min
                )

                arm["history"].append(result["grad_l2"])
                outputs[name] = result

                step_rows.append({
                    "seed": seed,
                    "risk": args.risk,
                    "arm": name,
                    "epoch": epoch,
                    "batch": batch_no,
                    "step": (epoch - 1) * steps_per_epoch + batch_no,
                    "grad_l2": result["grad_l2"],
                    "regret_grad_l2": result["regret_grad_l2"],
                    "mu_cost_grad_l2": result["mu_cost_grad_l2"],
                    "grad_ratio": ratio,
                    "grad_spike": spike,
                    "train_cost": result["cost"],
                    "train_mse": result["mse"],
                    "train_objective_without_oracle": result["objective"],
                })

            for solver in solvers:
                a = outputs.get(f"{solver}_base")
                b = outputs.get(f"{solver}_perturbed")

                if a is None or b is None:
                    continue

                grad_diff = float(
                    np.linalg.norm(
                        a["flat_grad"] - b["flat_grad"]
                    )
                )

                pair_rows.append({
                    "seed": seed,
                    "risk": args.risk,
                    "solver": solver,
                    "epoch": epoch,
                    "batch": batch_no,
                    "step": (epoch - 1) * steps_per_epoch + batch_no,
                    "grad_diff_l2": grad_diff,
                    "grad_diff_over_threshold": bool(
                        grad_diff > args.pair_grad_threshold
                    ),
                    "prediction_diff_max": float(
                        (a["mu"] - b["mu"]).abs().max()
                    ),
                    "weight_diff_max": float(
                        (a["w"] - b["w"]).abs().max()
                    ),
                    "state_change_candidate": bool(torch.any(
                        state(a["w"], args.max_weight)
                        != state(b["w"], args.max_weight)
                    )),
                })

            # All four gradients are computed before applying any update.
            for name in outputs:
                arms[name]["optim"].step()

        for name, arm in arms.items():
            if arm["finished"]:
                continue

            stats = evaluate(
                arm["net"],
                layer,
                x,
                y,
                arm["cov"],
                val_ix,
                args,
                arm["solver"],
                device,
            )

            val_loss = (
                args.alpha * stats["test_cost"]
                + (1 - args.alpha)
                * args.mse_scale * stats["test_mse"]
            )

            epoch_rows.append({
                "seed": seed,
                "risk": args.risk,
                "arm": name,
                "epoch": epoch,
                "val_objective_without_oracle": val_loss,
            })

            if val_loss < arm["best"] - 1e-12:
                arm["best"] = val_loss
                arm["best_epoch"] = epoch
                arm["best_state"] = copy.deepcopy(
                    arm["net"].state_dict()
                )
                arm["bad_epochs"] = 0

                torch.save(
                    arm["best_state"],
                    out_dir / f"{name}_best.pt",
                )

            else:
                arm["bad_epochs"] += 1

                if arm["bad_epochs"] >= args.patience:
                    arm["finished"] = True

        pd.DataFrame(step_rows).to_csv(
            out_dir / "training_steps.csv", index=False
        )
        pd.DataFrame(pair_rows).to_csv(
            out_dir / "paired_steps.csv", index=False
        )
        pd.DataFrame(epoch_rows).to_csv(
            out_dir / "validation_history.csv", index=False
        )

        print(
            f"[{args.risk} seed={seed}] epoch={epoch} "
            + " | ".join(
                f"{k}:best{v['best_epoch']},stop{int(v['finished'])}"
                for k, v in arms.items()
            ),
            flush=True,
        )

        if all(arm["finished"] for arm in arms.values()):
            break

    test_rows, daily_rows, spike_rows = [], [], []
    steps = pd.DataFrame(step_rows)

    for name, arm in arms.items():
        arm["net"].load_state_dict(arm["best_state"])

        subset = steps[steps.arm == name]

        spike_rows.append({
            "seed": seed,
            "risk": args.risk,
            "arm": name,
            "num_steps": len(subset),
            "spike_count": int(subset.grad_spike.sum()),
            "spike_rate": float(subset.grad_spike.mean()),
            "max_grad_l2": float(subset.grad_l2.max()),
            "p95_grad_l2": float(subset.grad_l2.quantile(.95)),
            "best_epoch": arm["best_epoch"],
        })

        for evaluation, solver_args, cov in (
            ("common_precise_base", solvers["precise"], base_cov),
            ("native", arm["solver"], arm["cov"]),
        ):
            stats, daily, weights, mu = evaluate(
                arm["net"],
                layer,
                x,
                y,
                cov,
                test_ix,
                args,
                solver_args,
                device,
                save_daily=True,
                dates=sample.target_dates,
            )

            test_rows.append({
                "seed": seed,
                "risk": args.risk,
                "arm": name,
                "evaluation": evaluation,
                "best_epoch": arm["best_epoch"],
                **stats,
            })

            daily.insert(0, "evaluation", evaluation)
            daily.insert(0, "arm", name)
            daily.insert(0, "seed", seed)
            daily_rows.append(daily)

    pd.DataFrame(test_rows).to_csv(
        out_dir / "test_metrics.csv", index=False
    )

    pair_df = pd.DataFrame(pair_rows)

    if not pair_df.empty:
        paired_summary = pair_df.groupby("solver").agg(
            matched_steps=("step", "size"),
            count_grad_diff_over_threshold=(
                "grad_diff_over_threshold", "sum"
            ),
            fraction_grad_diff_over_threshold=(
                "grad_diff_over_threshold", "mean"
            ),
            median_grad_diff=("grad_diff_l2", "median"),
            p95_grad_diff=(
                "grad_diff_l2", lambda z: z.quantile(.95)
            ),
            max_grad_diff=("grad_diff_l2", "max"),
            median_prediction_diff_max=(
                "prediction_diff_max", "median"
            ),
            max_prediction_diff_max=(
                "prediction_diff_max", "max"
            ),
        ).reset_index()

        paired_summary.to_csv(
            out_dir / "paired_gradient_summary.csv", index=False
        )

    test_df = pd.DataFrame(test_rows)
    common = test_df[
        test_df.evaluation == "common_precise_base"
    ]

    diffs = []

    for solver in solvers:
        a = common[
            common.arm == f"{solver}_base"
        ].iloc[0]

        b = common[
            common.arm == f"{solver}_perturbed"
        ].iloc[0]

        diffs.append({
            "solver": solver,
            "delta_sharpe_pert_minus_base": float(
                b.sharpe_ratio - a.sharpe_ratio
            ),
            "delta_cagr_pert_minus_base": float(
                b.annualized_return_cagr
                - a.annualized_return_cagr
            ),
            "abs_delta_sharpe": float(
                abs(b.sharpe_ratio - a.sharpe_ratio)
            ),
            "abs_delta_cagr": float(
                abs(b.annualized_return_cagr - a.annualized_return_cagr)
            ),
            "delta_best_epoch": int(
                b.best_epoch - a.best_epoch
            ),
        })

    pd.DataFrame(diffs).to_csv(
        out_dir / "performance_sensitivity.csv", index=False
    )

    pd.concat(daily_rows, ignore_index=True).to_csv(
        out_dir / "test_daily.csv", index=False
    )

    pd.DataFrame(spike_rows).to_csv(
        out_dir / "gradient_spikes.csv", index=False
    )

    print("\n[RESULTS: COMMON PRECISE TEST]")

    print(
        pd.DataFrame(test_rows).query(
            "evaluation == 'common_precise_base'"
        )[
            [
                "arm",
                "best_epoch",
                "annualized_return_cagr",
                "sharpe_ratio",
                "annualized_volatility",
                "maximum_drawdown",
                "test_mse",
            ]
        ].to_string(index=False)
    )

    print("\n[PERFORMANCE SENSITIVITY]")
    print(pd.DataFrame(diffs).to_string(index=False))

    if not pair_df.empty:
        print("\n[MATCHED-STEP GRADIENT SENSITIVITY]")
        print(paired_summary.to_string(index=False))

    print(f"[DONE] {out_dir.resolve()}")


def main():
    args = arguments()

    if (
        args.max_weight != 0.2
        or args.perturbation <= 0
        or not 0 < args.alpha <= 1
    ):
        raise ValueError(
            "Check max-weight (0.2), perturbation and alpha"
        )

    torch.set_num_threads(1)
    dataset = build_data(args)

    for seed in args.seeds:
        one_seed(args, seed, dataset)


if __name__ == "__main__":
    main()
