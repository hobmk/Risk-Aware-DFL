
"""Full 2x2 DFL experiment: solver precision x covariance perturbation.

Run from the repository root. --evaluate-only reuses all four saved
best-model checkpoints and does not rerun training.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from implementations.common.portfolio_evaluation import evaluate_portfolio
from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset, chronological_split, fit_feature_standardizer,
)
from implementations.covariance_baselines.src.dataset import CovarianceBaselineDataset
from implementations.dfl_mvo_lee2025.scripts.train_mlp_markowitz_combined import MLPWithMarkowitz
from implementations.dfl_mvo_lee2025.src.optimization import covariance_to_risk_factor
from implementations.dfl_mvo_lee2025.src.losses import markowitz_cost


ARM_NAMES = ("default_base", "default_perturbed", "tight_base", "tight_perturbed")


def parse_args():
    p = argparse.ArgumentParser(description="Full training: solver x covariance perturbation")
    p.add_argument("--risk", choices=["sample", "ridge"], required=True)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    p.add_argument("--out-dir", default="implementations/final_benchmark/outputs/diagnostics/solver_2x2")
    p.add_argument("--label", default="full")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--trace-every", type=int, default=25)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--validation-end", default="2022-12-31")
    p.add_argument("--test-end", default="2024-12-31")
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
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    return p.parse_args()


def batches(indices, batch_size, max_batches=0):
    for k, start in enumerate(range(0, len(indices), batch_size)):
        if max_batches and k >= max_batches:
            break
        yield indices[start:start + batch_size]


def flat(model, grads=False):
    values = [
        (p.grad if grads else p.detach()).reshape(-1)
        for p in model.return_model.parameters()
    ]
    return torch.cat(values).detach().cpu().numpy().copy()


def forward_objective(model, x, y, covariance, solver_args, args):
    prediction = model.return_model(x)
    factor = covariance_to_risk_factor(covariance, args.risk_aversion)

    (weights,) = model.markowitz_layer(
        prediction.to(torch.float64), factor, solver_args=solver_args,
    )

    costs = markowitz_cost(weights, y.to(torch.float64), factor)
    mse = F.mse_loss(prediction, y)

    # Oracle cost는 Predictor 파라미터에 대한 상수다.
    # 따라서 학습 Gradient에는 영향을 주지 않는다.
    objective = (
        args.alpha * costs.mean()
        + (1 - args.alpha) * args.mse_scale * mse.to(torch.float64)
    )

    return objective, prediction, weights, costs, mse


def oracle_costs(model, y, covariance, indices, solver_args, args):
    results = []

    with torch.no_grad():
        for ix in batches(indices, args.batch_size):
            factor = covariance_to_risk_factor(
                covariance[ix], args.risk_aversion,
            )

            (oracle,) = model.markowitz_layer(
                y[ix].to(torch.float64),
                factor,
                solver_args=solver_args,
            )

            cost = markowitz_cost(
                oracle, y[ix].to(torch.float64), factor,
            )

            results.append(cost.detach().cpu().numpy())

    return np.concatenate(results)


def evaluate(model, x, y, covariance, indices, solver_args, args,
             reference=None, collect=False):
    model.eval()
    cost_sum, mse_sum, count = 0.0, 0.0, 0
    prediction_list, weight_list = [], []

    with torch.no_grad():
        for ix in batches(indices, args.batch_size):
            _, prediction, weights, costs, mse = forward_objective(
                model, x[ix], y[ix], covariance[ix], solver_args, args,
            )

            cost_sum += float(costs.sum().item())
            mse_sum += float(mse.item()) * len(ix)
            count += len(ix)

            if collect:
                prediction_list.append(
                    prediction.detach().cpu().numpy().copy()
                )
                weight_list.append(
                    weights.detach().cpu().numpy().copy()
                )

    if count == 0:
        raise RuntimeError("Empty evaluation split")

    mean_cost = cost_sum / count
    mean_mse = mse_sum / count
    oracle_mean = (
        float(np.mean(reference))
        if reference is not None
        else float("nan")
    )
    regret = mean_cost - oracle_mean

    metrics = {
        "total": (
            args.alpha * regret
            + (1 - args.alpha) * args.mse_scale * mean_mse
        ),
        "regret": regret,
        "mse": mean_mse,
        "cost": mean_cost,
        "oracle_cost": oracle_mean,
    }

    if collect:
        return (
            metrics,
            np.concatenate(prediction_list),
            np.concatenate(weight_list),
        )

    return metrics


def prepare_output(args):
    out = (
        Path(args.out_dir)
        / args.risk
        / f"seed_{args.seed}"
        / args.label
    )

    if args.evaluate_only and args.overwrite:
        raise ValueError(
            "--evaluate-only and --overwrite cannot be used together"
        )

    if args.evaluate_only:
        missing = [
            str(out / f"best_{name}.pt")
            for name in ARM_NAMES
            if not (out / f"best_{name}.pt").is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing existing checkpoint(s):\n"
                + "\n".join(missing)
            )

        print(
            f"[EVALUATE-ONLY] Checkpoints found in: {out}",
            flush=True,
        )

    else:
        if out.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out}\n"
                "To preserve checkpoints, use --evaluate-only, "
                "or choose a new --label."
            )

        if out.exists() and args.overwrite:
            shutil.rmtree(out)

        out.mkdir(parents=True, exist_ok=False)

    return out


def make_arms(args, n_assets, covariances):
    solver_configs = {
        "default": {},
        "tight": {
            "eps": args.tight_eps,
            "max_iters": args.tight_max_iters,
            "acceleration_lookback": 0,
        },
    }

    torch.manual_seed(args.seed)

    initial = MLPWithMarkowitz(
        n_assets,
        args.lookback,
        args.hidden_dim,
        0.0,
        args.risk_aversion,
        0.2,
    ).float()

    initial_state = copy.deepcopy(
        initial.return_model.state_dict()
    )

    arms = {}

    for solver, solver_args in solver_configs.items():
        for change, covariance in covariances.items():
            name = f"{solver}_{change}"

            model = MLPWithMarkowitz(
                n_assets,
                args.lookback,
                args.hidden_dim,
                0.0,
                args.risk_aversion,
                0.2,
            ).float()

            model.return_model.load_state_dict(
                initial_state, strict=True,
            )

            arms[name] = {
                "model": model,
                "optimizer": torch.optim.AdamW(
                    model.return_model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                ),
                "solver_args": solver_args,
                "covariance": covariance,
                "active": True,
                "best": float("inf"),
                "best_epoch": 0,
                "patience_count": 0,
            }

    return arms


def train_arms(arms, x, y, train_ix, val_ix, args, out):
    print(
        "[ORACLE] Computing validation oracle costs...",
        flush=True,
    )

    for name, arm in arms.items():
        arm["val_oracle"] = oracle_costs(
            arm["model"],
            y,
            arm["covariance"],
            val_ix,
            arm["solver_args"],
            args,
        )

        print(
            f"  {name}: mean={arm['val_oracle'].mean():.8f}",
            flush=True,
        )

    history, trace = [], []
    generator = torch.Generator().manual_seed(args.seed)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        order = train_ix[
            torch.randperm(
                len(train_ix),
                generator=generator,
            ).numpy()
        ]

        train_stats = {
            name: [0.0, 0]
            for name in arms
        }

        for ix in batches(
            order, args.batch_size, args.max_train_batches
        ):
            global_step += 1
            diagnostics = {}

            for name, arm in arms.items():
                if not arm["active"]:
                    continue

                model = arm["model"]
                optimizer = arm["optimizer"]

                model.train()
                optimizer.zero_grad(set_to_none=True)

                loss, prediction, _, _, _ = forward_objective(
                    model,
                    x[ix],
                    y[ix],
                    arm["covariance"][ix],
                    arm["solver_args"],
                    args,
                )

                loss.backward()

                grad = flat(model, grads=True)
                prediction_before = (
                    prediction.detach().cpu().numpy().copy()
                )

                optimizer.step()

                diagnostics[name] = {
                    "gradient": grad,
                    "prediction": prediction_before,
                    "parameters": flat(model),
                }

                train_stats[name][0] += (
                    float(loss.detach().item()) * len(ix)
                )
                train_stats[name][1] += len(ix)

            if (
                global_step == 1
                or global_step % args.trace_every == 0
            ):
                comparisons = [
                    (
                        "default_base",
                        "default_perturbed",
                        "perturb_default",
                    ),
                    (
                        "tight_base",
                        "tight_perturbed",
                        "perturb_tight",
                    ),
                    (
                        "default_base",
                        "tight_base",
                        "solver_only_base",
                    ),
                ]

                for first, second, contrast in comparisons:
                    if (
                        first not in diagnostics
                        or second not in diagnostics
                    ):
                        continue

                    a = diagnostics[first]
                    b = diagnostics[second]

                    ga = a["gradient"]
                    gb = b["gradient"]

                    na = np.linalg.norm(ga)
                    nb = np.linalg.norm(gb)

                    cosine = (
                        float(np.dot(ga, gb) / (na * nb))
                        if na > 0 and nb > 0
                        else np.nan
                    )

                    trace.append({
                        "epoch": epoch,
                        "step": global_step,
                        "contrast": contrast,
                        "gradient_diff_l2": float(
                            np.linalg.norm(ga - gb)
                        ),
                        "gradient_cosine": cosine,
                        "pre_step_prediction_max_abs_diff": float(
                            np.max(
                                np.abs(
                                    a["prediction"]
                                    - b["prediction"]
                                )
                            )
                        ),
                        "post_step_parameter_diff_l2": float(
                            np.linalg.norm(
                                a["parameters"]
                                - b["parameters"]
                            )
                        ),
                    })

        n_active_before = sum(
            arm["active"] for arm in arms.values()
        )

        for name, arm in arms.items():
            if not arm["active"]:
                continue

            scores = evaluate(
                arm["model"],
                x,
                y,
                arm["covariance"],
                val_ix,
                arm["solver_args"],
                args,
                reference=arm["val_oracle"],
            )

            history.append({
                "epoch": epoch,
                "arm": name,
                "train_proxy_loss": (
                    train_stats[name][0]
                    / max(train_stats[name][1], 1)
                ),
                **scores,
            })

            if scores["total"] < arm["best"]:
                arm["best"] = float(scores["total"])
                arm["best_epoch"] = epoch
                arm["patience_count"] = 0

                # Python 기본 float로 변환해 저장한다.
                # 이후 생성되는 Checkpoint는 weights_only=True와도 호환된다.
                torch.save({
                    "model_state_dict": (
                        arm["model"].return_model.state_dict()
                    ),
                    "best_epoch": int(epoch),
                    "best_validation_total": float(
                        arm["best"]
                    ),
                }, out / f"best_{name}.pt")

            else:
                arm["patience_count"] += 1

                if arm["patience_count"] >= args.patience:
                    arm["active"] = False

        if (
            epoch == 1
            or epoch % 5 == 0
            or n_active_before != sum(
                arm["active"] for arm in arms.values()
            )
        ):
            status = " | ".join(
                f"{name}: {arm['best']:.6g}"
                f"@{arm['best_epoch']} "
                f"{'RUN' if arm['active'] else 'STOP'}"
                for name, arm in arms.items()
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Step {global_step:06d} | {status}",
                flush=True,
            )

        pd.DataFrame(history).to_csv(
            out / "epoch_history.csv",
            index=False,
        )

        pd.DataFrame(trace).to_csv(
            out / "step_trace.csv",
            index=False,
        )

        if not any(
            arm["active"] for arm in arms.values()
        ):
            print(
                "All four arms reached early stopping."
            )
            break


def load_saved_checkpoint(path):
    # 이전 버전의 Checkpoint는 NumPy Scalar를 포함할 수 있다.
    # weights_only=False는 직접 생성한 신뢰할 수 있는 파일에만 사용한다.
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "model_state_dict",
        "best_epoch",
        "best_validation_total",
    }

    if (
        not isinstance(payload, dict)
        or not required.issubset(payload)
    ):
        raise ValueError(
            f"Invalid checkpoint format: {path}"
        )

    return payload


def test_arms(arms, sample, x, y, test_ix, args, out):
    rows, predictions = [], {}

    dates = sample.target_dates[
        test_ix
    ].strftime("%Y-%m-%d")

    targets = (
        y[test_ix].detach().cpu().numpy()
    )

    for name, arm in arms.items():
        checkpoint = load_saved_checkpoint(
            out / f"best_{name}.pt"
        )

        arm["best_epoch"] = int(
            checkpoint["best_epoch"]
        )
        arm["best"] = float(
            checkpoint["best_validation_total"]
        )

        arm["model"].return_model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        scores, predicted, weights = evaluate(
            arm["model"],
            x,
            y,
            arm["covariance"],
            test_ix,
            arm["solver_args"],
            args,
            collect=True,
        )

        predictions[name] = predicted

        daily_returns = np.sum(
            weights * targets,
            axis=1,
        )

        performance = evaluate_portfolio(
            daily_returns,
            weights,
            targets,
            cost_bps=(0.0, 5.0, 10.0),
        )

        rows.append({
            "risk": args.risk,
            "seed": args.seed,
            "arm": name,
            "best_epoch": arm["best_epoch"],
            "best_validation_loss": arm["best"],
            "test_mse": scores["mse"],
            **performance,
        })

        for kind, values in [
            ("predictions", predicted),
            ("weights", weights),
        ]:
            frame = pd.DataFrame(
                values,
                columns=sample.tickers,
            )

            frame.insert(0, "date", dates)

            frame.to_csv(
                out / f"test_{kind}_{name}.csv",
                index=False,
            )

        print(
            f"[TEST] {name:20s} | "
            f"best_epoch={arm['best_epoch']:3d} "
            f"| Sharpe={performance['gross_sharpe']:.4f} "
            f"| CAGR={performance['gross_annualized_return']:.4f}",
            flush=True,
        )

    comparisons = []

    for first, second, contrast in [
        (
            "default_base",
            "default_perturbed",
            "perturb_default",
        ),
        (
            "tight_base",
            "tight_perturbed",
            "perturb_tight",
        ),
        (
            "default_base",
            "tight_base",
            "solver_only_base",
        ),
        (
            "default_perturbed",
            "tight_perturbed",
            "solver_only_perturbed",
        ),
    ]:
        difference = (
            predictions[first]
            - predictions[second]
        )

        comparisons.append({
            "contrast": contrast,
            "max_abs_prediction_diff": float(
                np.abs(difference).max()
            ),
            "prediction_rmse": float(
                np.sqrt(
                    np.mean(difference ** 2)
                )
            ),
        })

    interaction = (
        predictions["default_perturbed"]
        - predictions["default_base"]
        - predictions["tight_perturbed"]
        + predictions["tight_base"]
    )

    comparisons.append({
        "contrast": "solver_x_perturb_interaction",
        "max_abs_prediction_diff": float(
            np.abs(interaction).max()
        ),
        "prediction_rmse": float(
            np.sqrt(
                np.mean(interaction ** 2)
            )
        ),
    })

    summary = pd.DataFrame(rows)
    contrasts = pd.DataFrame(comparisons)

    summary.to_csv(
        out / "test_portfolio_summary.csv",
        index=False,
    )

    contrasts.to_csv(
        out / "test_prediction_contrasts.csv",
        index=False,
    )

    print(
        "\n" + contrasts.to_string(index=False)
    )

    print(
        "\n" + summary[[
            "arm",
            "best_epoch",
            "best_validation_loss",
            "gross_sharpe",
            "gross_annualized_return",
            "gross_max_drawdown",
        ]].to_string(index=False)
    )


def main():
    args = parse_args()

    if min(
        args.epochs,
        args.patience,
        args.batch_size,
        args.trace_every,
    ) < 1:
        raise ValueError(
            "epochs, patience, batch-size, trace-every must be positive"
        )

    if (
        not 0 <= args.alpha <= 1
        or args.perturbation <= 0
        or not 0 <= args.rho <= 1
    ):
        raise ValueError(
            "Invalid alpha, perturbation or rho"
        )

    if (
        args.max_train_batches < 0
        or args.max_val_batches < 0
    ):
        raise ValueError(
            "max-train-batches and max-val-batches cannot be negative"
        )

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    out = prepare_output(args)

    common = {
        "price_csv": args.price_csv,
        "lookback": args.lookback,
        "covariance_jitter": 1e-6,
        "dtype": torch.float64,
    }

    sample = RollingMVODataset(
        **common,
        risk_cache_dir=(
            "implementations/"
            "dfl_mvo_lee2025/"
            "cache/risk_matrix"
        ),
    )

    ridge = CovarianceBaselineDataset(
        **common,
        method="ridge",
        ridge_shrinkage=args.rho,
    )

    if (
        sample.tickers != ridge.tickers
        or not sample.target_dates.equals(
            ridge.target_dates
        )
    ):
        raise RuntimeError(
            "Sample/Ridge alignment mismatch"
        )

    train, val, test = chronological_split(
        sample,
        args.train_end,
        args.validation_end,
    )

    train_ix = np.asarray(
        train.indices, dtype=int,
    )
    val_ix = np.asarray(
        val.indices, dtype=int,
    )
    test_ix = np.asarray(
        test.indices, dtype=int,
    )

    test_ix = test_ix[
        sample.target_dates[test_ix]
        <= pd.Timestamp(args.test_end)
    ]

    if args.max_val_batches:
        val_ix = val_ix[
            :args.max_val_batches * args.batch_size
        ]

    if (
        not len(train_ix)
        or not len(val_ix)
        or not len(test_ix)
    ):
        raise RuntimeError(
            "Training, validation or test split is empty"
        )

    mean, std = fit_feature_standardizer(
        sample,
        train,
    )

    mean, std = mean.float(), std.float()

    features = torch.stack([
        sample[i]["features"].float()
        for i in range(len(sample))
    ])

    x = (
        features - mean
    ) / std

    y = torch.stack([
        sample[i]["target"].float()
        for i in range(len(sample))
    ])

    sample_cov = (
        sample.covariances
        .detach()
        .cpu()
        .numpy()
    )

    ridge_cov = (
        ridge.covariances
        .detach()
        .cpu()
        .numpy()
    )

    base = (
        sample_cov
        if args.risk == "sample"
        else ridge_cov
    )

    rng = np.random.default_rng(
        args.direction_seed
    )

    direction = rng.normal(
        size=sample.n_assets
    )
    direction /= np.linalg.norm(
        direction
    )

    delta = (
        args.perturbation
        * np.linalg.norm(
            sample_cov,
            axis=(1, 2),
        )[:, None, None]
        * np.outer(
            direction,
            direction,
        )[None]
    )

    covariances = {
        "base": torch.from_numpy(
            np.ascontiguousarray(base)
        ),
        "perturbed": torch.from_numpy(
            np.ascontiguousarray(
                base + delta
            )
        ),
    }

    arms = make_arms(
        args,
        sample.n_assets,
        covariances,
    )

    print(
        f"RISK={args.risk} | "
        f"SEED={args.seed} | "
        f"MAX_WEIGHT=0.2 | "
        f"TRAIN={len(train_ix)} | "
        f"VAL={len(val_ix)} | "
        f"TEST={len(test_ix)}",
        flush=True,
    )

    if not args.evaluate_only:
        (out / "config.json").write_text(
            json.dumps(
                vars(args),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        train_arms(
            arms,
            x,
            y,
            train_ix,
            val_ix,
            args,
            out,
        )

    else:
        print(
            "[EVALUATE-ONLY] "
            "Skipping training and validation oracle.",
            flush=True,
        )

    test_arms(
        arms,
        sample,
        x,
        y,
        test_ix,
        args,
        out,
    )

    (out / "evaluation_config.json").write_text(
        json.dumps(
            vars(args),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[DONE] {out.resolve()}"
    )


if __name__ == "__main__":
    main()
