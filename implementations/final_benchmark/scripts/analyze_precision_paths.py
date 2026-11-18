
"""Analyze saved 2x2 training paths and re-evaluate checkpoints identically.

No training or checkpoint modification. Run from repository root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from implementations.final_benchmark.scripts import run_solver_precision_2x2 as core
from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset, chronological_split, fit_feature_standardizer,
)
from implementations.covariance_baselines.src.dataset import CovarianceBaselineDataset


CONTRASTS = (
    "perturb_default", "perturb_tight",
    "solver_only_base", "solver_only_perturbed",
)


def parse_args():
    p = argparse.ArgumentParser(description="Offline path analysis + common solver test")
    p.add_argument("--risk", choices=["ridge", "sample"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--label", default="precision_ultra_traced_v1")
    p.add_argument("--root", default="implementations/final_benchmark/outputs/diagnostics/solver_2x2")
    p.add_argument("--grad-threshold", type=float, default=0.01)
    p.add_argument("--prediction-threshold", type=float, default=1e-4)
    p.add_argument("--parameter-threshold", type=float, default=1e-4)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def earliest(df, column, threshold):
    selected = df.loc[df[column].gt(threshold)]
    if selected.empty:
        return None, None
    return int(selected.iloc[0]["step"]), int(selected.iloc[0]["epoch"])


def analyze_paths(folder, cli):
    path = folder / "paired_gradient_steps.csv"
    if not path.is_file():
        print(f"[SKIP] No paired gradient steps: {path}")
        return

    df = pd.read_csv(path)
    required = {
        "contrast", "step", "epoch", "gradient_diff_l2",
        "prediction_diff_max", "post_step_parameter_diff_l2",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")

    rows, windows = [], []
    for contrast in CONTRASTS:
        part = df.loc[df.contrast.eq(contrast)].sort_values("step").reset_index(drop=True)
        if part.empty:
            continue

        grad_step, grad_epoch = earliest(part, "gradient_diff_l2", cli.grad_threshold)
        pred_step, pred_epoch = earliest(part, "prediction_diff_max", cli.prediction_threshold)
        param_step, param_epoch = earliest(part, "post_step_parameter_diff_l2", cli.parameter_threshold)

        rows.append({
            "risk": cli.risk,
            "seed": cli.seed,
            "contrast": contrast,
            "matched_steps": len(part),
            "first_observed_step": int(part.iloc[0]["step"]),
            "initial_gradient_diff": float(part.iloc[0].gradient_diff_l2),
            "initial_prediction_diff": float(part.iloc[0].prediction_diff_max),
            "initial_post_parameter_diff": float(part.iloc[0].post_step_parameter_diff_l2),
            "first_grad_over_threshold_step": grad_step,
            "first_grad_over_threshold_epoch": grad_epoch,
            "first_prediction_over_threshold_step": pred_step,
            "first_prediction_over_threshold_epoch": pred_epoch,
            "first_parameter_over_threshold_step": param_step,
            "first_parameter_over_threshold_epoch": param_epoch,
            "fraction_grad_over_threshold": float(
                part.gradient_diff_l2.gt(cli.grad_threshold).mean()
            ),
            "max_gradient_diff": float(part.gradient_diff_l2.max()),
            "max_prediction_diff": float(part.prediction_diff_max.max()),
        })

        for event, step in (("gradient", grad_step), ("prediction", pred_step)):
            if step is None:
                continue
            near = part.loc[
                part.step.between(step - cli.window, step + cli.window)
            ].copy()
            near.insert(0, "event", event)
            near.insert(0, "event_step", step)
            windows.append(near)

    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "first_divergence_summary.csv", index=False)

    if windows:
        pd.concat(windows, ignore_index=True).to_csv(
            folder / "divergence_windows.csv", index=False
        )

    display = [
        "contrast", "initial_gradient_diff",
        "first_grad_over_threshold_step",
        "first_prediction_over_threshold_step",
        "first_parameter_over_threshold_step",
        "fraction_grad_over_threshold",
    ]

    print("\n[FIRST DIVERGENCE]")
    print(summary[display].to_string(index=False))


def prepare_test_data(config, risk):
    common = dict(
        price_csv=config.price_csv,
        lookback=config.lookback,
        covariance_jitter=1e-6,
        dtype=torch.float64,
    )

    sample = RollingMVODataset(
        **common,
        risk_cache_dir="implementations/dfl_mvo_lee2025/cache/risk_matrix",
    )

    if risk == "ridge":
        ridge = CovarianceBaselineDataset(
            **common,
            method="ridge",
            ridge_shrinkage=config.rho,
        )

        if (
            sample.tickers != ridge.tickers
            or not sample.target_dates.equals(ridge.target_dates)
        ):
            raise RuntimeError("Sample/Ridge dataset alignment mismatch")

        base_covariance = ridge.covariances
    else:
        base_covariance = sample.covariances

    train, _, test = chronological_split(
        sample, config.train_end, config.validation_end
    )

    test_ix = np.asarray(test.indices, dtype=int)
    test_ix = test_ix[
        sample.target_dates[test_ix] <= pd.Timestamp(config.test_end)
    ]

    if len(test_ix) == 0:
        raise RuntimeError("Empty test interval")

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

    return sample, x, y, base_covariance, test_ix


def common_test(folder, cli, config):
    sample, x, y, base_cov, test_ix = prepare_test_data(config, cli.risk)

    solver_args = {
        "eps": config.tight_eps,
        "max_iters": config.tight_max_iters,
        "acceleration_lookback": 0,
    }

    if config.tight_eps > 1e-10:
        print(
            f"[NOTE] Saved run eps={config.tight_eps:g}; "
            "common test uses that value."
        )

    models = core.make_arms(
        config,
        sample.n_assets,
        {"base": base_cov, "perturbed": base_cov},
    )

    targets = y[test_ix].detach().cpu().numpy()
    records = []

    for arm_name in core.ARM_NAMES:
        checkpoint = folder / f"best_{arm_name}.pt"

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        payload = core.load_saved_checkpoint(checkpoint)
        model = models[arm_name]["model"]

        model.return_model.load_state_dict(
            payload["model_state_dict"], strict=True
        )

        scores, predictions, weights = core.evaluate(
            model, x, y, base_cov, test_ix,
            solver_args, config, collect=True,
        )

        daily_returns = np.sum(weights * targets, axis=1)

        performance = core.evaluate_portfolio(
            daily_returns,
            weights,
            targets,
            cost_bps=(0.0, 5.0, 10.0),
        )

        records.append({
            "risk": cli.risk,
            "seed": cli.seed,
            "arm": arm_name,
            "evaluation": "common_precise_base",
            "best_epoch": int(payload["best_epoch"]),
            "test_mse": float(scores["mse"]),
            **performance,
        })

        print(
            f"[COMMON TEST] {arm_name:20s} "
            f"epoch={payload['best_epoch']:3d} "
            f"Sharpe={performance['gross_sharpe']:.5f} "
            f"CAGR={performance['gross_annualized_return']:.5f}",
            flush=True,
        )

    common = pd.DataFrame(records)
    common.to_csv(folder / "common_solver_test.csv", index=False)

    native_path = folder / "test_portfolio_summary.csv"

    if native_path.exists():
        native = pd.read_csv(native_path)

        columns = [
            "arm", "gross_sharpe",
            "gross_annualized_return",
            "gross_max_drawdown", "test_mse",
        ]

        native = native[columns].rename(columns={
            c: f"native_{c}"
            for c in columns if c != "arm"
        })

        comparison = common.merge(
            native, on="arm", how="left", validate="one_to_one"
        )

        for metric in (
            "gross_sharpe",
            "gross_annualized_return",
            "gross_max_drawdown",
        ):
            comparison[f"common_minus_native_{metric}"] = (
                comparison[metric]
                - comparison[f"native_{metric}"]
            )

        comparison.to_csv(
            folder / "common_vs_native.csv", index=False
        )

    sensitivity = []

    for solver in ("default", "tight"):
        pair = common.set_index("arm")
        a = pair.loc[f"{solver}_base"]
        b = pair.loc[f"{solver}_perturbed"]

        sensitivity.append({
            "solver": solver,
            "base_sharpe": float(a.gross_sharpe),
            "perturbed_sharpe": float(b.gross_sharpe),
            "abs_sharpe_gap": float(
                abs(b.gross_sharpe - a.gross_sharpe)
            ),
            "abs_cagr_gap": float(
                abs(b.gross_annualized_return - a.gross_annualized_return)
            ),
            "abs_mdd_gap": float(
                abs(b.gross_max_drawdown - a.gross_max_drawdown)
            ),
        })

    sensitivity_df = pd.DataFrame(sensitivity)

    sensitivity_df.to_csv(
        folder / "common_solver_sensitivity.csv", index=False
    )

    print("\n[COMMON SOLVER PERTURBATION SENSITIVITY]")
    print(sensitivity_df.to_string(index=False))

    (folder / "common_solver_config.json").write_text(
        json.dumps({
            "solver": solver_args,
            "covariance": "original",
            "evaluation_data": "same test split as original run",
            "checkpoints": list(core.ARM_NAMES),
        }, indent=2),
        encoding="utf-8",
    )


def main():
    cli = parse_args()

    folder = (
        Path(cli.root)
        / cli.risk
        / f"seed_{cli.seed}"
        / cli.label
    )

    if not folder.is_dir():
        raise FileNotFoundError(folder)

    config_path = folder / "config.json"

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    config = SimpleNamespace(**json.loads(
        config_path.read_text(encoding="utf-8")
    ))

    if config.risk != cli.risk or int(config.seed) != cli.seed:
        raise ValueError("Risk/seed do not match saved config.json")

    torch.set_num_threads(1)

    analyze_paths(folder, cli)

    if not cli.skip_eval:
        common_test(folder, cli, config)

    print(f"\n[DONE] {folder.resolve()}")


if __name__ == "__main__":
    main()
