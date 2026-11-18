
"""Gradient-traced full 2x2 training using the original experiment pipeline.

The existing run_solver_precision_2x2.py remains unchanged.
Only read-only gradient diagnostics are added to its training procedure.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import torch

from implementations.final_benchmark.scripts import run_solver_precision_2x2 as core


SPIKE_WINDOW = 20
SPIKE_RATIO = 5.0
SPIKE_MIN = 0.01
PAIR_GRAD_THRESHOLD = 0.01

batches = core.batches
flat = core.flat
forward_objective = core.forward_objective
oracle_costs = core.oracle_costs
evaluate = core.evaluate


def save_gradient_diagnostics(out, grad_rows, pair_rows):
    """Save combined MLP parameter gradient diagnostics."""
    gradients = pd.DataFrame(grad_rows)
    pairs = pd.DataFrame(pair_rows)

    gradients.to_csv(out / "gradient_steps.csv", index=False)
    pairs.to_csv(out / "paired_gradient_steps.csv", index=False)

    if not gradients.empty:
        summary = gradients.groupby("arm", sort=False).agg(
            n_steps=("gradient_l2", "size"),
            n_spikes=("gradient_spike", "sum"),
            spike_rate=("gradient_spike", "mean"),
            median_grad=("gradient_l2", "median"),
            p95_grad=("gradient_l2", lambda x: x.quantile(.95)),
            max_grad=("gradient_l2", "max"),
        ).reset_index()

        summary.to_csv(
            out / "gradient_spikes.csv",
            index=False,
        )

    if not pairs.empty:
        summary = pairs.groupby("contrast", sort=False).agg(
            matched_steps=("gradient_diff_l2", "size"),
            n_large_diff=("large_gradient_diff", "sum"),
            large_diff_rate=("large_gradient_diff", "mean"),
            median_grad_diff=("gradient_diff_l2", "median"),
            p95_grad_diff=("gradient_diff_l2", lambda x: x.quantile(.95)),
            max_grad_diff=("gradient_diff_l2", "max"),
            max_prediction_diff=("prediction_diff_max", "max"),
            first_spike_rate=("first_spike", "mean"),
            second_spike_rate=("second_spike", "mean"),
        ).reset_index()

        summary.to_csv(
            out / "paired_gradient_summary.csv",
            index=False,
        )


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
    gradient_steps, paired_steps = [], []

    recent = {
        name: deque(maxlen=SPIKE_WINDOW)
        for name in arms
    }

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

                # Original backward computation.
                loss.backward()

                grad = flat(model, grads=True)

                # Read-only gradient diagnostics.
                gnorm = float(np.linalg.norm(grad))
                old = recent[name]

                previous_median = (
                    float(np.median(old))
                    if len(old) >= 5
                    else np.nan
                )

                ratio = (
                    gnorm / max(previous_median, 1e-8)
                    if np.isfinite(previous_median)
                    else np.nan
                )

                spike = bool(
                    np.isfinite(ratio)
                    and ratio >= SPIKE_RATIO
                    and gnorm >= SPIKE_MIN
                )

                old.append(gnorm)

                gradient_steps.append({
                    "epoch": epoch,
                    "step": global_step,
                    "arm": name,
                    "batch_size": len(ix),
                    "gradient_l2": gnorm,
                    "previous_median_l2": previous_median,
                    "gradient_ratio": ratio,
                    "gradient_spike": spike,
                })

                prediction_before = (
                    prediction.detach()
                    .cpu()
                    .numpy()
                    .copy()
                )

                # Original optimizer update.
                optimizer.step()

                diagnostics[name] = {
                    "gradient": grad,
                    "gradient_spike": spike,
                    "prediction": prediction_before,
                    "parameters": flat(model),
                }

                train_stats[name][0] += (
                    float(loss.detach().item()) * len(ix)
                )
                train_stats[name][1] += len(ix)

            # Preserve the original step_trace output.
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

            # Additional full-resolution paired diagnostics.
            for first, second, contrast in (
                ("default_base", "default_perturbed", "perturb_default"),
                ("tight_base", "tight_perturbed", "perturb_tight"),
                ("default_base", "tight_base", "solver_only_base"),
                ("default_perturbed", "tight_perturbed", "solver_only_perturbed"),
            ):
                if (
                    first not in diagnostics
                    or second not in diagnostics
                ):
                    continue

                a = diagnostics[first]
                b = diagnostics[second]

                ga = a["gradient"]
                gb = b["gradient"]

                difference = float(
                    np.linalg.norm(ga - gb)
                )

                paired_steps.append({
                    "epoch": epoch,
                    "step": global_step,
                    "contrast": contrast,
                    "first": first,
                    "second": second,
                    "gradient_diff_l2": difference,
                    "gradient_relative_diff": (
                        difference / max(
                            float(np.linalg.norm(ga)),
                            float(np.linalg.norm(gb)),
                            1e-8,
                        )
                    ),
                    "large_gradient_diff": (
                        difference > PAIR_GRAD_THRESHOLD
                    ),
                    "prediction_diff_max": float(
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
                    "first_spike": a["gradient_spike"],
                    "second_spike": b["gradient_spike"],
                })

        n_active_before = sum(
            arm["active"] for arm in arms.values()
        )

        # Original validation and early stopping.
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

        if (
            epoch == 1
            or epoch % 5 == 0
            or not any(a["active"] for a in arms.values())
        ):
            save_gradient_diagnostics(
                out,
                gradient_steps,
                paired_steps,
            )

        if not any(
            arm["active"] for arm in arms.values()
        ):
            print(
                "All four arms reached early stopping."
            )
            break

    save_gradient_diagnostics(
        out,
        gradient_steps,
        paired_steps,
    )


if __name__ == "__main__":
    # Keep the original main(), setup, arms and testing logic.
    # Replace only the training function with its traced equivalent.
    core.train_arms = train_arms
    core.main()
