from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .losses import compute_losses
from .model import EigenRCRDFL


@dataclass
class EpochMetrics:
    total_loss: float
    mse: float
    regret: float
    gradient_norm: float
    weight_sum_error: float
    minimum_weight: float
    maximum_weight: float
    minimum_effective_eigenvalue: float
    n_samples: int


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    test_metrics: EpochMetrics
    history: pd.DataFrame
    checkpoint_path: Path


def _gradient_norm(
    parameters: list[torch.nn.Parameter],
) -> float:
    squared = 0.0

    for parameter in parameters:
        if parameter.grad is not None:
            squared += (
                parameter.grad.detach().norm(2).item() ** 2
            )

    return squared ** 0.5


def run_epoch(
    model: EigenRCRDFL,
    dataloader: DataLoader,
    device: torch.device,
    alpha: float,
    mse_scale: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)

    total_sum = 0.0
    mse_sum = 0.0
    regret_sum = 0.0
    gradient_sum = 0.0

    weight_sum_error = 0.0
    minimum_weight = float("inf")
    maximum_weight = float("-inf")
    minimum_effective_eigenvalue = float("inf")

    n_samples = 0

    for batch_index, batch in enumerate(dataloader):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        features = batch["features"].to(
            device=device,
            dtype=torch.float32,
        )

        targets = batch["target"].to(
            device=device,
            dtype=torch.float32,
        )

        covariance = batch["covariance"].to(
            device="cpu",
            dtype=torch.float64,
        )

        eigen_risk = batch["eigen_risk"].to(
            device="cpu",
            dtype=torch.float64,
        )

        batch_size = features.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(
                features=features,
                covariance=covariance,
                eigen_risk=eigen_risk,
            )

            oracle_weights = model.solve_oracle(
                true_returns=targets,
                risk_factor=output.risk_factor,
            )

            losses = compute_losses(
                predicted_returns=output.predicted_returns,
                predicted_weights=output.predicted_weights,
                oracle_weights=oracle_weights,
                true_returns=targets,
                risk_factor=output.risk_factor,
                alpha=alpha,
                mse_scale=mse_scale,
            )

        gradient_norm = 0.0

        if training:
            losses.total.backward()

            parameters = [
                p
                for p in model.return_model.parameters()
                if p.requires_grad
            ]

            gradient_norm = _gradient_norm(
                parameters
            )

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    gradient_clip_norm,
                )

            optimizer.step()

        weights = output.predicted_weights.detach()

        min_eigenvalue = torch.linalg.eigvalsh(
            output.effective_covariance.detach()
        ).min().item()

        total_sum += (
            losses.total.detach().item()
            * batch_size
        )

        mse_sum += (
            losses.mse.detach().item()
            * batch_size
        )

        regret_sum += (
            losses.regret.detach().item()
            * batch_size
        )

        gradient_sum += (
            gradient_norm
            * batch_size
        )

        n_samples += batch_size

        weight_sum_error = max(
            weight_sum_error,
            (
                weights.sum(dim=-1)
                - 1.0
            ).abs().max().item(),
        )

        minimum_weight = min(
            minimum_weight,
            weights.min().item(),
        )

        maximum_weight = max(
            maximum_weight,
            weights.max().item(),
        )

        minimum_effective_eigenvalue = min(
            minimum_effective_eigenvalue,
            min_eigenvalue,
        )

    if n_samples == 0:
        raise RuntimeError("DataLoader에 sample이 없습니다.")

    return EpochMetrics(
        total_loss=total_sum / n_samples,
        mse=mse_sum / n_samples,
        regret=regret_sum / n_samples,
        gradient_norm=gradient_sum / n_samples,
        weight_sum_error=weight_sum_error,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        minimum_effective_eigenvalue=minimum_effective_eigenvalue,
        n_samples=n_samples,
    )


def fit(
    model: EigenRCRDFL,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    alpha: float,
    mse_scale: float,
    output_dir: str | Path,
    gradient_clip_norm: float | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    max_test_batches: int | None = None,
) -> TrainingResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "best_model.pt"
    )

    history_path = (
        output_dir
        / "history.csv"
    )

    model.to(device)

    best_epoch = 0
    best_validation_loss = float("inf")
    no_improvement = 0

    history_rows = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            alpha,
            mse_scale,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
            max_batches=max_train_batches,
        )

        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            alpha,
            mse_scale,
            max_batches=max_validation_batches,
        )

        row = {"epoch": epoch}

        row.update({
            f"train_{k}": v
            for k, v in asdict(train_metrics).items()
        })

        row.update({
            f"validation_{k}": v
            for k, v in asdict(validation_metrics).items()
        })

        history_rows.append(row)

        pd.DataFrame(
            history_rows
        ).to_csv(
            history_path,
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Total {train_metrics.total_loss:.8f} | "
            f"MSE {train_metrics.mse:.8f} | "
            f"Regret {train_metrics.regret:.8f} | "
            f"Val Total {validation_metrics.total_loss:.8f} | "
            f"Val MSE {validation_metrics.mse:.8f} | "
            f"Val Regret {validation_metrics.regret:.8f} | "
            f"Grad {train_metrics.gradient_norm:.3e}"
        )

        if (
            validation_metrics.total_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_metrics.total_loss
            )

            best_epoch = epoch
            no_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )

        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(
                f"Early stopping: "
                f"{patience} epochs without improvement."
            )
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        alpha,
        mse_scale,
        max_batches=max_test_batches,
    )

    pd.DataFrame([
        asdict(test_metrics)
    ]).to_csv(
        output_dir / "test_metrics.csv",
        index=False,
    )

    print(
        f"Best Epoch {best_epoch:03d} | "
        f"Best Val {best_validation_loss:.8f} | "
        f"Test Total {test_metrics.total_loss:.8f} | "
        f"Test MSE {test_metrics.mse:.8f} | "
        f"Test Regret {test_metrics.regret:.8f}"
    )

    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        test_metrics=test_metrics,
        history=pd.DataFrame(history_rows),
        checkpoint_path=checkpoint_path,
    )