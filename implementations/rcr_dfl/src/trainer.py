from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .decision_model import RCRMLPWithMarkowitz
from .losses import compute_rcr_losses


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: pd.DataFrame
    test_metrics: EpochMetrics
    checkpoint_path: Path


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared_norm = 0.0

    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().norm(2).item() ** 2

    return squared_norm**0.5


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = batch["features"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )
    targets = batch["target"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )

    # CVXPYLayer는 CPU float64를 사용한다.
    covariance = batch["covariance"].to(
        device="cpu",
        dtype=torch.float64,
    )
    residual_covariance = batch["residual_covariance"].to(
        device="cpu",
        dtype=torch.float64,
    )

    return features, targets, covariance, residual_covariance


def run_epoch(
    model: RCRMLPWithMarkowitz,
    dataloader: DataLoader,
    device: torch.device,
    alpha: float,
    mse_scale: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> EpochMetrics:
    if max_batches is not None and max_batches <= 0:
        raise ValueError(
            f"max_batches는 None 또는 0보다 커야 합니다: {max_batches}"
        )

    training = optimizer is not None
    model.train(training)

    total_sum = 0.0
    mse_sum = 0.0
    regret_sum = 0.0
    gradient_norm_sum = 0.0

    weight_sum_error = 0.0
    minimum_weight = float("inf")
    maximum_weight = float("-inf")
    minimum_effective_eigenvalue = float("inf")

    n_samples = 0

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        features, targets, covariance, residual_covariance = _move_batch(
            batch,
            device,
        )
        batch_size = features.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(
                features=features,
                covariance=covariance,
                residual_covariance=residual_covariance,
            )

            oracle_weights = model.solve_oracle(
                true_returns=targets,
                risk_factor=output.risk_factor,
            )

            losses = compute_rcr_losses(
                predicted_returns=output.predicted_returns,
                predicted_weights=output.predicted_weights,
                oracle_weights=oracle_weights,
                true_returns=targets,
                risk_factor=output.risk_factor,
                alpha=alpha,
                mse_scale=mse_scale,
            )

        batch_gradient_norm = 0.0

        if training:
            losses.total.backward()

            parameters = [
                parameter
                for parameter in model.return_model.parameters()
                if parameter.requires_grad
            ]

            batch_gradient_norm = _gradient_norm(parameters)

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    gradient_clip_norm,
                )

            optimizer.step()

        predicted_weights = output.predicted_weights.detach()
        effective_covariance = output.effective_covariance.detach()

        total_sum += losses.total.detach().item() * batch_size
        mse_sum += losses.mse.detach().item() * batch_size
        regret_sum += losses.regret.detach().item() * batch_size
        gradient_norm_sum += batch_gradient_norm * batch_size
        n_samples += batch_size

        batch_weight_sum_error = (
            predicted_weights.sum(dim=-1) - 1.0
        ).abs().max().item()

        weight_sum_error = max(
            weight_sum_error,
            batch_weight_sum_error,
        )
        minimum_weight = min(
            minimum_weight,
            predicted_weights.min().item(),
        )
        maximum_weight = max(
            maximum_weight,
            predicted_weights.max().item(),
        )

        batch_minimum_eigenvalue = torch.linalg.eigvalsh(
            effective_covariance
        ).min().item()

        minimum_effective_eigenvalue = min(
            minimum_effective_eigenvalue,
            batch_minimum_eigenvalue,
        )

    if n_samples == 0:
        raise RuntimeError("DataLoader에 sample이 없습니다.")

    return EpochMetrics(
        total_loss=total_sum / n_samples,
        mse=mse_sum / n_samples,
        regret=regret_sum / n_samples,
        gradient_norm=gradient_norm_sum / n_samples,
        weight_sum_error=weight_sum_error,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        minimum_effective_eigenvalue=minimum_effective_eigenvalue,
        n_samples=n_samples,
    )


def save_checkpoint(
    path: Path,
    model: RCRMLPWithMarkowitz,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def fit(
    model: RCRMLPWithMarkowitz,
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
    metadata: dict[str, Any] | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    max_test_batches: int | None = None,
) -> TrainingResult:
    if epochs <= 0:
        raise ValueError(
            f"epochs는 0보다 커야 합니다: {epochs}"
        )

    if patience <= 0:
        raise ValueError(
            f"patience는 0보다 커야 합니다: {patience}"
        )

    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError(
            "gradient_clip_norm은 None 또는 0보다 큰 값이어야 합니다: "
            f"{gradient_clip_norm}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path / "best_model.pt"
    history_path = output_path / "history.csv"

    model.to(device)

    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    history_rows: list[dict[str, float | int]] = []
    checkpoint_metadata = {} if metadata is None else metadata

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            alpha=alpha,
            mse_scale=mse_scale,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
            max_batches=max_train_batches,
        )

        validation_metrics = run_epoch(
            model=model,
            dataloader=validation_loader,
            device=device,
            alpha=alpha,
            mse_scale=mse_scale,
            max_batches=max_validation_batches,
        )

        row: dict[str, float | int] = {
            "epoch": epoch,
        }

        row.update(
            {
                f"train_{key}": value
                for key, value in asdict(train_metrics).items()
            }
        )
        row.update(
            {
                f"validation_{key}": value
                for key, value in asdict(validation_metrics).items()
            }
        )

        history_rows.append(row)

        pd.DataFrame(history_rows).to_csv(
            history_path,
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Total: {train_metrics.total_loss:.8f} | "
            f"Train MSE: {train_metrics.mse:.8f} | "
            f"Train RCR: {train_metrics.regret:.8f} | "
            f"Val Total: {validation_metrics.total_loss:.8f} | "
            f"Val MSE: {validation_metrics.mse:.8f} | "
            f"Val RCR: {validation_metrics.regret:.8f} | "
            f"Grad: {train_metrics.gradient_norm:.3e}"
        )

        if validation_metrics.total_loss < best_validation_loss:
            best_validation_loss = validation_metrics.total_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=best_validation_loss,
                metadata=checkpoint_metadata,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping: {patience} epoch 동안 "
                "validation loss가 개선되지 않았습니다."
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
        model=model,
        dataloader=test_loader,
        device=device,
        alpha=alpha,
        mse_scale=mse_scale,
        max_batches=max_test_batches,
    )

    pd.DataFrame(
        [asdict(test_metrics)]
    ).to_csv(
        output_path / "test_metrics.csv",
        index=False,
    )

    print(
        f"Best Epoch: {best_epoch:03d} | "
        f"Best Val Total: {best_validation_loss:.8f} | "
        f"Test Total: {test_metrics.total_loss:.8f} | "
        f"Test MSE: {test_metrics.mse:.8f} | "
        f"Test RCR: {test_metrics.regret:.8f}"
    )

    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        history=pd.DataFrame(history_rows),
        test_metrics=test_metrics,
        checkpoint_path=checkpoint_path,
    )