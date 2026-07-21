from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset,
    chronological_split,
)
from implementations.dfl_mvo_lee2025.src.model import ReturnMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--train-end",
        default="2021-12-31",
    )
    parser.add_argument(
        "--validation-end",
        default="2022-12-31",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            predictions = model(features)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "DataLoader에 샘플이 없습니다."
        )

    return total_loss / total_samples


def make_output_dir(
    price_path: Path,
    args: argparse.Namespace,
) -> Path:
    dataset_name = price_path.stem.replace(
        "_adjusted_close",
        "",
    )

    return (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "tuning_mse"
        / dataset_name
        / f"hidden_{args.hidden_dim}"
        / f"dropout_{args.dropout:.2f}"
        / f"lr_{args.learning_rate:g}"
        / f"wd_{args.weight_decay:g}"
        / f"seed_{args.seed}"
    )


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(
            "--dropout은 0 이상 1 미만이어야 합니다."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate는 0보다 커야 합니다."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight-decay는 0 이상이어야 합니다."
        )

    if args.patience < 1:
        raise ValueError(
            "--patience는 1 이상이어야 합니다."
        )

    if args.min_delta < 0:
        raise ValueError(
            "--min-delta는 0 이상이어야 합니다."
        )

    set_seed(args.seed)

    price_path = Path(args.price_csv)

    if not price_path.is_absolute():
        price_path = PROJECT_ROOT / price_path

    output_dir = make_output_dir(
        price_path=price_path,
        args=args,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.csv"
    summary_path = output_dir / "summary.json"

    dataset = RollingMVODataset(
        price_csv=price_path,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float32,
    )

    train_set, validation_set, test_set = chronological_split(
        dataset=dataset,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ReturnMLP(
        n_assets=dataset.n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).float().to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_validation_loss = float("inf")
    best_train_loss = float("nan")
    best_epoch = 0
    patience_count = 0
    history_rows = []

    print("=" * 80)
    print("MSE Hyperparameter Tuning")
    print("=" * 80)
    print("Device:", device)
    print("Hidden dimension:", args.hidden_dim)
    print("Dropout:", args.dropout)
    print("Learning rate:", args.learning_rate)
    print("Weight decay:", args.weight_decay)
    print("Seed:", args.seed)
    print("Trainable parameters:", parameter_count)
    print(
        "Train / Validation / Test:",
        len(train_set),
        len(validation_set),
        len(test_set),
    )
    print("=" * 80)

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_loss = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
                "generalization_gap":
                    validation_loss - train_loss,
            }
        )

        pd.DataFrame(history_rows).to_csv(
            history_path,
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_loss:.8f} | "
            f"Validation MSE: {validation_loss:.8f} | "
            f"Gap: {validation_loss - train_loss:.8f}"
        )

        improved = (
            validation_loss
            < best_validation_loss - args.min_delta
        )

        if improved:
            best_validation_loss = validation_loss
            best_train_loss = train_loss
            best_epoch = epoch
            patience_count = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "best_epoch":
                        best_epoch,
                    "train_mse":
                        best_train_loss,
                    "validation_mse":
                        best_validation_loss,
                    "generalization_gap":
                        best_validation_loss
                        - best_train_loss,
                    "n_assets":
                        dataset.n_assets,
                    "lookback":
                        args.lookback,
                    "hidden_dim":
                        args.hidden_dim,
                    "dropout":
                        args.dropout,
                    "learning_rate":
                        args.learning_rate,
                    "weight_decay":
                        args.weight_decay,
                    "seed":
                        args.seed,
                    "parameter_count":
                        parameter_count,
                    "tickers":
                        dataset.tickers,
                },
                best_model_path,
            )
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"{args.patience} epochs 동안 개선 없음"
            )
            break

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    summary = {
        "dataset":
            price_path.stem.replace(
                "_adjusted_close",
                "",
            ),
        "hidden_dim":
            args.hidden_dim,
        "dropout":
            args.dropout,
        "learning_rate":
            args.learning_rate,
        "weight_decay":
            args.weight_decay,
        "seed":
            args.seed,
        "parameter_count":
            parameter_count,
        "best_epoch":
            best_epoch,
        "best_train_mse":
            best_train_loss,
        "best_validation_mse":
            best_validation_loss,
        "generalization_gap":
            best_validation_loss - best_train_loss,
        "test_mse":
            None,
    }

    if args.evaluate_test:
        test_loss = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        summary["test_mse"] = test_loss

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("Tuning result")
    print("=" * 80)
    print("Best epoch:", best_epoch)
    print(
        f"Best Train MSE: "
        f"{best_train_loss:.8f}"
    )
    print(
        f"Best Validation MSE: "
        f"{best_validation_loss:.8f}"
    )
    print(
        f"Generalization gap: "
        f"{best_validation_loss - best_train_loss:.8f}"
    )

    if args.evaluate_test:
        print(
            f"Test MSE: "
            f"{summary['test_mse']:.8f}"
        )

    print("Saved directory:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
