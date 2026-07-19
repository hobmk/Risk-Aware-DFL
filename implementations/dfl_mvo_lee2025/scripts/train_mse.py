from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument(
        "--train-end",
        default="2021-12-31",
    )
    parser.add_argument(
        "--validation-end",
        default="2022-12-31",
    )

    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            predictions = model(features)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = features.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)

    price_path = Path(args.price_csv)

    if not price_path.is_absolute():
        price_path = PROJECT_ROOT / price_path

    output_dir = (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "mse_baseline"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("자산 수:", dataset.n_assets)
    print(
        "Train / Validation / Test:",
        len(train_set),
        len(validation_set),
        len(test_set),
    )

    model = ReturnMLP(
        n_assets=dataset.n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
    )

    # Dataset이 float32이므로 모델도 float32로 통일
    model = model.float().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    best_validation_loss = float("inf")
    patience_count = 0

    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
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

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_loss:.8f} | "
            f"Validation MSE: {validation_loss:.8f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_assets": dataset.n_assets,
                    "lookback": args.lookback,
                    "hidden_dim": args.hidden_dim,
                    "validation_loss": validation_loss,
                    "tickers": dataset.tickers,
                },
                best_model_path,
            )
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(
                f"Early stopping: "
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

    test_loss = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print()
    print("=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(
        f"Best Validation MSE: "
        f"{checkpoint['validation_loss']:.8f}"
    )
    print(f"Test MSE: {test_loss:.8f}")
    print(f"모델 저장 위치: {best_model_path}")


if __name__ == "__main__":
    main()