from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
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
    parser.add_argument("--risk-aversion", type=float, default=1.0)

    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--validation-end", default="2022-12-31")

    return parser.parse_args()


def build_markowitz_layer(n_assets: int) -> CvxpyLayer:
    """
    w*(mu, Sigma) = argmax_w mu^T w - lambda * w^T Sigma w

    risk_factor.T @ risk_factor = lambda * Sigma 로 두면
    lambda * w^T Sigma w = ||risk_factor @ w||_2^2 이다.
    """
    weights = cp.Variable(n_assets)
    expected_returns = cp.Parameter(n_assets)
    risk_factor = cp.Parameter((n_assets, n_assets))

    objective = cp.Minimize(
        cp.sum_squares(risk_factor @ weights)
        - expected_returns @ weights
    )

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= 1,
    ]

    problem = cp.Problem(objective, constraints)

    if not problem.is_dpp():
        raise ValueError("Markowitz problem이 DPP 조건을 만족하지 않습니다.")

    return CvxpyLayer(
        problem,
        parameters=[expected_returns, risk_factor],
        variables=[weights],
    )


def covariance_to_risk_factor(
    covariance: torch.Tensor,
    risk_aversion: float,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    covariance: [batch, n_assets, n_assets]

    covariance = L @ L.T 일 때
    R = sqrt(lambda) * L.T 로 설정하면
    R.T @ R = lambda * covariance 이다.
    """
    n_assets = covariance.size(-1)

    eye = torch.eye(
        n_assets,
        dtype=covariance.dtype,
        device=covariance.device,
    ).unsqueeze(0)

    regularized_covariance = covariance + jitter * eye
    chol = torch.linalg.cholesky(regularized_covariance)

    return (risk_aversion ** 0.5) * chol.transpose(-1, -2)


class MLPWithMarkowitz(nn.Module):
    """
    MLP가 다음 날 예상수익률을 출력하고,
    Markowitz layer가 그 예상수익률로 포트폴리오 비중을 계산한다.
    """

    def __init__(
        self,
        n_assets: int,
        lookback: int,
        hidden_dim: int,
        risk_aversion: float,
    ) -> None:
        super().__init__()

        self.return_model = ReturnMLP(
            n_assets=n_assets,
            lookback=lookback,
            hidden_dim=hidden_dim,
        )
        self.markowitz_layer = build_markowitz_layer(n_assets)
        self.risk_aversion = risk_aversion

    def forward(
        self,
        features: torch.Tensor,
        covariance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted_returns = self.return_model(features)

        # cvxpylayers는 CPU float64로 실행한다.
        predicted_returns_solver = predicted_returns.to(
            device="cpu",
            dtype=torch.float64,
        )
        covariance_solver = covariance.to(
            device="cpu",
            dtype=torch.float64,
        )

        risk_factor = covariance_to_risk_factor(
            covariance=covariance_solver,
            risk_aversion=self.risk_aversion,
        )

        portfolio_weights, = self.markowitz_layer(
            predicted_returns_solver,
            risk_factor,
        )

        return predicted_returns, portfolio_weights


def run_epoch(
    model: MLPWithMarkowitz,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_mse = 0.0
    total_samples = 0

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device)
        covariance = batch["covariance"]

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            predicted_returns, portfolio_weights = model(
                features=features,
                covariance=covariance,
            )

            # 핵심:
            # Markowitz 비중까지 계산하지만 loss는 수익률 예측값에 적용한다.
            loss = nn.functional.mse_loss(
                predicted_returns,
                targets,
            )

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = features.size(0)
        total_mse += loss.item() * batch_size
        total_samples += batch_size

    return total_mse / total_samples


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    price_path = Path(args.price_csv)

    if not price_path.is_absolute():
        price_path = PROJECT_ROOT / price_path

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

    model = MLPWithMarkowitz(
        n_assets=dataset.n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        risk_aversion=args.risk_aversion,
    ).float().to(device)

    optimizer = torch.optim.AdamW(
        model.return_model.parameters(),
        lr=args.learning_rate,
    )

    output_dir = (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "mlp_markowitz_mse"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"

    best_validation_mse = float("inf")
    patience_count = 0

    print("Device:", device)
    print("Loss: Return MSE")
    print("Risk aversion:", args.risk_aversion)

    for epoch in range(1, args.epochs + 1):
        train_mse = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )

        validation_mse = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_mse:.8f} | "
            f"Validation MSE: {validation_mse:.8f}"
        )

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": model.return_model.state_dict(),
                    "validation_mse": validation_mse,
                    "risk_aversion": args.risk_aversion,
                    "n_assets": dataset.n_assets,
                    "lookback": args.lookback,
                    "hidden_dim": args.hidden_dim,
                    "tickers": dataset.tickers,
                },
                best_model_path,
            )
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print("Early stopping")
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.return_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_mse = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
    )

    print(f"Best Validation MSE: {best_validation_mse:.8f}")
    print(f"Test MSE: {test_mse:.8f}")
    print(f"Saved: {best_model_path}")


if __name__ == "__main__":
    main()
