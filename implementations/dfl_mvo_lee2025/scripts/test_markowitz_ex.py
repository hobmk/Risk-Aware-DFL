from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset,
)
from implementations.dfl_mvo_lee2025.scripts.train_mlp_markowitz_regret import (
    MLPWithMarkowitz,
)
from implementations.dfl_mvo_lee2025.src.losses import (
    markowitz_cost,
)


def main() -> None:
    price_path = PROJECT_ROOT / "data/raw/dow30_adjusted_close.csv"

    dataset = RollingMVODataset(
        price_csv=price_path,
        lookback=60,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float32,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    batch = next(iter(loader))

    features = batch["features"]
    targets = batch["target"]
    covariance = batch["covariance"]

    model = MLPWithMarkowitz(
    n_assets=dataset.n_assets,
    lookback=60,
    hidden_dim=256,
    risk_aversion=1.0,
    max_weight=0.2,
)

    model.eval()

    predicted_returns, predicted_weights, risk_factor = model(
        features=features,
        covariance=covariance,
    )

    true_returns_solver = targets.to(
        dtype=torch.float64,
        device="cpu",
    )

    oracle_weights, = model.markowitz_layer(
        true_returns_solver,
        risk_factor,
    )

    predicted_cost = markowitz_cost(
        weights=predicted_weights,
        true_returns=true_returns_solver,
        risk_factor=risk_factor,
    )

    oracle_cost = markowitz_cost(
        weights=oracle_weights,
        true_returns=true_returns_solver,
        risk_factor=risk_factor,
    )

    regret = predicted_cost - oracle_cost

    print("=" * 60)
    print("Tensor shape")
    print("=" * 60)
    print("예측 수익률:", predicted_returns.shape)
    print("예측 포트폴리오:", predicted_weights.shape)
    print("Oracle 포트폴리오:", oracle_weights.shape)

    print()
    print("=" * 60)
    print("첫 번째 샘플")
    print("=" * 60)
    print("예측 수익률:")
    print(predicted_returns[0].detach())

    print("\n예측 포트폴리오 비중:")
    print(predicted_weights[0].detach())

    print("\nOracle 포트폴리오 비중:")
    print(oracle_weights[0].detach())

    print()
    print("=" * 60)
    print("제약조건 확인")
    print("=" * 60)
    print(
        "예측 비중 합:",
        predicted_weights[0].sum().item(),
    )
    print(
        "예측 비중 최솟값:",
        predicted_weights[0].min().item(),
    )
    print(
        "예측 비중 최댓값:",
        predicted_weights[0].max().item(),
    )

    print()
    print("=" * 60)
    print("Regret 확인")
    print("=" * 60)
    print("Predicted cost:", predicted_cost.detach())
    print("Oracle cost:", oracle_cost.detach())
    print("Regret:", regret.detach())


if __name__ == "__main__":
    main()