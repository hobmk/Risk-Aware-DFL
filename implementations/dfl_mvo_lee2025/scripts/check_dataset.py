from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset,
    chronological_split,
)

PRICE_CSV = PROJECT_ROOT / "data/raw/dow30_adjusted_close.csv"


def main():
    dataset = RollingMVODataset(
        price_csv=PRICE_CSV,
        lookback=60,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float64,
    )

    train_set, val_set, test_set = chronological_split(
        dataset,
        train_end="2021-12-31",
        validation_end="2022-12-31",
    )

    sample = dataset[0]

    print("가격 파일:", PRICE_CSV)
    print("자산 수:", dataset.n_assets)
    print("전체 샘플:", len(dataset))
    print("Train / Val / Test:", len(train_set), len(val_set), len(test_set))

    print("features:", sample["features"].shape)
    print("target:", sample["target"].shape)
    print("covariance:", sample["covariance"].shape)
    print("target date:", sample["target_date"])

    eigenvalues = torch.linalg.eigvalsh(sample["covariance"])
    print("공분산 최소 고유값:", eigenvalues.min().item())

    loader = DataLoader(train_set, batch_size=16, shuffle=False)
    batch = next(iter(loader))

    print("batch features:", batch["features"].shape)
    print("batch target:", batch["target"].shape)
    print("batch covariance:", batch["covariance"].shape)


if __name__ == "__main__":
    main()