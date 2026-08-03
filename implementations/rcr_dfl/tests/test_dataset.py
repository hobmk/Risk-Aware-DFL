from pathlib import Path

import numpy as np
import pandas as pd
import torch

from implementations.dfl_mvo_lee2025.src.dataset import RollingMVODataset
from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset, chronological_split


def _prices_from_returns(returns: np.ndarray, initial: float = 100.0) -> np.ndarray:
    return np.concatenate([[initial], initial * np.cumprod(1.0 + returns)])


def _write_synthetic_csvs(tmp_path: Path) -> tuple[Path, Path, pd.DatetimeIndex]:
    dates = pd.date_range("2020-01-01", periods=13, freq="D")
    market_returns = np.array([0.01, -0.005, 0.004, 0.002, -0.003, 0.006, 0.001, -0.002, 0.005, 0.003, -0.004, 0.002])
    asset_1 = 0.0005 + 0.8 * market_returns + np.array([0.001, -0.001, 0.0005, 0.0, -0.0005, 0.001, -0.001, 0.0004, 0.0, 0.0006, -0.0004, 0.0002])
    asset_2 = -0.0002 + 1.2 * market_returns + np.array([-0.0005, 0.0007, -0.0003, 0.0004, 0.0, -0.0006, 0.0008, -0.0002, 0.0005, -0.0004, 0.0003, -0.0001])

    asset_csv = tmp_path / "assets.csv"
    market_csv = tmp_path / "market.csv"
    pd.DataFrame({
        "Date": dates,
        "A": _prices_from_returns(asset_1),
        "B": _prices_from_returns(asset_2),
    }).to_csv(asset_csv, index=False)
    pd.DataFrame({
        "Date": dates,
        "INDEX": _prices_from_returns(market_returns),
    }).to_csv(market_csv, index=False)
    return asset_csv, market_csv, dates


def test_external_market_dataset_alignment_and_shapes(tmp_path: Path) -> None:
    asset_csv, market_csv, _ = _write_synthetic_csvs(tmp_path)
    dataset = RCRRollingMVODataset(
        price_csv=asset_csv,
        market_mode="external",
        market_price_csv=market_csv,
        market_column="INDEX",
        lookback=4,
        covariance_jitter=1e-6,
    )
    sample = dataset[0]

    assert len(dataset) == 8
    assert sample["features"].shape == (4, 2)
    assert sample["target"].shape == (2,)
    assert sample["market_window"].shape == (4,)
    assert sample["residuals"].shape == (4, 2)
    assert sample["covariance"].shape == (2, 2)
    assert sample["residual_covariance"].shape == (2, 2)
    assert torch.allclose(sample["covariance"], sample["covariance"].T, atol=1e-12)
    assert torch.allclose(sample["residual_covariance"], sample["residual_covariance"].T, atol=1e-12)
    assert sample["residuals"].mean(dim=0).abs().max() < 1e-12


def test_feature_target_time_alignment(tmp_path: Path) -> None:
    asset_csv, market_csv, _ = _write_synthetic_csvs(tmp_path)
    dataset = RCRRollingMVODataset(
        price_csv=asset_csv,
        market_mode="external",
        market_price_csv=market_csv,
        market_column="INDEX",
        lookback=4,
    )
    sample = dataset[0]

    expected_features = torch.tensor(dataset.returns[:4], dtype=torch.float64)
    expected_target = torch.tensor(dataset.returns[4], dtype=torch.float64)
    assert torch.allclose(sample["features"], expected_features, atol=1e-12)
    assert torch.allclose(sample["target"], expected_target, atol=1e-12)


def test_equal_weight_mode_matches_benchmark_inputs(tmp_path: Path) -> None:
    asset_csv, _, _ = _write_synthetic_csvs(tmp_path)
    benchmark = RollingMVODataset(price_csv=asset_csv, lookback=4, covariance_jitter=1e-6)
    rcr = RCRRollingMVODataset(
        price_csv=asset_csv,
        market_mode="equal_weight",
        lookback=4,
        covariance_jitter=1e-6,
    )

    for index in [0, len(rcr) // 2, len(rcr) - 1]:
        benchmark_sample = benchmark[index]
        rcr_sample = rcr[index]
        assert torch.allclose(rcr_sample["features"], benchmark_sample["features"], atol=1e-12)
        assert torch.allclose(rcr_sample["target"], benchmark_sample["target"], atol=1e-12)
        assert torch.allclose(rcr_sample["covariance"], benchmark_sample["covariance"], atol=1e-12)
        assert rcr_sample["target_date"] == benchmark_sample["target_date"]


def test_chronological_split(tmp_path: Path) -> None:
    asset_csv, market_csv, _ = _write_synthetic_csvs(tmp_path)
    dataset = RCRRollingMVODataset(
        price_csv=asset_csv,
        market_mode="external",
        market_price_csv=market_csv,
        market_column="INDEX",
        lookback=4,
    )
    train, validation, test = chronological_split(
        dataset,
        train_end=str(dataset.target_dates[2].date()),
        validation_end=str(dataset.target_dates[5].date()),
    )

    assert len(train) == 3
    assert len(validation) == 3
    assert len(test) == 2
