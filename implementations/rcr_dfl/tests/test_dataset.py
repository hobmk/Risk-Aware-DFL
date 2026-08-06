from pathlib import Path

import numpy as np
import pandas as pd
import torch

from implementations.rcr_dfl.src.dataset import RCRRollingMVODataset, chronological_split


def _prices_from_returns(returns: np.ndarray, initial: float = 100.0) -> np.ndarray:
    return np.concatenate([[initial], initial * np.cumprod(1.0 + returns)])


def _write_synthetic_csvs(tmp_path: Path) -> tuple[Path, Path]:
    dates = pd.date_range("2020-01-01", periods=13, freq="D")
    market = np.array([0.01, -0.005, 0.004, 0.002, -0.003, 0.006, 0.001, -0.002, 0.005, 0.003, -0.004, 0.002])
    noise_1 = np.array([0.001, -0.001, 0.0005, 0.0, -0.0005, 0.001, -0.001, 0.0004, 0.0, 0.0006, -0.0004, 0.0002])
    noise_2 = np.array([-0.0005, 0.0007, -0.0003, 0.0004, 0.0, -0.0006, 0.0008, -0.0002, 0.0005, -0.0004, 0.0003, -0.0001])
    asset_csv = tmp_path / "assets.csv"
    market_csv = tmp_path / "market.csv"
    pd.DataFrame({
        "Date": dates,
        "A": _prices_from_returns(0.0005 + 0.8 * market + noise_1),
        "B": _prices_from_returns(-0.0002 + 1.2 * market + noise_2),
    }).to_csv(asset_csv, index=False)
    pd.DataFrame({"Date": dates, "INDEX": _prices_from_returns(market)}).to_csv(market_csv, index=False)
    return asset_csv, market_csv


def test_external_market_dataset_generates_residual_correlation(tmp_path: Path) -> None:
    asset_csv, market_csv = _write_synthetic_csvs(tmp_path)
    dataset = RCRRollingMVODataset(
        price_csv=asset_csv,
        market_mode="external",
        market_price_csv=market_csv,
        market_column="INDEX",
        lookback=4,
        covariance_jitter=1e-6,
        residual_correlation_shrinkage=0.1,
        correlation_scaling="trace",
    )
    sample = dataset[0]
    assert len(dataset) == 8
    assert sample["residual_correlation_raw"].shape == (2, 2)
    assert sample["residual_correlation"].shape == (2, 2)
    assert sample["a_res"].shape == (2, 2)
    assert torch.allclose(torch.diagonal(sample["residual_correlation"]), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(torch.trace(sample["a_res"]), torch.trace(sample["covariance"]), atol=1e-12)
    assert sample["residuals"].mean(dim=0).abs().max() < 1e-12


def test_feature_target_time_alignment(tmp_path: Path) -> None:
    asset_csv, market_csv = _write_synthetic_csvs(tmp_path)
    dataset = RCRRollingMVODataset(
        price_csv=asset_csv,
        market_mode="external",
        market_price_csv=market_csv,
        market_column="INDEX",
        lookback=4,
    )
    sample = dataset[0]
    assert torch.allclose(sample["features"], torch.tensor(dataset.returns[:4], dtype=torch.float64))
    assert torch.allclose(sample["target"], torch.tensor(dataset.returns[4], dtype=torch.float64))


def test_chronological_split(tmp_path: Path) -> None:
    asset_csv, market_csv = _write_synthetic_csvs(tmp_path)
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
    assert (len(train), len(validation), len(test)) == (3, 3, 2)
