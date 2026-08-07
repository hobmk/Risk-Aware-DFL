from pathlib import Path

import numpy as np
import pandas as pd
import torch

from implementations.rcr_dfl.src.dataset import (
    FeatureStandardizedSubset,
    RCRRollingMVODataset,
    chronological_split,
    fit_feature_standardizer,
)


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
        residual_correlation_shrinkage=0.0,
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



def test_feature_standardization_uses_train_data_only(
    tmp_path: Path,
) -> None:
    asset_csv, market_csv = _write_synthetic_csvs(tmp_path)

    dataset = RCRRollingMVODataset(
        price_csv=asset_csv,
        market_mode="external",
        market_price_csv=market_csv,
        market_column="INDEX",
        lookback=4,
        residual_correlation_shrinkage=0.0,
        correlation_scaling="trace",
    )

    train, validation, test = chronological_split(
        dataset,
        train_end=str(dataset.target_dates[2].date()),
        validation_end=str(dataset.target_dates[5].date()),
    )

    feature_mean, feature_std = fit_feature_standardizer(
        dataset,
        train,
    )

    feature_positions: list[int] = []

    for sample_index in train.indices:
        target_position = int(
            dataset.target_positions[int(sample_index)]
        )
        feature_positions.extend(
            range(
                target_position - dataset.lookback,
                target_position,
            )
        )

    unique_positions = np.unique(
        np.asarray(feature_positions, dtype=np.int64)
    )
    expected_features = dataset.returns[unique_positions]

    expected_mean = torch.tensor(
        expected_features.mean(axis=0),
        dtype=dataset.dtype,
    )
    expected_std = torch.tensor(
        expected_features.std(axis=0, ddof=0),
        dtype=dataset.dtype,
    )

    assert torch.allclose(feature_mean, expected_mean, atol=1e-12)
    assert torch.allclose(feature_std, expected_std, atol=1e-12)

    standardized_train = FeatureStandardizedSubset(
        train,
        feature_mean,
        feature_std,
    )

    raw_sample = train[0]
    standardized_sample = standardized_train[0]

    assert torch.allclose(
        standardized_sample["features"],
        (
            raw_sample["features"] - feature_mean
        ) / feature_std,
        atol=1e-12,
    )

    for key in (
        "target",
        "covariance",
        "residuals",
        "residual_correlation_raw",
        "residual_correlation",
        "a_res",
    ):
        assert torch.allclose(
            standardized_sample[key],
            raw_sample[key],
            atol=1e-12,
        )

    assert (
        standardized_sample["target_date"]
        == raw_sample["target_date"]
    )
    assert len(validation) == 3
    assert len(test) == 2
