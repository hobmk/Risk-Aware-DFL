from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


ReturnType = Literal["simple", "log"]


class RollingMVODataset(Dataset):
    """
    과거 lookback일의 자산 수익률을 입력으로 사용하여
    다음 거래일의 자산별 수익률을 예측하기 위한 Dataset.

    각 sample이 반환하는 값
    -------------------------
    features:
        과거 lookback일 수익률, shape [lookback, n_assets]

    target:
        다음 거래일의 자산별 수익률, shape [n_assets]

    covariance:
        과거 lookback일 수익률로 계산한 표본공분산행렬,
        shape [n_assets, n_assets]

    target_date:
        target 수익률이 발생한 날짜
    """

    def __init__(
        self,
        price_csv: str | Path,
        lookback: int = 60,
        date_column: str = "Date",
        return_type: ReturnType = "simple",
        covariance_jitter: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()

        if lookback < 2:
            raise ValueError("lookback은 2 이상이어야 합니다.")

        if covariance_jitter < 0:
            raise ValueError("covariance_jitter는 음수가 될 수 없습니다.")

        if return_type not in {"simple", "log"}:
            raise ValueError(
                "return_type은 'simple' 또는 'log'여야 합니다."
            )

        self.price_csv = Path(price_csv)
        self.lookback = lookback
        self.date_column = date_column
        self.return_type = return_type
        self.covariance_jitter = covariance_jitter
        self.dtype = dtype

        prices = self._load_prices()
        returns = self._calculate_returns(prices)

        if len(returns) <= lookback:
            raise ValueError(
                f"수익률 관측치가 부족합니다. "
                f"관측치={len(returns)}, lookback={lookback}"
            )

        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)

        # Dataset 내부에서는 NumPy 배열로 보관한다.
        # shape: [n_dates, n_assets]
        self.returns = returns.to_numpy(dtype=np.float64)

        # 수익률 날짜와 target 날짜 확인에 사용한다.
        self.return_dates = pd.DatetimeIndex(returns.index)

        # position=t이면:
        # features = returns[t-lookback:t]
        # target   = returns[t]
        self.target_positions = np.arange(
            lookback,
            len(returns),
            dtype=np.int64,
        )

    def _load_prices(self) -> pd.DataFrame:
        if not self.price_csv.exists():
            raise FileNotFoundError(
                f"가격 CSV를 찾을 수 없습니다: {self.price_csv}"
            )

        dataframe = pd.read_csv(self.price_csv)

        if self.date_column not in dataframe.columns:
            raise ValueError(
                f"CSV에 날짜 열 '{self.date_column}'이 없습니다."
            )

        dataframe[self.date_column] = pd.to_datetime(
            dataframe[self.date_column],
            errors="raise",
        )

        dataframe = dataframe.set_index(self.date_column)
        dataframe = dataframe.sort_index()

        if dataframe.index.has_duplicates:
            duplicated_dates = dataframe.index[
                dataframe.index.duplicated()
            ].unique()

            raise ValueError(
                "중복 날짜가 존재합니다: "
                f"{duplicated_dates[:5].tolist()}"
            )

        if dataframe.shape[1] < 2:
            raise ValueError(
                "포트폴리오 실험을 위해 최소 2개 자산이 필요합니다."
            )

        # 문자열 등이 섞여 있을 경우 명확하게 오류를 발생시킨다.
        dataframe = dataframe.apply(
            pd.to_numeric,
            errors="raise",
        )

        missing_count = int(dataframe.isna().sum().sum())

        if missing_count > 0:
            raise ValueError(
                f"가격 데이터에 결측치가 {missing_count}개 있습니다. "
                "Dataset 내부에서 임의로 채우지 않고, "
                "데이터 전처리 단계에서 처리해야 합니다."
            )

        if (dataframe <= 0).any().any():
            raise ValueError(
                "가격 데이터에는 0 이하의 값이 포함될 수 없습니다."
            )

        return dataframe

    def _calculate_returns(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.return_type == "simple":
            returns = prices.pct_change(fill_method=None)
        else:
            returns = np.log(prices / prices.shift(1))

        returns = returns.iloc[1:]

        if returns.isna().any().any():
            raise ValueError(
                "수익률 계산 후 결측치가 발생했습니다. "
                "원본 가격 데이터의 정렬 및 결측치를 확인하세요."
            )

        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError(
                "수익률 데이터에 inf 또는 -inf가 존재합니다."
            )

        return returns

    def __len__(self) -> int:
        return len(self.target_positions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        target_position = int(self.target_positions[index])

        start_position = target_position - self.lookback
        end_position = target_position

        # target 날짜 직전까지의 60일 수익률
        feature_window = self.returns[
            start_position:end_position
        ]

        # 바로 다음 거래일 수익률
        target_return = self.returns[target_position]

        covariance = np.cov(
            feature_window,
            rowvar=False,
            ddof=1,
        )

        # 수치오차로 인한 비대칭 제거
        covariance = 0.5 * (
            covariance + covariance.T
        )

        # Cholesky 분해와 convex solver 안정화를 위한 diagonal jitter
        covariance = covariance + (
            self.covariance_jitter
            * np.eye(self.n_assets, dtype=np.float64)
        )

        target_date = self.return_dates[target_position]

        return {
            "features": torch.tensor(
                feature_window,
                dtype=self.dtype,
            ),
            "target": torch.tensor(
                target_return,
                dtype=self.dtype,
            ),
            "covariance": torch.tensor(
                covariance,
                dtype=self.dtype,
            ),
            "target_date": target_date.strftime("%Y-%m-%d"),
        }

    @property
    def target_dates(self) -> pd.DatetimeIndex:
        return self.return_dates[self.target_positions]


def chronological_split(
    dataset: RollingMVODataset,
    train_end: str,
    validation_end: str,
) -> tuple[Subset, Subset, Subset]:
    """
    target 날짜를 기준으로 Dataset을 시간순 분할한다.

    train:
        target_date <= train_end

    validation:
        train_end < target_date <= validation_end

    test:
        target_date > validation_end
    """

    train_end_timestamp = pd.Timestamp(train_end)
    validation_end_timestamp = pd.Timestamp(validation_end)

    if train_end_timestamp >= validation_end_timestamp:
        raise ValueError(
            "train_end는 validation_end보다 빨라야 합니다."
        )

    target_dates = dataset.target_dates

    train_indices = np.flatnonzero(
        target_dates <= train_end_timestamp
    ).tolist()

    validation_indices = np.flatnonzero(
        (target_dates > train_end_timestamp)
        & (target_dates <= validation_end_timestamp)
    ).tolist()

    test_indices = np.flatnonzero(
        target_dates > validation_end_timestamp
    ).tolist()

    if not train_indices:
        raise ValueError("Train sample이 없습니다.")

    if not validation_indices:
        raise ValueError("Validation sample이 없습니다.")

    if not test_indices:
        raise ValueError("Test sample이 없습니다.")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, validation_indices),
        Subset(dataset, test_indices),
    )
class FeatureStandardizedSubset(Dataset):
    """Subset의 features만 표준화하고 target과 covariance는 유지한다."""

    def __init__(
        self,
        subset: Subset,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
        self.subset = subset
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = dict(self.subset[index])
        sample["features"] = (
            sample["features"] - self.feature_mean
        ) / self.feature_std
        return sample


def fit_feature_standardizer(
    dataset: RollingMVODataset,
    train_subset: Subset,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train 입력에 실제 사용되는 수익률만으로 종목별 평균·표준편차를 계산한다."""

    feature_positions: list[int] = []

    for sample_index in train_subset.indices:
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
    train_features = dataset.returns[unique_positions]

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0, ddof=0)

    if not np.isfinite(feature_mean).all():
        raise ValueError("입력 표준화 평균에 비정상 값이 있습니다.")

    if not np.isfinite(feature_std).all():
        raise ValueError("입력 표준화 표준편차에 비정상 값이 있습니다.")

    feature_std = np.where(
        feature_std < eps,
        1.0,
        feature_std,
    )

    return (
        torch.tensor(feature_mean, dtype=dataset.dtype),
        torch.tensor(feature_std, dtype=dataset.dtype),
    )