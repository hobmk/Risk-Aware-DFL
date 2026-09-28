from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from .capm import fit_capm
from .eigen_risk import ScaleMode, build_eigenmode_risk, covariance_matrix


ReturnType = Literal["simple", "log"]
MarketMode = Literal["equal_weight", "external"]


class EigenRCRDataset(Dataset):
    """
    Rolling Eigenmode RCR dataset.

    각 시점 t:
        X_t          = 과거 lookback일 수익률
        y_t          = 다음 거래일 수익률
        Sigma_t      = Cov(X_t)
        epsilon_t    = CAPM residual
        Sigma_eps,t  = Cov(epsilon_t)
        A_res,t^(K)  = Top-K residual eigenmodes
    """

    def __init__(
        self,
        price_csv: str | Path,
        lookback: int = 60,
        date_column: str = "Date",
        return_type: ReturnType = "simple",
        covariance_jitter: float = 1e-6,
        market_mode: MarketMode = "equal_weight",
        market_price_csv: str | Path | None = None,
        market_column: str | None = None,
        risk_free_rate: float = 0.0,
        fit_intercept: bool = True,
        eigen_k: int = 5,
        scale_mode: ScaleMode = "none",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()

        if lookback < 2:
            raise ValueError("lookback은 2 이상이어야 합니다.")
        if covariance_jitter < 0:
            raise ValueError("covariance_jitter는 0 이상이어야 합니다.")
        if return_type not in {"simple", "log"}:
            raise ValueError("return_type은 'simple' 또는 'log'여야 합니다.")
        if market_mode not in {"equal_weight", "external"}:
            raise ValueError("market_mode은 'equal_weight' 또는 'external'이어야 합니다.")
        if market_mode == "external" and market_price_csv is None:
            raise ValueError("external market을 사용할 경우 market_price_csv가 필요합니다.")
        if eigen_k <= 0:
            raise ValueError("eigen_k는 1 이상이어야 합니다.")

        self.price_csv = Path(price_csv)
        self.lookback = lookback
        self.date_column = date_column
        self.return_type = return_type
        self.covariance_jitter = covariance_jitter
        self.market_mode = market_mode
        self.market_price_csv = Path(market_price_csv) if market_price_csv else None
        self.market_column = market_column
        self.risk_free_rate = float(risk_free_rate)
        self.fit_intercept = fit_intercept
        self.eigen_k = eigen_k
        self.scale_mode = scale_mode
        self.dtype = dtype

        asset_prices = self._load_price_frame(self.price_csv, "asset")

        if market_mode == "external":
            market_prices = self._load_price_frame(self.market_price_csv, "market")
            market_series = self._select_market_series(market_prices)

            common_dates = asset_prices.index.intersection(market_series.index).sort_values()
            asset_prices = asset_prices.loc[common_dates]
            market_series = market_series.loc[common_dates]

            asset_returns = self._calculate_returns(asset_prices)
            market_returns = self._calculate_returns(
                market_series.to_frame("market")
            )["market"]

            self.market_name = str(market_series.name)

        else:
            asset_returns = self._calculate_returns(asset_prices)
            market_returns = asset_returns.mean(axis=1)
            market_returns.name = "equal_weight_market"
            self.market_name = "equal_weight_market"

        if not asset_returns.index.equals(market_returns.index):
            raise RuntimeError("자산 수익률과 시장 수익률 날짜가 일치하지 않습니다.")

        if len(asset_returns) <= lookback:
            raise ValueError("lookback 대비 수익률 데이터가 부족합니다.")

        self.tickers = list(asset_returns.columns)
        self.n_assets = len(self.tickers)

        if eigen_k > self.n_assets:
            raise ValueError(
                f"eigen_k는 자산 수 이하이어야 합니다: "
                f"k={eigen_k}, N={self.n_assets}"
            )

        self.returns = asset_returns.to_numpy(dtype=np.float64)
        self.market_returns = market_returns.to_numpy(dtype=np.float64)
        self.return_dates = pd.DatetimeIndex(asset_returns.index)

        self.target_positions = np.arange(
            lookback,
            len(asset_returns),
            dtype=np.int64,
        )

        self._precompute()

    def _load_price_frame(
        self,
        path: Path | None,
        label: str,
    ) -> pd.DataFrame:
        if path is None or not path.exists():
            raise FileNotFoundError(f"{label} CSV를 찾을 수 없습니다: {path}")

        df = pd.read_csv(path)

        if self.date_column not in df.columns:
            raise ValueError(f"날짜 열 '{self.date_column}'이 없습니다.")

        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.set_index(self.date_column).sort_index()

        if df.index.has_duplicates:
            raise ValueError(f"{label} CSV에 중복 날짜가 존재합니다.")

        df = df.apply(pd.to_numeric, errors="raise")

        if df.isna().any().any():
            raise ValueError(f"{label} 데이터에 결측치가 있습니다.")

        if (df <= 0).any().any():
            raise ValueError(f"{label} 가격 데이터에는 0 이하 값이 존재할 수 없습니다.")

        return df

    def _select_market_series(
        self,
        market_prices: pd.DataFrame,
    ) -> pd.Series:
        if self.market_column is not None:
            if self.market_column not in market_prices.columns:
                raise ValueError(
                    f"market_column='{self.market_column}'을 찾을 수 없습니다."
                )
            return market_prices[self.market_column]

        if market_prices.shape[1] != 1:
            raise ValueError(
                "시장 CSV에 여러 가격 열이 있습니다. "
                "--market-column을 지정하세요."
            )

        return market_prices.iloc[:, 0]

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
            raise ValueError("수익률 계산 후 NaN이 발생했습니다.")

        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError("수익률에 inf 또는 -inf가 존재합니다.")

        return returns

    def _precompute(self) -> None:
        asset_windows = np.lib.stride_tricks.sliding_window_view(
            self.returns,
            window_shape=self.lookback,
            axis=0,
        ).transpose(0, 2, 1)[:-1]

        market_windows = np.lib.stride_tricks.sliding_window_view(
            self.market_returns,
            window_shape=self.lookback,
        )[:-1]

        asset_windows = torch.from_numpy(
            np.ascontiguousarray(asset_windows)
        ).to(self.dtype)

        market_windows = torch.from_numpy(
            np.ascontiguousarray(market_windows)
        ).to(self.dtype)

        capm = fit_capm(
            asset_returns=asset_windows,
            market_returns=market_windows,
            risk_free_rate=self.risk_free_rate,
            fit_intercept=self.fit_intercept,
        )

        covariance = covariance_matrix(asset_windows)

        eye = torch.eye(
            self.n_assets,
            dtype=self.dtype,
        ).unsqueeze(0)

        covariance = covariance + self.covariance_jitter * eye

        eigen = build_eigenmode_risk(
            residuals=capm.residuals,
            k=self.eigen_k,
            scale_mode=self.scale_mode,
            base_covariance=covariance,
        )

        self.covariances = covariance.contiguous()
        self.capm_alpha = capm.alpha.contiguous()
        self.capm_beta = capm.beta.contiguous()
        self.residual_covariances = eigen.residual_covariance.contiguous()
        self.eigen_risk = eigen.risk_matrix.contiguous()
        self.eigenvalues = eigen.eigenvalues.contiguous()
        self.explained_ratio = eigen.explained_ratio.contiguous()
        self.cumulative_explained_ratio = (
            eigen.cumulative_explained_ratio.contiguous()
        )
        self.retained_ratio = eigen.retained_ratio.contiguous()

    def __len__(self) -> int:
        return len(self.target_positions)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:
        target_position = int(self.target_positions[index])
        start = target_position - self.lookback

        features = torch.tensor(
            self.returns[start:target_position],
            dtype=self.dtype,
        )

        target = torch.tensor(
            self.returns[target_position],
            dtype=self.dtype,
        )

        target_date = self.return_dates[target_position]

        return {
            "features": features,
            "target": target,
            "covariance": self.covariances[index],
            "residual_covariance": self.residual_covariances[index],
            "eigen_risk": self.eigen_risk[index],
            "eigenvalues": self.eigenvalues[index],
            "explained_ratio": self.explained_ratio[index],
            "cumulative_explained_ratio": self.cumulative_explained_ratio[index],
            "retained_ratio": self.retained_ratio[index],
            "target_date": target_date.strftime("%Y-%m-%d"),
        }

    @property
    def target_dates(self) -> pd.DatetimeIndex:
        return self.return_dates[self.target_positions]


def chronological_split(
    dataset: EigenRCRDataset,
    train_end: str,
    validation_end: str,
) -> tuple[Subset, Subset, Subset]:
    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)

    if train_end >= validation_end:
        raise ValueError("train_end는 validation_end보다 빨라야 합니다.")

    dates = dataset.target_dates

    train_indices = np.flatnonzero(
        dates <= train_end
    ).tolist()

    validation_indices = np.flatnonzero(
        (dates > train_end)
        & (dates <= validation_end)
    ).tolist()

    test_indices = np.flatnonzero(
        dates > validation_end
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