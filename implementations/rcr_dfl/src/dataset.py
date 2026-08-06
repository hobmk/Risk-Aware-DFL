from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from .capm import fit_capm
from .residual_risk import (
    CorrelationScaling,
    correlation_matrix,
    covariance_matrix,
    scale_correlation_to_covariance,
    shrink_correlation,
)

ReturnType = Literal["simple", "log"]
MarketMode = Literal["equal_weight", "external"]


class RCRRollingMVODataset(Dataset):
    """
    기존 RollingMVODataset 정렬을 유지하면서 rolling CAPM과 residual correlation을 생성한다.

    각 시점 t에서:
        C_res,t = Corr(epsilon_t)
        C_bar,t = (1-rho) C_res,t + rho I
        A_res,t = tr(Sigma_t) / N * C_bar,t       (correlation_scaling="trace")

    주요 반환값:
        features: [lookback, N]
        target: [N]
        covariance: Sigma_t, [N, N]
        residual_correlation_raw: C_res,t, [N, N]
        residual_correlation: C_bar,t, [N, N]
        a_res: covariance 단위로 변환된 A_res,t, [N, N]
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
        residual_correlation_shrinkage: float = 0.1,
        correlation_scaling: CorrelationScaling = "trace",
        correlation_eps: float = 1e-12,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if lookback < 2:
            raise ValueError("lookback은 2 이상이어야 합니다.")
        if covariance_jitter < 0:
            raise ValueError("covariance_jitter는 음수가 될 수 없습니다.")
        if return_type not in {"simple", "log"}:
            raise ValueError("return_type은 'simple' 또는 'log'여야 합니다.")
        if market_mode not in {"equal_weight", "external"}:
            raise ValueError("market_mode는 'equal_weight' 또는 'external'이어야 합니다.")
        if market_mode == "external" and market_price_csv is None:
            raise ValueError("market_mode='external'이면 market_price_csv가 필요합니다.")
        if not np.isfinite(risk_free_rate):
            raise ValueError("risk_free_rate는 유한한 값이어야 합니다.")
        if not 0.0 <= residual_correlation_shrinkage <= 1.0:
            raise ValueError("residual_correlation_shrinkage는 0 이상 1 이하여야 합니다.")
        if correlation_scaling not in {"none", "trace"}:
            raise ValueError("correlation_scaling은 'none' 또는 'trace'여야 합니다.")
        if correlation_eps <= 0:
            raise ValueError("correlation_eps는 0보다 커야 합니다.")

        self.price_csv = Path(price_csv)
        self.lookback = lookback
        self.date_column = date_column
        self.return_type = return_type
        self.covariance_jitter = covariance_jitter
        self.market_mode = market_mode
        self.market_price_csv = Path(market_price_csv) if market_price_csv is not None else None
        self.market_column = market_column
        self.risk_free_rate = float(risk_free_rate)
        self.fit_intercept = fit_intercept
        self.residual_correlation_shrinkage = residual_correlation_shrinkage
        self.correlation_scaling = correlation_scaling
        self.correlation_eps = correlation_eps
        self.dtype = dtype

        asset_prices = self._load_price_frame(self.price_csv, minimum_columns=2, label="자산")
        if self.market_mode == "external":
            market_prices = self._load_price_frame(self.market_price_csv, minimum_columns=1, label="시장")
            market_series = self._select_market_series(market_prices)
            common_dates = asset_prices.index.intersection(market_series.index).sort_values()
            if len(common_dates) < lookback + 2:
                raise ValueError(
                    "자산과 시장 가격의 공통 날짜가 부족합니다. "
                    f"공통 가격 관측치={len(common_dates)}, lookback={lookback}"
                )
            asset_prices = asset_prices.loc[common_dates]
            market_series = market_series.loc[common_dates]
            asset_returns = self._calculate_returns(asset_prices)
            market_returns = self._calculate_returns(market_series.to_frame("market"))["market"]
            self.market_name = str(market_series.name)
        else:
            asset_returns = self._calculate_returns(asset_prices)
            market_returns = asset_returns.mean(axis=1)
            market_returns.name = "equal_weight_market"
            self.market_name = "equal_weight_market"

        if not asset_returns.index.equals(market_returns.index):
            raise RuntimeError("자산 수익률과 시장수익률의 날짜 정렬에 실패했습니다.")
        if len(asset_returns) <= lookback:
            raise ValueError(
                "수익률 관측치가 부족합니다. "
                f"관측치={len(asset_returns)}, lookback={lookback}"
            )

        self.tickers = list(asset_returns.columns)
        self.n_assets = len(self.tickers)
        self.returns = asset_returns.to_numpy(dtype=np.float64)
        self.market_returns = market_returns.to_numpy(dtype=np.float64)
        self.return_dates = pd.DatetimeIndex(asset_returns.index)
        self.target_positions = np.arange(lookback, len(asset_returns), dtype=np.int64)
        self._precompute_risk_inputs()

    def _load_price_frame(self, path: Path | None, minimum_columns: int, label: str) -> pd.DataFrame:
        if path is None or not path.exists():
            raise FileNotFoundError(f"{label} 가격 CSV를 찾을 수 없습니다: {path}")
        dataframe = pd.read_csv(path)
        if self.date_column not in dataframe.columns:
            raise ValueError(f"{label} CSV에 날짜 열 '{self.date_column}'이 없습니다.")
        dataframe[self.date_column] = pd.to_datetime(dataframe[self.date_column], errors="raise")
        dataframe = dataframe.set_index(self.date_column).sort_index()
        if dataframe.index.has_duplicates:
            duplicated = dataframe.index[dataframe.index.duplicated()].unique()
            raise ValueError(f"{label} 가격 CSV에 중복 날짜가 존재합니다: {duplicated[:5].tolist()}")
        if dataframe.shape[1] < minimum_columns:
            raise ValueError(f"{label} 가격 CSV에는 최소 {minimum_columns}개 가격 열이 필요합니다.")
        dataframe = dataframe.apply(pd.to_numeric, errors="raise")
        missing_count = int(dataframe.isna().sum().sum())
        if missing_count > 0:
            raise ValueError(f"{label} 가격 데이터에 결측치가 {missing_count}개 있습니다.")
        if (dataframe <= 0).any().any():
            raise ValueError(f"{label} 가격 데이터에는 0 이하의 값이 포함될 수 없습니다.")
        return dataframe

    def _select_market_series(self, market_prices: pd.DataFrame) -> pd.Series:
        if self.market_column is None:
            if market_prices.shape[1] != 1:
                raise ValueError(
                    "시장 CSV에 가격 열이 여러 개입니다. market_column을 명시하세요. "
                    f"columns={market_prices.columns.tolist()}"
                )
            column = str(market_prices.columns[0])
        else:
            column = self.market_column
            if column not in market_prices.columns:
                raise ValueError(
                    f"시장 CSV에 market_column='{column}'이 없습니다. "
                    f"columns={market_prices.columns.tolist()}"
                )
        return market_prices[column].rename(column)

    def _calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change(fill_method=None) if self.return_type == "simple" else np.log(prices / prices.shift(1))
        returns = returns.iloc[1:]
        if returns.isna().any().any():
            raise ValueError("수익률 계산 후 결측치가 발생했습니다.")
        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError("수익률 데이터에 inf 또는 -inf가 존재합니다.")
        return returns

    def _precompute_risk_inputs(self) -> None:
        asset_windows = np.lib.stride_tricks.sliding_window_view(
            self.returns, window_shape=self.lookback, axis=0
        ).transpose(0, 2, 1)[:-1]
        market_windows = np.lib.stride_tricks.sliding_window_view(
            self.market_returns, window_shape=self.lookback
        )[:-1]
        asset_windows_tensor = torch.from_numpy(np.ascontiguousarray(asset_windows)).to(self.dtype)
        market_windows_tensor = torch.from_numpy(np.ascontiguousarray(market_windows)).to(self.dtype)

        capm = fit_capm(
            asset_returns=asset_windows_tensor,
            market_returns=market_windows_tensor,
            risk_free_rates=self.risk_free_rate,
            fit_intercept=self.fit_intercept,
        )
        covariance = covariance_matrix(asset_windows_tensor)
        eye = torch.eye(self.n_assets, dtype=self.dtype).unsqueeze(0)
        covariance = covariance + self.covariance_jitter * eye

        residual_correlation_raw = correlation_matrix(
            capm.residuals, eps=self.correlation_eps
        )
        residual_correlation = shrink_correlation(
            residual_correlation_raw,
            shrinkage=self.residual_correlation_shrinkage,
        )
        a_res = scale_correlation_to_covariance(
            correlation=residual_correlation,
            reference_covariance=covariance,
            scaling=self.correlation_scaling,
        )

        self.capm_alpha = capm.alpha.contiguous()
        self.capm_beta = capm.beta.contiguous()
        self.covariances = covariance.contiguous()
        self.residual_correlations_raw = residual_correlation_raw.contiguous()
        self.residual_correlations = residual_correlation.contiguous()
        self.a_res_matrices = a_res.contiguous()

    def __len__(self) -> int:
        return len(self.target_positions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        target_position = int(self.target_positions[index])
        start_position = target_position - self.lookback
        features = torch.tensor(self.returns[start_position:target_position], dtype=self.dtype)
        target = torch.tensor(self.returns[target_position], dtype=self.dtype)
        market_window = torch.tensor(self.market_returns[start_position:target_position], dtype=self.dtype)
        alpha = self.capm_alpha[index]
        beta = self.capm_beta[index]
        asset_excess = features - self.risk_free_rate
        market_excess = market_window - self.risk_free_rate
        residuals = asset_excess - alpha.unsqueeze(0) - market_excess.unsqueeze(-1) * beta.unsqueeze(0)
        target_date = self.return_dates[target_position]
        return {
            "features": features,
            "target": target,
            "covariance": self.covariances[index],
            "market_window": market_window,
            "capm_alpha": alpha,
            "capm_beta": beta,
            "residuals": residuals,
            "residual_correlation_raw": self.residual_correlations_raw[index],
            "residual_correlation": self.residual_correlations[index],
            "a_res": self.a_res_matrices[index],
            "target_date": target_date.strftime("%Y-%m-%d"),
        }

    @property
    def target_dates(self) -> pd.DatetimeIndex:
        return self.return_dates[self.target_positions]


def chronological_split(
    dataset: RCRRollingMVODataset,
    train_end: str,
    validation_end: str,
) -> tuple[Subset, Subset, Subset]:
    """target 날짜를 기준으로 train/validation/test를 시간순 분할한다."""
    train_end_timestamp = pd.Timestamp(train_end)
    validation_end_timestamp = pd.Timestamp(validation_end)
    if train_end_timestamp >= validation_end_timestamp:
        raise ValueError("train_end는 validation_end보다 빨라야 합니다.")
    target_dates = dataset.target_dates
    train_indices = np.flatnonzero(target_dates <= train_end_timestamp).tolist()
    validation_indices = np.flatnonzero(
        (target_dates > train_end_timestamp) & (target_dates <= validation_end_timestamp)
    ).tolist()
    test_indices = np.flatnonzero(target_dates > validation_end_timestamp).tolist()
    if not train_indices:
        raise ValueError("Train sample이 없습니다.")
    if not validation_indices:
        raise ValueError("Validation sample이 없습니다.")
    if not test_indices:
        raise ValueError("Test sample이 없습니다.")
    return Subset(dataset, train_indices), Subset(dataset, validation_indices), Subset(dataset, test_indices)
