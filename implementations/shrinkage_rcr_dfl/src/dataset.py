from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from implementations.eigen_rcr_dfl.src.capm import (
    fit_capm,
)
from implementations.eigen_rcr_dfl.src.dataset import (
    EigenRCRDataset,
    MarketMode,
    ReturnType,
)
from implementations.eigen_rcr_dfl.src.eigen_risk import (
    ScaleMode,
    covariance_matrix,
)

from .shrinkage_risk import (
    build_shrinkage_risk,
)


class ShrinkageRCRDataset(EigenRCRDataset):
    """
    Adaptive Eigenvalue Shrinkage RCR dataset.

    기존 EigenRCRDataset의 데이터 로딩 / return 계산 /
    __getitem__ 구조를 그대로 재사용한다.

    차이:
        Top-K eigen_risk 대신
        adaptive shrinkage residual risk를 사용한다.
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
        shrinkage_c: float = 0.5,
        scale_mode: ScaleMode = "none",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if shrinkage_c < 0:
            raise ValueError(
                "shrinkage_c는 0 이상이어야 합니다."
            )

        self.shrinkage_c = float(
            shrinkage_c
        )

        # 부모 클래스의 데이터 로딩 구조를 그대로 사용한다.
        # eigen_k는 Shrinkage에서는 사용하지 않지만,
        # 부모 validation을 통과시키기 위해 1로 전달한다.
        super().__init__(
            price_csv=price_csv,
            lookback=lookback,
            date_column=date_column,
            return_type=return_type,
            covariance_jitter=covariance_jitter,
            market_mode=market_mode,
            market_price_csv=market_price_csv,
            market_column=market_column,
            risk_free_rate=risk_free_rate,
            fit_intercept=fit_intercept,
            eigen_k=1,
            scale_mode=scale_mode,
            dtype=dtype,
        )

        # Shrinkage는 모든 N개 eigenmode를 사용한다.
        self.eigen_k = self.n_assets

    def _precompute(self) -> None:
        asset_windows = np.lib.stride_tricks.sliding_window_view(
            self.returns,
            window_shape=self.lookback,
            axis=0,
        ).transpose(
            0,
            2,
            1,
        )[:-1]

        market_windows = np.lib.stride_tricks.sliding_window_view(
            self.market_returns,
            window_shape=self.lookback,
        )[:-1]

        asset_windows = torch.from_numpy(
            np.ascontiguousarray(
                asset_windows
            )
        ).to(
            self.dtype
        )

        market_windows = torch.from_numpy(
            np.ascontiguousarray(
                market_windows
            )
        ).to(
            self.dtype
        )

        capm = fit_capm(
            asset_returns=asset_windows,
            market_returns=market_windows,
            risk_free_rate=self.risk_free_rate,
            fit_intercept=self.fit_intercept,
        )

        covariance = covariance_matrix(
            asset_windows
        )

        eye = torch.eye(
            self.n_assets,
            dtype=self.dtype,
        ).unsqueeze(0)

        covariance = (
            covariance
            + self.covariance_jitter * eye
        )

        shrinkage = build_shrinkage_risk(
            residuals=capm.residuals,
            shrinkage_c=self.shrinkage_c,
            scale_mode=self.scale_mode,
            base_covariance=covariance,
        )

        self.covariances = (
            covariance.contiguous()
        )

        self.capm_alpha = (
            capm.alpha.contiguous()
        )

        self.capm_beta = (
            capm.beta.contiguous()
        )

        self.residual_covariances = (
            shrinkage
            .residual_covariance
            .contiguous()
        )

        # 기존 EigenRCRDFL model과 trainer가
        # 그대로 사용할 수 있도록 이름을 eigen_risk로 유지.
        self.eigen_risk = (
            shrinkage
            .risk_matrix
            .contiguous()
        )

        self.eigenvalues = (
            shrinkage
            .eigenvalues
            .contiguous()
        )

        self.explained_ratio = (
            shrinkage
            .explained_ratio
            .contiguous()
        )

        self.cumulative_explained_ratio = (
            shrinkage
            .cumulative_explained_ratio
            .contiguous()
        )

        self.retained_ratio = (
            shrinkage
            .retained_ratio
            .contiguous()
        )

        self.shrunk_eigenvalues = (
            shrinkage
            .shrunk_eigenvalues
            .contiguous()
        )

        self.shrinkage_factors = (
            shrinkage
            .shrinkage_factors
            .contiguous()
        )

        self.shrinkage_tau = (
            shrinkage
            .tau
            .contiguous()
        )

    def __getitem__(
        self,
        index: int,
    ):
        item = super().__getitem__(
            index
        )

        item[
            "shrunk_eigenvalues"
        ] = self.shrunk_eigenvalues[index]

        item[
            "shrinkage_factors"
        ] = self.shrinkage_factors[index]

        item[
            "shrinkage_tau"
        ] = self.shrinkage_tau[index]

        return item