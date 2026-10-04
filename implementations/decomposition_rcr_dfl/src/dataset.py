from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from implementations.eigen_rcr_dfl.src.capm import fit_capm
from implementations.eigen_rcr_dfl.src.dataset import (
    EigenRCRDataset,
    MarketMode,
    ReturnType,
)

from .decomposition_risk import build_decomposition_risk


class DecompositionRCRDataset(EigenRCRDataset):
    """
    Residual Risk Decomposition & Reconstruction dataset.

    Sigma_eff^(K)
        = Sigma_market
        + U_K Lambda_K U_K^T
        + D_K
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
        decomposition_k: int = 3,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if decomposition_k < 0:
            raise ValueError(
                "decomposition_k는 0 이상이어야 합니다."
            )

        self.decomposition_k = decomposition_k

        # 부모 클래스의 loading / return 계산 /
        # target position 생성 구조를 재사용.
        #
        # 부모는 eigen_k >= 1을 요구하므로 임시로 1 전달.
        # 실제 risk 계산은 override한 _precompute에서 수행.
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
            scale_mode="none",
            dtype=dtype,
        )

        if decomposition_k > self.n_assets:
            raise ValueError(
                f"decomposition_k는 자산 수 이하이어야 합니다: "
                f"k={decomposition_k}, N={self.n_assets}"
            )

        self.eigen_k = decomposition_k

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

        result = build_decomposition_risk(
            asset_returns=asset_windows,
            market_returns=market_windows,
            beta=capm.beta,
            residuals=capm.residuals,
            k=self.decomposition_k,
            covariance_jitter=self.covariance_jitter,
        )

        self.base_covariances = (
            result.base_covariance.contiguous()
        )

        self.market_covariances = (
            result.market_covariance.contiguous()
        )

        self.residual_covariances = (
            result.residual_covariance.contiguous()
        )

        self.dominant_residual = (
            result.dominant_residual.contiguous()
        )

        self.residual_diagonal = (
            result.residual_diagonal.contiguous()
        )

        self.reconstructed_residual = (
            result.reconstructed_residual.contiguous()
        )

        # 중요:
        # 기존 EigenRCRDFL의 covariance 입력 자체에
        # 완성된 Sigma_eff^(K)를 전달한다.
        self.covariances = (
            result.effective_covariance.contiguous()
        )

        # 기존 trainer/model interface를 유지하기 위한 zero tensor.
        # model은 eta=0으로 사용하므로 실제 계산에는 관여하지 않는다.
        self.eigen_risk = torch.zeros_like(
            self.covariances
        )

        self.capm_alpha = (
            capm.alpha.contiguous()
        )

        self.capm_beta = (
            capm.beta.contiguous()
        )

        self.eigenvalues = (
            result.eigenvalues.contiguous()
        )

        self.explained_ratio = (
            result.explained_ratio.contiguous()
        )

        self.cumulative_explained_ratio = (
            result
            .cumulative_explained_ratio
            .contiguous()
        )

        self.retained_ratio = (
            result.retained_ratio.contiguous()
        )

        self.decomposition_error_fro = (
            result
            .decomposition_error_fro
            .contiguous()
        )

    def __getitem__(
        self,
        index: int,
    ):
        item = super().__getitem__(
            index
        )

        # Oracle cache가 shuffle과 무관하게 정확한 sample을 찾도록 보장.
        item["sample_index"] = index

        item["base_covariance"] = (
            self.base_covariances[index]
        )

        item["market_covariance"] = (
            self.market_covariances[index]
        )

        item["dominant_residual"] = (
            self.dominant_residual[index]
        )

        item["residual_diagonal"] = (
            self.residual_diagonal[index]
        )

        item["reconstructed_residual"] = (
            self.reconstructed_residual[index]
        )

        item["decomposition_error_fro"] = (
            self.decomposition_error_fro[index]
        )

        return item