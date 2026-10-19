from __future__ import annotations

import numpy as np
import torch

from implementations.eigen_rcr_dfl.src.capm import fit_capm
from implementations.eigen_rcr_dfl.src.dataset import EigenRCRDataset
from implementations.eigen_rcr_dfl.src.eigen_risk import covariance_matrix

from .rank_tail_risk import build_rank_tail_residual_risk

from .risk_cache import (
    CACHED_TENSOR_FIELDS,
    build_risk_cache_config,
    load_risk_cache,
    risk_cache_key,
    risk_cache_path,
    save_risk_cache,
)

class RankTailRCRDataset(EigenRCRDataset):
    """
    기존 EigenRCRDataset의 가격/수익률 로딩 로직은 재사용하고,
    _precompute만 Rank-Tail risk definition으로 교체한다.

    최종:
        Sigma_eff
        = Sigma_market
        + A_RT
        + covariance_jitter * I

    A_RT:
        residual tail rank dependence를 residual variance scale로
        복원한 PSD risk matrix.

    기존 EigenRCRDFL 모델을 그대로 사용하기 위해
        covariance = Sigma_eff
        eigen_risk = 0
    으로 제공한다.
    """

    def __init__(
        self,
        *args,
        tail_q: float = 0.20,
        tail_side: str = "down",
        risk_cache_dir: str | None = (
            "implementations/"
            "rank_tail_rcr_dfl/"
            "cache/risk_matrix"
        ),
        force_rebuild_risk_cache: bool = False,
        **kwargs,
    ) -> None:

        if not 0.0 < tail_q < 1.0:
            raise ValueError("tail_q는 0과 1 사이여야 합니다.")

        if tail_side not in {"down", "up"}:
            raise ValueError("tail_side는 'down' 또는 'up'이어야 합니다.")

        self.tail_q = float(tail_q)
        self.tail_side = tail_side

        self.risk_cache_dir = risk_cache_dir
        self.force_rebuild_risk_cache = bool(
            force_rebuild_risk_cache
        )

        self.risk_cache_hit = False
        self.risk_cache_key = None
        self.risk_cache_path = None

        # 부모 생성자는 eigen_k를 요구하지만,
        # subclass _precompute가 호출되므로 eigenmode risk는 계산되지 않는다.
        kwargs["eigen_k"] = 1
        kwargs["scale_mode"] = "none"

        super().__init__(
            *args,
            **kwargs,
        )

    def _precompute(self) -> None:
        cache_config = build_risk_cache_config(
            self
        )

        cache_key = risk_cache_key(
            cache_config
        )

        self.risk_cache_key = cache_key

        if self.risk_cache_dir is not None:
            cache_path = risk_cache_path(
                self.risk_cache_dir,
                cache_key,
            )

            self.risk_cache_path = str(
                cache_path
            )

            if (
                cache_path.exists()
                and not self.force_rebuild_risk_cache
            ):
                tensors = load_risk_cache(
                    cache_path
                )

                for name in CACHED_TENSOR_FIELDS:
                    setattr(
                        self,
                        name,
                        tensors[name],
                    )

                self.eigen_risk = torch.zeros_like(
                    self.covariances
                )

                self.risk_cache_hit = True

                print(
                    f"[RiskCache] HIT: "
                    f"{cache_path}"
                )

                return

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

        base_raw_covariance = covariance_matrix(
            asset_windows
        )

        residual_covariance = covariance_matrix(
            capm.residuals
        )

        market_variance = market_windows.var(
            dim=1,
            unbiased=True,
        )

        market_covariance = (
            capm.beta.unsqueeze(-1)
            * capm.beta.unsqueeze(-2)
            * market_variance[:, None, None]
        )

        market_covariance = 0.5 * (
            market_covariance
            + market_covariance.transpose(-1, -2)
        )

        rank_tail = build_rank_tail_residual_risk(
            residuals=capm.residuals,
            tail_q=self.tail_q,
            tail_side=self.tail_side,
        )

        eye = torch.eye(
            self.n_assets,
            dtype=self.dtype,
        ).unsqueeze(0)

        effective_covariance = (
            market_covariance
            + rank_tail.risk_matrix
            + self.covariance_jitter * eye
        )

        effective_covariance = 0.5 * (
            effective_covariance
            + effective_covariance.transpose(-1, -2)
        )

        base_covariance = (
            base_raw_covariance
            + self.covariance_jitter * eye
        )

        self.base_covariances = base_covariance.contiguous()
        self.covariances = effective_covariance.contiguous()

        self.capm_alpha = capm.alpha.contiguous()
        self.capm_beta = capm.beta.contiguous()

        self.market_covariances = market_covariance.contiguous()
        self.residual_covariances = residual_covariance.contiguous()
        self.rank_tail_dependence = (
            rank_tail.rank_tail_dependence.contiguous()
        )
        self.rank_tail_risk = rank_tail.risk_matrix.contiguous()

        # Existing model/trainer compatibility.
        self.eigen_risk = torch.zeros_like(
            self.covariances
        )

        self.tail_fraction = rank_tail.tail_fraction.contiguous()
        self.mean_offdiag_dependence = (
            rank_tail.mean_offdiag_dependence.contiguous()
        )
        self.mean_abs_offdiag_dependence = (
            rank_tail.mean_abs_offdiag_dependence.contiguous()
        )
        self.diagonal_preservation_error = (
            rank_tail.diagonal_preservation_error.contiguous()
        )

        # CAPM decomposition diagnostic:
        # Sigma ~= Sigma_market + Sigma_epsilon
        decomposition_error = torch.linalg.matrix_norm(
            base_raw_covariance
            - (
                market_covariance
                + residual_covariance
            ),
            ord="fro",
            dim=(-2, -1),
        )

        self.capm_decomposition_error = (
            decomposition_error.contiguous()
        )

        self.minimum_effective_eigenvalue = (
            torch.linalg.eigvalsh(
                self.covariances
            )
            .amin(dim=-1)
            .contiguous()
        )
        if self.risk_cache_dir is not None:
            cache_path = risk_cache_path(
                self.risk_cache_dir,
                cache_key,
            )

            tensors = {
                name: getattr(self, name)
                for name in CACHED_TENSOR_FIELDS
            }

            save_risk_cache(
                path=cache_path,
                config=cache_config,
                tensors=tensors,
            )

            self.risk_cache_path = str(
                cache_path
            )

            self.risk_cache_hit = False

            print(
                f"[RiskCache] BUILD: "
                f"{cache_path}"
            )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:
        target_position = int(
            self.target_positions[index]
        )
        start = (
            target_position
            - self.lookback
        )

        features = torch.tensor(
            self.returns[
                start:target_position
            ],
            dtype=self.dtype,
        )

        target = torch.tensor(
            self.returns[
                target_position
            ],
            dtype=self.dtype,
        )

        target_date = self.return_dates[
            target_position
        ]

        return {
            "sample_index": index,
            "features": features,
            "target": target,

            "covariance": self.covariances[index],
            "eigen_risk": self.eigen_risk[index],

            "base_covariance": self.base_covariances[index],
            "market_covariance": self.market_covariances[index],
            "residual_covariance": self.residual_covariances[index],
            "rank_tail_dependence": self.rank_tail_dependence[index],
            "rank_tail_risk": self.rank_tail_risk[index],
            "tail_fraction": self.tail_fraction[index],
            "mean_offdiag_dependence": self.mean_offdiag_dependence[index],
            "mean_abs_offdiag_dependence": self.mean_abs_offdiag_dependence[index],
            "diagonal_preservation_error": self.diagonal_preservation_error[index],
            "capm_decomposition_error": self.capm_decomposition_error[index],
            "minimum_effective_eigenvalue": self.minimum_effective_eigenvalue[index],

            "target_date": target_date.strftime("%Y-%m-%d"),
        }