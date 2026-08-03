from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .effective_covariance import Normalization, build_effective_covariance
from .model import ReturnMLP
from .optimization import build_markowitz_layer, covariance_to_risk_factor


@dataclass(frozen=True)
class RCRForwardOutput:
    predicted_returns: torch.Tensor
    predicted_weights: torch.Tensor
    effective_covariance: torch.Tensor
    risk_factor: torch.Tensor


class RCRMLPWithMarkowitz(nn.Module):
    """
    4-layer MLP와 Residual Collective Risk-aware MVO layer를 연결한다.

    MLP는 일반 학습 device에서 실행하고, CVXPYLayer 입력은
    기존 벤치마크와 동일하게 CPU float64로 변환한다.
    """

    def __init__(
        self,
        n_assets: int,
        lookback: int = 60,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        risk_aversion: float = 1.0,
        max_weight: float = 1.0,
        eta: float = 0.5,
        normalization: Normalization = "trace",
        effective_jitter: float = 0.0,
        project_psd: bool = False,
        minimum_eigenvalue: float = 0.0,
    ) -> None:
        super().__init__()
        if risk_aversion <= 0:
            raise ValueError(
                "risk_aversion은 0보다 커야 합니다: "
                f"{risk_aversion}"
            )
        if eta < 0:
            raise ValueError(f"eta는 0 이상이어야 합니다: {eta}")

        self.return_model = ReturnMLP(
            n_assets=n_assets,
            lookback=lookback,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.markowitz_layer = build_markowitz_layer(
            n_assets=n_assets,
            max_weight=max_weight,
        )

        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.eta = eta
        self.normalization = normalization
        self.effective_jitter = effective_jitter
        self.project_psd = project_psd
        self.minimum_eigenvalue = minimum_eigenvalue

    def forward(
        self,
        features: torch.Tensor,
        covariance: torch.Tensor,
        residual_covariance: torch.Tensor,
    ) -> RCRForwardOutput:
        predicted_returns = self.return_model(features)

        covariance_solver = covariance.to(
            device="cpu",
            dtype=torch.float64,
        )
        residual_covariance_solver = residual_covariance.to(
            device="cpu",
            dtype=torch.float64,
        )
        effective_covariance = build_effective_covariance(
            covariance=covariance_solver,
            residual_covariance=residual_covariance_solver,
            eta=self.eta,
            normalization=self.normalization,
            jitter=self.effective_jitter,
            project_psd=self.project_psd,
            minimum_eigenvalue=self.minimum_eigenvalue,
        )
        risk_factor = covariance_to_risk_factor(
            covariance=effective_covariance,
            risk_aversion=self.risk_aversion,
            jitter=0.0,
        )

        predicted_returns_solver = predicted_returns.to(
            device="cpu",
            dtype=torch.float64,
        )
        predicted_weights, = self.markowitz_layer(
            predicted_returns_solver,
            risk_factor,
        )

        return RCRForwardOutput(
            predicted_returns=predicted_returns,
            predicted_weights=predicted_weights,
            effective_covariance=effective_covariance,
            risk_factor=risk_factor,
        )

    def solve_oracle(
        self,
        true_returns: torch.Tensor,
        risk_factor: torch.Tensor,
    ) -> torch.Tensor:
        true_returns_solver = true_returns.to(
            device="cpu",
            dtype=torch.float64,
        )
        oracle_weights, = self.markowitz_layer(
            true_returns_solver,
            risk_factor,
        )
        return oracle_weights.detach()
