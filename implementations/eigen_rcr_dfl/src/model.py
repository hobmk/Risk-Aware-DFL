from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .optimization import (
    build_markowitz_layer,
    covariance_to_risk_factor,
)


class ReturnMLP(nn.Module):
    def __init__(
        self,
        n_assets: int,
        lookback: int = 60,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_assets <= 0:
            raise ValueError("n_assets는 1 이상이어야 합니다.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim은 1 이상이어야 합니다.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout은 0 이상 1 미만이어야 합니다.")

        input_dim = lookback * n_assets

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, n_assets),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "입력은 [B, lookback, N]이어야 합니다."
            )

        return self.network(
            x.flatten(start_dim=1)
        )


@dataclass
class EigenRCRForwardOutput:
    predicted_returns: torch.Tensor
    predicted_weights: torch.Tensor
    effective_covariance: torch.Tensor
    risk_factor: torch.Tensor


class EigenRCRDFL(nn.Module):
    def __init__(
        self,
        n_assets: int,
        lookback: int = 60,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        risk_aversion: float = 1.0,
        max_weight: float = 0.2,
        eta: float = 0.5,
        effective_jitter: float = 0.0,
    ) -> None:
        super().__init__()

        if eta < 0:
            raise ValueError("eta는 0 이상이어야 합니다.")

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
        self.effective_jitter = effective_jitter

    def forward(
        self,
        features: torch.Tensor,
        covariance: torch.Tensor,
        eigen_risk: torch.Tensor,
    ) -> EigenRCRForwardOutput:
        predicted_returns = self.return_model(
            features
        )

        covariance = covariance.to(
            device="cpu",
            dtype=torch.float64,
        )

        eigen_risk = eigen_risk.to(
            device="cpu",
            dtype=torch.float64,
        )

        # Sigma_eff = Sigma + eta * A_res^(K)
        effective_covariance = (
            covariance
            + self.eta * eigen_risk
        )

        effective_covariance = 0.5 * (
            effective_covariance
            + effective_covariance.transpose(-1, -2)
        )

        risk_factor = covariance_to_risk_factor(
            covariance=effective_covariance,
            risk_aversion=self.risk_aversion,
            jitter=self.effective_jitter,
        )

        predicted_returns_solver = predicted_returns.to(
            device="cpu",
            dtype=torch.float64,
        )

        predicted_weights, = self.markowitz_layer(
            predicted_returns_solver,
            risk_factor,
        )

        return EigenRCRForwardOutput(
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
        true_returns = true_returns.to(
            device="cpu",
            dtype=torch.float64,
        )

        oracle_weights, = self.markowitz_layer(
            true_returns,
            risk_factor,
        )

        return oracle_weights.detach()