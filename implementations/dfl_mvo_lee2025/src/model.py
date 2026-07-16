from __future__ import annotations

import torch
from torch import nn


class ReturnMLP(nn.Module):
    """
    입력:
        [batch_size, lookback, n_assets]

    출력:
        [batch_size, n_assets]
        다음 날 자산별 예측 수익률
    """

    def __init__(
        self,
        n_assets: int,
        lookback: int = 60,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        input_dim = lookback * n_assets

        # Linear layer 4개
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, n_assets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 60, N] -> [B, 60*N]
        x = x.flatten(start_dim=1)

        # [B, 60*N] -> [B, N]
        return self.network(x)