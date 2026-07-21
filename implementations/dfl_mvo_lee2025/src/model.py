from __future__ import annotations

import torch
from torch import nn


class ReturnMLP(nn.Module):
    """
    입력:
        [batch_size, lookback, n_assets]

    출력:
        [batch_size, n_assets]

    과거 수익률을 입력받아 다음 날 자산별 수익률을 예측한다.
    Linear layer 수는 기존과 동일하게 4개로 유지한다.
    """

    def __init__(
        self,
        n_assets: int,
        lookback: int = 60,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_assets <= 0:
            raise ValueError(
                f"n_assets는 0보다 커야 합니다: {n_assets}"
            )

        if lookback <= 0:
            raise ValueError(
                f"lookback은 0보다 커야 합니다: {lookback}"
            )

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim은 0보다 커야 합니다: {hidden_dim}"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout은 0 이상 1 미만이어야 합니다: {dropout}"
            )

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
        x = x.flatten(start_dim=1)
        return self.network(x)
