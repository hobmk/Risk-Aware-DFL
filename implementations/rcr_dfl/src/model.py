from __future__ import annotations

import torch
from torch import nn


class ReturnMLP(nn.Module):
    """
    과거 수익률을 입력받아 다음 거래일 자산별 수익률을 예측한다.

    입력: [B, lookback, N]
    출력: [B, N]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "입력 shape은 [B, lookback, N]이어야 합니다. "
                f"현재 shape={tuple(x.shape)}"
            )
        return self.network(x.flatten(start_dim=1))
