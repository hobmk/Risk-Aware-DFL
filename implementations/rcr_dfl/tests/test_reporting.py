from __future__ import annotations

import pytest
import torch

from implementations.rcr_dfl.src.reporting import (
    calculate_portfolio_metrics,
)


def test_portfolio_metrics_final_wealth_and_drawdown() -> None:
    returns = torch.tensor(
        [0.10, -0.10],
        dtype=torch.float64,
    )

    metrics, wealth, drawdown = (
        calculate_portfolio_metrics(
            returns,
            periods_per_year=2,
        )
    )

    assert wealth.tolist() == pytest.approx(
        [1.10, 0.99]
    )
    assert drawdown.tolist() == pytest.approx(
        [0.0, 0.10]
    )
    assert metrics["final_wealth"] == pytest.approx(
        0.99
    )
    assert metrics["total_return"] == pytest.approx(
        -0.01
    )
    assert metrics[
        "maximum_drawdown"
    ] == pytest.approx(0.10)


def test_portfolio_metrics_constant_positive_returns() -> None:
    returns = torch.tensor(
        [0.01, 0.01, 0.01],
        dtype=torch.float64,
    )

    metrics, _, drawdown = calculate_portfolio_metrics(
        returns,
        periods_per_year=3,
    )

    assert metrics["final_wealth"] == pytest.approx(
        1.01**3
    )
    assert metrics[
        "maximum_drawdown"
    ] == pytest.approx(0.0)
    assert torch.allclose(
        drawdown,
        torch.zeros_like(drawdown),
    )
