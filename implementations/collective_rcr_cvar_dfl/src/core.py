from __future__ import annotations

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

from implementations.dfl_mvo_lee2025.src.dataset import RollingMVODataset


class CollectiveCVaRDataset(RollingMVODataset):
    def __init__(self, *args, tail_q=0.35, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_q = tail_q
        self._precompute_residual_scenarios()

    def _precompute_residual_scenarios(self):
        windows = np.lib.stride_tricks.sliding_window_view(
            self.returns, self.lookback, axis=0
        ).transpose(0, 2, 1)[:-1]

        market = windows.mean(axis=2)
        x = market - market.mean(axis=1, keepdims=True)
        y = windows - windows.mean(axis=1, keepdims=True)

        beta = np.sum(x[:, :, None] * y, axis=1) / np.maximum(
            np.sum(x**2, axis=1, keepdims=True), 1e-12
        )
        alpha = windows.mean(axis=1) - beta * market.mean(axis=1, keepdims=True)
        residuals = windows - alpha[:, None, :] - market[:, :, None] * beta[:, None, :]

        order = np.argsort(residuals, axis=1)
        ranks = np.empty_like(order, dtype=np.float64)
        ordinal = np.arange(1, self.lookback + 1, dtype=np.float64)[None, :, None]
        np.put_along_axis(ranks, order, np.broadcast_to(ordinal, residuals.shape), axis=1)

        u = ranks / float(self.lookback + 1)
        z = np.maximum((self.tail_q - u) / self.tail_q, 0.0)

        breadth = (z > 0).mean(axis=2)
        severity = z.mean(axis=2)
        score = breadth * severity

        # exact zero probability는 CVXPY backward의 비유일성을 만들 수 있어 작은 floor만 부여
        score = score + 1e-8
        probability = score / score.sum(axis=1, keepdims=True)

        self.residual_scenarios = residuals
        self.collective_probabilities = probability

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["residual_scenarios"] = torch.tensor(
            self.residual_scenarios[index], dtype=self.dtype
        )
        sample["scenario_probabilities"] = torch.tensor(
            self.collective_probabilities[index], dtype=self.dtype
        )
        return sample


def build_collective_cvar_layer(
    n_assets,
    n_scenarios,
    gamma=0.1,
    cvar_alpha=0.9,
    max_weight=0.2,
):
    w = cp.Variable(n_assets)
    tau = cp.Variable()
    u = cp.Variable(n_scenarios)

    mu = cp.Parameter(n_assets)
    risk_factor = cp.Parameter((n_assets, n_assets))
    scenarios = cp.Parameter((n_scenarios, n_assets))
    probabilities = cp.Parameter(n_scenarios, nonneg=True)

    losses = -scenarios @ w

    cvar = tau + probabilities @ u / (1.0 - cvar_alpha)

    objective = cp.Minimize(
        cp.sum_squares(risk_factor @ w)
        - mu @ w
        + gamma * cvar
        + 1e-10 * (cp.sum_squares(u) + cp.square(tau))
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
        u >= losses - tau,
        u >= 0,
    ]

    problem = cp.Problem(objective, constraints)

    if not problem.is_dpp():
        raise RuntimeError("Collective CVaR problem is not DPP.")

    return CvxpyLayer(
        problem,
        parameters=[mu, risk_factor, scenarios, probabilities],
        variables=[w, tau, u],
    )


def collective_cvar_cost(
    weights,
    tau,
    u,
    true_returns,
    risk_factor,
    probabilities,
    gamma,
    cvar_alpha,
):
    realized_return = torch.sum(true_returns * weights, dim=-1)

    transformed = torch.matmul(
        risk_factor, weights.unsqueeze(-1)
    ).squeeze(-1)

    variance_cost = transformed.square().sum(dim=-1)

    cvar = tau + torch.sum(
        probabilities * u, dim=-1
    ) / (1.0 - cvar_alpha)

    return variance_cost - realized_return + gamma * cvar