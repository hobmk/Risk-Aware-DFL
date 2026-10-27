import torch
from torch.utils.data import DataLoader

from implementations.collective_rcr_cvar_dfl.src.core import (
    CollectiveCVaRDataset,
    build_collective_cvar_layer,
    collective_cvar_cost,
)
from implementations.dfl_mvo_lee2025.src.model import ReturnMLP
from implementations.dfl_mvo_lee2025.src.optimization import covariance_to_risk_factor


dataset = CollectiveCVaRDataset(
    price_csv="data/raw/dow30_adjusted_close.csv",
    lookback=60,
    covariance_jitter=1e-6,
    dtype=torch.float32,
    tail_q=0.35,
)

loader = DataLoader(dataset, batch_size=16, shuffle=False)
batch = next(iter(loader))

model = ReturnMLP(
    n_assets=dataset.n_assets,
    lookback=60,
    hidden_dim=64,
    dropout=0.0,
)

layer = build_collective_cvar_layer(
    n_assets=dataset.n_assets,
    n_scenarios=60,
    gamma=0.1,
    cvar_alpha=0.9,
    max_weight=0.2,
)

features = batch["features"]
targets = batch["target"]

mu = model(features).to(torch.float64)
cov = batch["covariance"].to(torch.float64)
scenarios = batch["residual_scenarios"].to(torch.float64)
probabilities = batch["scenario_probabilities"].to(torch.float64)

risk_factor = covariance_to_risk_factor(
    cov,
    risk_aversion=1.0,
)

pred_w, pred_tau, pred_u = layer(
    mu,
    risk_factor,
    scenarios,
    probabilities,
)

oracle_w, oracle_tau, oracle_u = layer(
    targets.to(torch.float64),
    risk_factor,
    scenarios,
    probabilities,
)

pred_cost = collective_cvar_cost(
    pred_w,
    pred_tau,
    pred_u,
    targets.to(torch.float64),
    risk_factor,
    probabilities,
    gamma=0.1,
    cvar_alpha=0.9,
)

oracle_cost = collective_cvar_cost(
    oracle_w.detach(),
    oracle_tau.detach(),
    oracle_u.detach(),
    targets.to(torch.float64),
    risk_factor,
    probabilities,
    gamma=0.1,
    cvar_alpha=0.9,
)

regret = (pred_cost - oracle_cost).mean()
mse = torch.nn.functional.mse_loss(
    mu.float(),
    targets,
)

loss = 0.5 * regret + 0.5 * 30.0 * mse
loss.backward()

grad_sq = 0.0

for p in model.parameters():
    if p.grad is not None:
        grad_sq += p.grad.norm().item() ** 2

print(f"Loss    : {loss.item():.10f}")
print(f"Regret  : {regret.item():.10f}")
print(f"MSE     : {mse.item():.10f}")
print(f"GradNorm: {grad_sq ** 0.5:.10f}")
print(f"W sum   : {pred_w.sum(dim=1).mean().item():.10f}")
print(f"W min   : {pred_w.min().item():.10f}")
print(f"W max   : {pred_w.max().item():.10f}")