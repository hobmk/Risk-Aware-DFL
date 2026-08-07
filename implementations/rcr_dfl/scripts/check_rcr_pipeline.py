from __future__ import annotations

import argparse

import torch

from implementations.rcr_dfl.src.decision_model import RCRMLPWithMarkowitz
from implementations.rcr_dfl.src.losses import compute_rcr_losses
from implementations.rcr_dfl.src.residual_risk import (
    correlation_matrix,
    matrix_diagnostics,
    scale_correlation_to_covariance,
    shrink_correlation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual-correlation RCR-DFL synthetic smoke test")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--risk-aversion", type=float, default=100.0)
    parser.add_argument("--max-weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mse-scale", type=float, default=15.0)
    parser.add_argument("--residual-correlation-shrinkage", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_spd_matrix(batch_size: int, n_assets: int, scale: float) -> torch.Tensor:
    matrix = torch.randn(batch_size, n_assets, n_assets, dtype=torch.float64)
    covariance = matrix @ matrix.transpose(-1, -2)
    covariance = covariance / n_assets * scale
    return covariance + 1e-5 * torch.eye(n_assets, dtype=torch.float64)


def gradient_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().norm(2).item() ** 2
    return total**0.5


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    batch_size = 3
    lookback = 10
    n_assets = 5
    features = torch.randn(batch_size, lookback, n_assets, dtype=torch.float32) * 0.01
    targets = torch.randn(batch_size, n_assets, dtype=torch.float32) * 0.01
    covariance = make_spd_matrix(batch_size, n_assets, scale=2e-4)

    synthetic_residuals = torch.randn(batch_size, 30, n_assets, dtype=torch.float64)
    synthetic_residuals[..., 1:] += 0.35 * synthetic_residuals[..., :1]
    residual_correlation = shrink_correlation(
        correlation_matrix(synthetic_residuals),
        shrinkage=args.residual_correlation_shrinkage,
    )
    a_res = scale_correlation_to_covariance(
        residual_correlation,
        reference_covariance=covariance,
        scaling="trace",
    )

    model = RCRMLPWithMarkowitz(
        n_assets=n_assets,
        lookback=lookback,
        hidden_dim=32,
        dropout=0.0,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        eta=args.eta,
    ).float()
    output = model(features=features, covariance=covariance, a_res=a_res)
    oracle_weights = model.solve_oracle(true_returns=targets, risk_factor=output.risk_factor)
    losses = compute_rcr_losses(
        predicted_returns=output.predicted_returns,
        predicted_weights=output.predicted_weights,
        oracle_weights=oracle_weights,
        true_returns=targets,
        risk_factor=output.risk_factor,
        alpha=args.alpha,
        mse_scale=args.mse_scale,
    )
    losses.total.backward()

    diagnostics = matrix_diagnostics(output.effective_covariance)
    weight_sum_error = (output.predicted_weights.sum(dim=-1) - 1.0).abs().max().item()
    print(f"eta: {args.eta}")
    print(f"risk_aversion: {args.risk_aversion}")
    print(f"max_weight: {args.max_weight}")
    print(f"predicted_returns: {tuple(output.predicted_returns.shape)}")
    print(f"predicted_weights: {tuple(output.predicted_weights.shape)}")
    print(f"effective_covariance: {tuple(output.effective_covariance.shape)}")
    print(f"combined loss: {losses.total.item():.8f}")
    print(f"RCR regret: {losses.regret.item():.8f}")
    print(f"MSE: {losses.mse.item():.8f}")
    print(f"gradient norm: {gradient_norm(model.return_model):.3e}")
    print(f"weight sum error: {weight_sum_error:.3e}")
    print(f"minimum weight: {output.predicted_weights.min().item():.3e}")
    print(f"maximum weight: {output.predicted_weights.max().item():.3e}")
    print(f"symmetry error Sigma_eff: {diagnostics.symmetry_error.max().item():.3e}")
    print(f"minimum eigenvalue Sigma_eff: {diagnostics.minimum_eigenvalue.min().item():.3e}")
    print(f"maximum condition number Sigma_eff: {diagnostics.condition_number.max().item():.3e}")


if __name__ == "__main__":
    main()
