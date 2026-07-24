from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from implementations.dfl_mvo_lee2025.src.losses import mvo_regret
from implementations.dfl_mvo_lee2025.src.optimization import (
    build_markowitz_layer,
    covariance_to_risk_factor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=1e-4)
    return parser.parse_args()


def build_covariance(n_assets: int) -> torch.Tensor:
    variances = torch.linspace(4e-4, 9e-4, n_assets, dtype=torch.float64)
    covariance = torch.diag(variances)
    covariance += torch.full((n_assets, n_assets), 2e-5, dtype=torch.float64)
    covariance += 1e-8 * torch.eye(n_assets, dtype=torch.float64)
    return covariance.unsqueeze(0)


def evaluate_regret(
    predicted_returns: torch.Tensor,
    true_returns: torch.Tensor,
    risk_factor: torch.Tensor,
    markowitz_layer,
    oracle_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_weights, = markowitz_layer(predicted_returns, risk_factor)
    regret = mvo_regret(
        predicted_weights=predicted_weights,
        oracle_weights=oracle_weights,
        true_returns=true_returns,
        risk_factor=risk_factor,
        reduction="mean",
    )
    return regret, predicted_weights


def finite_difference_gradient(
    predicted_returns: torch.Tensor,
    true_returns: torch.Tensor,
    risk_factor: torch.Tensor,
    markowitz_layer,
    oracle_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    base = predicted_returns.detach().clone()
    gradient = torch.zeros_like(base)

    for asset_index in range(base.shape[-1]):
        plus = base.clone()
        minus = base.clone()
        plus[0, asset_index] += eps
        minus[0, asset_index] -= eps

        plus_regret, _ = evaluate_regret(
            plus, true_returns, risk_factor, markowitz_layer, oracle_weights
        )
        minus_regret, _ = evaluate_regret(
            minus, true_returns, risk_factor, markowitz_layer, oracle_weights
        )

        gradient[0, asset_index] = (
            plus_regret.detach() - minus_regret.detach()
        ) / (2.0 * eps)

    return gradient


def print_vector(name: str, tensor: torch.Tensor) -> None:
    values = ", ".join(
        f"{value:+.8e}" for value in tensor.detach().flatten().tolist()
    )
    print(f"{name}: [{values}]")


def run_case(
    name: str,
    predicted_values: list[float],
    true_values: list[float],
    risk_aversion: float,
    max_weight: float,
    eps: float,
) -> None:
    n_assets = len(predicted_values)
    covariance = build_covariance(n_assets)
    risk_factor = covariance_to_risk_factor(
        covariance=covariance,
        risk_aversion=risk_aversion,
        jitter=0.0,
    )
    markowitz_layer = build_markowitz_layer(
        n_assets=n_assets,
        max_weight=max_weight,
    )

    predicted_returns = torch.tensor(
        [predicted_values],
        dtype=torch.float64,
        requires_grad=True,
    )
    true_returns = torch.tensor([true_values], dtype=torch.float64)

    oracle_weights, = markowitz_layer(true_returns, risk_factor)
    oracle_weights = oracle_weights.detach()

    regret, predicted_weights = evaluate_regret(
        predicted_returns,
        true_returns,
        risk_factor,
        markowitz_layer,
        oracle_weights,
    )
    autograd_gradient, = torch.autograd.grad(regret, predicted_returns)

    finite_difference = finite_difference_gradient(
        predicted_returns,
        true_returns,
        risk_factor,
        markowitz_layer,
        oracle_weights,
        eps,
    )

    absolute_error = (autograd_gradient - finite_difference).abs()
    denominator = finite_difference.abs().clamp_min(1e-10)
    relative_error = absolute_error / denominator

    auto_norm = autograd_gradient.norm().item()
    fd_norm = finite_difference.norm().item()

    cosine_similarity = float("nan")
    if auto_norm > 0 and fd_norm > 0:
        cosine_similarity = torch.nn.functional.cosine_similarity(
            autograd_gradient.flatten(),
            finite_difference.flatten(),
            dim=0,
        ).item()

    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"Risk aversion: {risk_aversion}")
    print(f"Maximum weight: {max_weight}")
    print(f"Finite-difference epsilon: {eps:.1e}")
    print(f"Regret: {regret.item():.10f}")
    print_vector("Predicted weights", predicted_weights)
    print_vector("Oracle weights", oracle_weights)
    print_vector("Autograd gradient", autograd_gradient)
    print_vector("Finite-difference gradient", finite_difference)
    print_vector("Absolute error", absolute_error)
    print_vector("Relative error", relative_error)
    print(f"Autograd norm: {auto_norm:.12e}")
    print(f"Finite-difference norm: {fd_norm:.12e}")
    print(f"Maximum absolute error: {absolute_error.max().item():.12e}")
    print(f"Cosine similarity: {cosine_similarity:.8f}")

    active_assets = (predicted_weights.detach() > 1e-3).sum(dim=-1).item()
    capped_assets = (
        predicted_weights.detach() >= max_weight - 1e-4
    ).sum(dim=-1).item()

    print(f"Active assets: {active_assets}")
    print(f"Capped assets: {capped_assets}")

    if auto_norm < 1e-8 and fd_norm < 1e-8:
        print(
            "판정: 두 gradient가 모두 거의 0입니다. "
            "현재 active set 안에서는 포트폴리오가 locally constant입니다."
        )
    elif cosine_similarity >= 0.99 and absolute_error.max().item() <= 1e-3:
        print("판정: autograd와 finite difference가 대체로 일치합니다.")
    else:
        print(
            "판정: gradient 불일치 가능성이 있습니다. "
            "eps를 1e-3, 1e-4, 1e-5로 바꿔 재확인하세요."
        )


def main() -> None:
    args = parse_args()

    run_case(
        name="Case 1: Smooth reference",
        predicted_values=[0.00100, 0.00110, 0.00090, 0.00105, 0.00095, 0.00102],
        true_values=[0.00120, 0.00070, 0.00110, 0.00080, 0.00100, 0.00090],
        risk_aversion=100.0,
        max_weight=1.0,
        eps=args.eps,
    )

    run_case(
        name="Case 2: Paper-like k=1 boundary",
        predicted_values=[0.020, 0.012, 0.008, 0.004, 0.001, -0.002],
        true_values=[0.004, 0.018, 0.006, 0.002, -0.001, 0.003],
        risk_aversion=1.0,
        max_weight=1.0,
        eps=args.eps,
    )

    run_case(
        name="Case 3: Paper-like k=5 boundary",
        predicted_values=[0.020, 0.015, 0.011, 0.008, 0.004, -0.003],
        true_values=[0.004, 0.018, 0.006, 0.002, -0.001, 0.003],
        risk_aversion=1.0,
        max_weight=0.2,
        eps=args.eps,
    )


if __name__ == "__main__":
    main()
