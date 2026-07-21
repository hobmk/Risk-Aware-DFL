from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from implementations.dfl_mvo_lee2025.src.dataset import (
    RollingMVODataset,
    chronological_split,
)
from implementations.dfl_mvo_lee2025.src.losses import mvo_regret
from implementations.dfl_mvo_lee2025.src.model import ReturnMLP
from implementations.dfl_mvo_lee2025.src.optimization import (
    build_markowitz_layer,
    covariance_to_risk_factor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # 역전파 시에만 regret loss에 곱하는 값이다.
    # 출력되는 Train/Validation/Test Regret 값은 원래 단위로 유지된다.
    parser.add_argument(
        "--backward-scale",
        type=float,
        default=1.0,
    )

    # 포트폴리오 진단용 threshold
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--cap-tolerance",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--train-end",
        default="2021-12-31",
    )
    parser.add_argument(
        "--validation-end",
        default="2022-12-31",
    )

    return parser.parse_args()


class MLPWithMarkowitz(nn.Module):
    def __init__(
        self,
        n_assets: int,
        lookback: int,
        hidden_dim: int,
        risk_aversion: float,
        max_weight: float,
    ) -> None:
        super().__init__()

        self.return_model = ReturnMLP(
            n_assets=n_assets,
            lookback=lookback,
            hidden_dim=hidden_dim,
        )

        self.markowitz_layer = build_markowitz_layer(
            n_assets=n_assets,
            max_weight=max_weight,
        )

        # 진단 코드와 checkpoint 저장을 위해 모델 속성으로 보관한다.
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight

    def forward(
        self,
        features: torch.Tensor,
        covariance: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns
        -------
        predicted_returns:
            MLP가 예측한 다음 날 종목별 수익률 [B, N]

        predicted_weights:
            예측 수익률로 구성한 최적 포트폴리오 비중 [B, N]

        risk_factor:
            L.T @ L = lambda * Sigma를 만족하는 위험 factor [B, N, N]
        """
        predicted_returns = self.return_model(features)

        # cvxpylayers 계산은 CPU float64에서 수행한다.
        # device 이동 연산은 autograd graph를 유지한다.
        predicted_returns_solver = predicted_returns.to(
            device="cpu",
            dtype=torch.float64,
        )

        covariance_solver = covariance.to(
            device="cpu",
            dtype=torch.float64,
        )

        risk_factor = covariance_to_risk_factor(
            covariance=covariance_solver,
            risk_aversion=self.risk_aversion,
            # dataset.py에서 이미 covariance_jitter=1e-6을 적용한다.
            jitter=0.0,
        )

        predicted_weights, = self.markowitz_layer(
            predicted_returns_solver,
            risk_factor,
        )

        return (
            predicted_returns,
            predicted_weights,
            risk_factor,
        )


def compute_regret_loss(
    model: MLPWithMarkowitz,
    features: torch.Tensor,
    targets: torch.Tensor,
    covariance: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    예측 포트폴리오와 Oracle 포트폴리오 사이의 MVO Regret을 계산한다.
    """
    (
        predicted_returns,
        predicted_weights,
        risk_factor,
    ) = model(
        features=features,
        covariance=covariance,
    )

    true_returns_solver = targets.to(
        device="cpu",
        dtype=torch.float64,
    )

    # 실제 다음 날 수익률을 사전에 알고 있었다고 가정한
    # Oracle 포트폴리오를 계산한다.
    oracle_weights, = model.markowitz_layer(
        true_returns_solver,
        risk_factor,
    )

    # Oracle 포트폴리오는 모델 학습 대상이 아니므로 graph에서 분리한다.
    oracle_weights = oracle_weights.detach()

    regret_loss = mvo_regret(
        predicted_weights=predicted_weights,
        oracle_weights=oracle_weights,
        true_returns=true_returns_solver,
        risk_factor=risk_factor,
        reduction="mean",
    )

    return (
        regret_loss,
        predicted_returns,
        predicted_weights,
        risk_factor,
    )


def calculate_gradient_norm(
    module: nn.Module,
) -> tuple[float, bool]:
    """
    주어진 모듈의 전체 L2 gradient norm을 계산한다.
    """
    grad_norm_squared = 0.0
    has_gradient = False

    for parameter in module.parameters():
        if parameter.grad is None:
            continue

        has_gradient = True

        parameter_grad_norm = (
            parameter.grad.detach().norm(2).item()
        )

        grad_norm_squared += parameter_grad_norm**2

    grad_norm = grad_norm_squared**0.5

    return grad_norm, has_gradient


def print_portfolio_diagnostics(
    model: MLPWithMarkowitz,
    targets: torch.Tensor,
    predicted_weights: torch.Tensor,
    risk_factor: torch.Tensor,
    regret_loss: torch.Tensor,
    grad_norm: float,
    has_gradient: bool,
    active_threshold: float,
    cap_tolerance: float,
) -> None:
    """
    첫 번째 학습 batch에서 포트폴리오 경계해 여부와
    MVO 목적함수 scale을 출력한다.
    """
    with torch.no_grad():
        weights_debug = predicted_weights.detach()

        targets_solver = targets.detach().to(
            device="cpu",
            dtype=torch.float64,
        )

        # 종목별 weight가 threshold보다 큰 경우 active로 판정한다.
        active_assets = (
            weights_debug > active_threshold
        ).sum(dim=-1)

        # 최대 비중 제약에 거의 도달한 종목 수를 계산한다.
        capped_assets = (
            weights_debug
            >= model.max_weight - cap_tolerance
        ).sum(dim=-1)

        # 실제 수익률 기준 포트폴리오 수익률 항
        realized_return = torch.sum(
            targets_solver * weights_debug,
            dim=-1,
        )

        # lambda * w.T Sigma w
        transformed_weights = torch.matmul(
            risk_factor.detach(),
            weights_debug.unsqueeze(-1),
        ).squeeze(-1)

        risk_penalty = torch.sum(
            transformed_weights.square(),
            dim=-1,
        )

        weight_sum = weights_debug.sum(dim=-1)

        print("-" * 80)
        print("첫 번째 학습 Batch 진단")
        print("-" * 80)

        print(
            f"Batch Regret: {regret_loss.item():.8f}"
        )
        print(
            f"Gradient Norm: {grad_norm:.12f}"
        )
        print(
            f"Has Gradient: {has_gradient}"
        )

        print(
            f"Average Active Assets: "
            f"{active_assets.float().mean().item():.2f}"
        )
        print(
            f"Average Capped Assets: "
            f"{capped_assets.float().mean().item():.2f}"
        )

        print(
            f"Average Weight Sum: "
            f"{weight_sum.mean().item():.10f}"
        )
        print(
            f"Minimum Weight: "
            f"{weights_debug.min().item():.10f}"
        )
        print(
            f"Maximum Weight: "
            f"{weights_debug.max().item():.10f}"
        )

        print(
            f"Mean Absolute Return Term: "
            f"{realized_return.abs().mean().item():.10f}"
        )
        print(
            f"Mean Risk Term: "
            f"{risk_penalty.mean().item():.10f}"
        )

        print("-" * 80)


def run_epoch(
    model: MLPWithMarkowitz,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    backward_scale: float = 1.0,
    active_threshold: float = 1e-5,
    cap_tolerance: float = 1e-4,
    show_diagnostics: bool = False,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_regret = 0.0
    total_mse = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(loader):
        features = batch["features"].to(device)
        targets = batch["target"].to(device)

        # covariance는 cvxpylayers에서 CPU float64로 사용할 것이므로
        # 여기서는 GPU로 이동시키지 않는다.
        covariance = batch["covariance"]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            (
                regret_loss,
                predicted_returns,
                predicted_weights,
                risk_factor,
            ) = compute_regret_loss(
                model=model,
                features=features,
                targets=targets,
                covariance=covariance,
            )

            # MSE는 현재 순수 Regret 학습에서는 성능 확인용 지표이다.
            mse = nn.functional.mse_loss(
                predicted_returns,
                targets,
            )

            if is_training:
                # 실제 로그에는 원래 regret를 사용하고,
                # backward에서만 선택적으로 scale을 적용한다.
                backward_loss = (
                    backward_scale * regret_loss
                )

                backward_loss.backward()

                # 모든 batch를 출력하면 로그가 지나치게 길어지므로
                # 각 epoch의 첫 번째 batch에서만 진단한다.
                if show_diagnostics and batch_index == 0:
                    grad_norm, has_gradient = (
                        calculate_gradient_norm(
                            model.return_model
                        )
                    )

                    print_portfolio_diagnostics(
                        model=model,
                        targets=targets,
                        predicted_weights=predicted_weights,
                        risk_factor=risk_factor,
                        regret_loss=regret_loss,
                        grad_norm=grad_norm,
                        has_gradient=has_gradient,
                        active_threshold=active_threshold,
                        cap_tolerance=cap_tolerance,
                    )

                optimizer.step()

        batch_size = features.size(0)

        total_regret += (
            regret_loss.detach().item() * batch_size
        )
        total_mse += (
            mse.detach().item() * batch_size
        )
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "DataLoader에 학습 또는 평가 샘플이 없습니다."
        )

    return (
        total_regret / total_samples,
        total_mse / total_samples,
    )


def main() -> None:
    args = parse_args()

    if args.backward_scale <= 0:
        raise ValueError(
            "--backward-scale은 0보다 커야 합니다."
        )

    if args.active_threshold < 0:
        raise ValueError(
            "--active-threshold는 0 이상이어야 합니다."
        )

    if args.cap_tolerance < 0:
        raise ValueError(
            "--cap-tolerance는 0 이상이어야 합니다."
        )

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    price_path = Path(args.price_csv)

    if not price_path.is_absolute():
        price_path = PROJECT_ROOT / price_path

    dataset = RollingMVODataset(
        price_csv=price_path,
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float32,
    )

    train_set, validation_set, test_set = chronological_split(
        dataset=dataset,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 첫 backward 시 CUDA context 경고가 발생하는 것을 방지한다.
    if device.type == "cuda":
        torch.cuda.init()

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = MLPWithMarkowitz(
        n_assets=dataset.n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
    ).float().to(device)

    optimizer = torch.optim.AdamW(
        model.return_model.parameters(),
        lr=args.learning_rate,
    )

    output_dir = (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "mlp_markowitz_regret"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        output_dir / "best_model.pt"
    )

    best_validation_regret = float("inf")
    patience_count = 0

    print("=" * 80)
    print("DFL-MVO Pure Regret Training")
    print("=" * 80)
    print("Device:", device)
    print("Loss: MVO Regret")
    print("Risk aversion:", args.risk_aversion)
    print("Maximum weight:", args.max_weight)
    print("Seed:", args.seed)
    print("Backward scale:", args.backward_scale)
    print("Train samples:", len(train_set))
    print("Validation samples:", len(validation_set))
    print("Test samples:", len(test_set))
    print("=" * 80)

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_regret, train_mse = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            backward_scale=args.backward_scale,
            active_threshold=args.active_threshold,
            cap_tolerance=args.cap_tolerance,
            show_diagnostics=True,
        )

        validation_regret, validation_mse = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            optimizer=None,
            active_threshold=args.active_threshold,
            cap_tolerance=args.cap_tolerance,
            show_diagnostics=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Regret: {train_regret:.8f} | "
            f"Train MSE: {train_mse:.8f} | "
            f"Validation Regret: "
            f"{validation_regret:.8f} | "
            f"Validation MSE: "
            f"{validation_mse:.8f}"
        )

        if validation_regret < best_validation_regret:
            best_validation_regret = validation_regret
            patience_count = 0

            torch.save(
                {
                    "model_state_dict":
                        model.return_model.state_dict(),
                    "validation_regret":
                        validation_regret,
                    "validation_mse":
                        validation_mse,
                    "risk_aversion":
                        args.risk_aversion,
                    "max_weight":
                        args.max_weight,
                    "seed":
                        args.seed,
                    "backward_scale":
                        args.backward_scale,
                    "n_assets":
                        dataset.n_assets,
                    "lookback":
                        args.lookback,
                    "hidden_dim":
                        args.hidden_dim,
                    "tickers":
                        dataset.tickers,
                },
                best_model_path,
            )
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.return_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_regret, test_mse = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
        active_threshold=args.active_threshold,
        cap_tolerance=args.cap_tolerance,
        show_diagnostics=False,
    )

    print("=" * 80)
    print(
        f"Best Validation Regret: "
        f"{best_validation_regret:.8f}"
    )
    print(
        f"Test Regret: {test_regret:.8f}"
    )
    print(
        f"Test MSE: {test_mse:.8f}"
    )
    print(
        f"Saved: {best_model_path}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()