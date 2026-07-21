from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
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
from implementations.dfl_mvo_lee2025.src.losses import (
    combined_loss,
    markowitz_cost,
    mvo_regret,
)
from implementations.dfl_mvo_lee2025.src.metrics import (
    calculate_portfolio_metrics,
)
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
        "--alpha",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--mse-scale",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
    "--dropout",
    type=float,
    default=0.1,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=1e-3,
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
        dropout: float,
        risk_aversion: float,
        max_weight: float,
    ) -> None:
        super().__init__()

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

    def forward(
        self,
        features: torch.Tensor,
        covariance: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        predicted_returns = self.return_model(features)

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


def compute_losses(
    model: MLPWithMarkowitz,
    features: torch.Tensor,
    targets: torch.Tensor,
    covariance: torch.Tensor,
    alpha: float,
    mse_scale: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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

    oracle_weights, = model.markowitz_layer(
        true_returns_solver,
        risk_factor,
    )
    oracle_weights = oracle_weights.detach()

    regret = mvo_regret(
        predicted_weights=predicted_weights,
        oracle_weights=oracle_weights,
        true_returns=true_returns_solver,
        risk_factor=risk_factor,
        reduction="mean",
    )

    mse = nn.functional.mse_loss(
        predicted_returns,
        targets,
    )

    total_loss = combined_loss(
        regret_loss=regret,
        mse_loss=mse,
        alpha=alpha,
        mse_scale=mse_scale,
    )

    return (
        total_loss,
        regret,
        mse,
        predicted_weights,
        risk_factor,
    )


def calculate_gradient_norm(
    module: nn.Module,
) -> tuple[float, bool]:
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

    return grad_norm_squared**0.5, has_gradient


def print_training_diagnostics(
    model: MLPWithMarkowitz,
    targets: torch.Tensor,
    total_loss: torch.Tensor,
    regret: torch.Tensor,
    mse: torch.Tensor,
    predicted_weights: torch.Tensor,
    risk_factor: torch.Tensor,
    alpha: float,
    mse_scale: float,
    grad_norm: float,
    has_gradient: bool,
    active_threshold: float,
    cap_tolerance: float,
) -> None:
    with torch.no_grad():
        weights = predicted_weights.detach()

        targets_solver = targets.detach().to(
            device="cpu",
            dtype=torch.float64,
        )

        active_assets = (
            weights > active_threshold
        ).sum(dim=-1)

        capped_assets = (
            weights
            >= model.max_weight - cap_tolerance
        ).sum(dim=-1)

        realized_return = torch.sum(
            targets_solver * weights,
            dim=-1,
        )

        transformed_weights = torch.matmul(
            risk_factor.detach(),
            weights.unsqueeze(-1),
        ).squeeze(-1)

        risk_penalty = torch.sum(
            transformed_weights.square(),
            dim=-1,
        )

        mvo_contribution = (
            alpha * regret.detach()
        ).item()

        mse_contribution = (
            (1.0 - alpha)
            * mse_scale
            * mse.detach()
        ).item()

        print("-" * 80)
        print("첫 번째 학습 Batch 진단")
        print("-" * 80)
        print(
            f"Combined Loss: {total_loss.item():.8f}"
        )
        print(
            f"Raw Regret: {regret.item():.8f}"
        )
        print(
            f"Raw MSE: {mse.item():.8f}"
        )
        print(
            f"Weighted MVO Contribution: "
            f"{mvo_contribution:.8f}"
        )
        print(
            f"Weighted MSE Contribution: "
            f"{mse_contribution:.8f}"
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
            f"Minimum Weight: "
            f"{weights.min().item():.10f}"
        )
        print(
            f"Maximum Weight: "
            f"{weights.max().item():.10f}"
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
    alpha: float,
    mse_scale: float,
    optimizer: torch.optim.Optimizer | None = None,
    active_threshold: float = 1e-3,
    cap_tolerance: float = 1e-4,
    show_diagnostics: bool = False,
) -> tuple[
    float,
    float,
    float,
]:
    is_training = optimizer is not None
    model.train(is_training)

    total_combined = 0.0
    total_regret = 0.0
    total_mse = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(loader):
        features = batch["features"].to(device)
        targets = batch["target"].to(device)
        covariance = batch["covariance"]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            (
                total_loss,
                regret,
                mse,
                predicted_weights,
                risk_factor,
            ) = compute_losses(
                model=model,
                features=features,
                targets=targets,
                covariance=covariance,
                alpha=alpha,
                mse_scale=mse_scale,
            )

            if is_training:
                total_loss.backward()

                if show_diagnostics and batch_index == 0:
                    grad_norm, has_gradient = (
                        calculate_gradient_norm(
                            model.return_model
                        )
                    )

                    print_training_diagnostics(
                        model=model,
                        targets=targets,
                        total_loss=total_loss,
                        regret=regret,
                        mse=mse,
                        predicted_weights=predicted_weights,
                        risk_factor=risk_factor,
                        alpha=alpha,
                        mse_scale=mse_scale,
                        grad_norm=grad_norm,
                        has_gradient=has_gradient,
                        active_threshold=active_threshold,
                        cap_tolerance=cap_tolerance,
                    )

                optimizer.step()

        batch_size = features.size(0)

        total_combined += (
            total_loss.detach().item() * batch_size
        )
        total_regret += (
            regret.detach().item() * batch_size
        )
        total_mse += (
            mse.detach().item() * batch_size
        )
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "DataLoader에 샘플이 없습니다."
        )

    return (
        total_combined / total_samples,
        total_regret / total_samples,
        total_mse / total_samples,
    )


def evaluate_and_save_test_results(
    model: MLPWithMarkowitz,
    loader: DataLoader,
    device: torch.device,
    tickers: list[str],
    output_dir: Path,
    dataset_name: str,
    alpha: float,
    mse_scale: float,
    risk_aversion: float,
    max_weight: float,
    seed: int,
    active_threshold: float,
    checkpoint: dict,
) -> dict[str, float | int | str]:
    model.eval()

    asset_rows: list[dict] = []
    daily_rows: list[dict] = []
    portfolio_returns: list[float] = []

    total_combined = 0.0
    total_regret = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["target"].to(device)
            covariance = batch["covariance"].to(
                device="cpu",
                dtype=torch.float64,
            )
            dates = list(batch["target_date"])

            (
                predicted_returns,
                predicted_weights,
                risk_factor,
            ) = model(
                features=features,
                covariance=covariance,
            )

            predicted_returns_cpu = (
                predicted_returns.detach()
                .to(device="cpu", dtype=torch.float64)
            )
            true_returns_cpu = (
                targets.detach()
                .to(device="cpu", dtype=torch.float64)
            )
            predicted_weights = (
                predicted_weights.detach()
                .to(device="cpu", dtype=torch.float64)
            )

            oracle_weights, = model.markowitz_layer(
                true_returns_cpu,
                risk_factor,
            )
            oracle_weights = oracle_weights.detach()

            regret_per_sample = mvo_regret(
                predicted_weights=predicted_weights,
                oracle_weights=oracle_weights,
                true_returns=true_returns_cpu,
                risk_factor=risk_factor,
                reduction="none",
            )

            mse_per_sample = (
                predicted_returns_cpu
                - true_returns_cpu
            ).square().mean(dim=-1)

            combined_per_sample = (
                alpha * regret_per_sample
                + (1.0 - alpha)
                * mse_scale
                * mse_per_sample
            )

            predicted_cost = markowitz_cost(
                weights=predicted_weights,
                true_returns=true_returns_cpu,
                risk_factor=risk_factor,
            )

            oracle_cost = markowitz_cost(
                weights=oracle_weights,
                true_returns=true_returns_cpu,
                risk_factor=risk_factor,
            )

            daily_portfolio_return = torch.sum(
                predicted_weights * true_returns_cpu,
                dim=-1,
            )

            portfolio_variance = torch.einsum(
                "bi,bij,bj->b",
                predicted_weights,
                covariance,
                predicted_weights,
            )

            risk_penalty = (
                risk_aversion * portfolio_variance
            )

            active_asset_count = (
                predicted_weights > active_threshold
            ).sum(dim=-1)

            batch_size = features.size(0)

            total_combined += (
                combined_per_sample.sum().item()
            )
            total_regret += (
                regret_per_sample.sum().item()
            )
            total_mse += (
                mse_per_sample.sum().item()
            )
            total_samples += batch_size

            portfolio_returns.extend(
                daily_portfolio_return.tolist()
            )

            for sample_index, date in enumerate(dates):
                daily_rows.append(
                    {
                        "date": date,
                        "portfolio_return":
                            daily_portfolio_return[
                                sample_index
                            ].item(),
                        "portfolio_variance":
                            portfolio_variance[
                                sample_index
                            ].item(),
                        "risk_penalty":
                            risk_penalty[
                                sample_index
                            ].item(),
                        "predicted_cost":
                            predicted_cost[
                                sample_index
                            ].item(),
                        "oracle_cost":
                            oracle_cost[
                                sample_index
                            ].item(),
                        "regret":
                            regret_per_sample[
                                sample_index
                            ].item(),
                        "active_asset_count":
                            int(
                                active_asset_count[
                                    sample_index
                                ].item()
                            ),
                    }
                )

                for asset_index, ticker in enumerate(
                    tickers
                ):
                    true_return = true_returns_cpu[
                        sample_index,
                        asset_index,
                    ].item()

                    predicted_return = (
                        predicted_returns_cpu[
                            sample_index,
                            asset_index,
                        ].item()
                    )

                    predicted_weight = (
                        predicted_weights[
                            sample_index,
                            asset_index,
                        ].item()
                    )

                    oracle_weight = (
                        oracle_weights[
                            sample_index,
                            asset_index,
                        ].item()
                    )

                    asset_rows.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "true_return":
                                true_return,
                            "predicted_return":
                                predicted_return,
                            "prediction_bias":
                                predicted_return
                                - true_return,
                            "predicted_weight":
                                predicted_weight,
                            "oracle_weight":
                                oracle_weight,
                            "is_up":
                                predicted_return > 0,
                            "is_in":
                                predicted_weight
                                > active_threshold,
                        }
                    )

    if total_samples == 0:
        raise RuntimeError(
            "Test DataLoader에 샘플이 없습니다."
        )

    returns_tensor = torch.tensor(
        portfolio_returns,
        dtype=torch.float64,
    )

    (
        portfolio_metrics,
        wealth,
        drawdown,
    ) = calculate_portfolio_metrics(
        returns_tensor
    )

    daily_dataframe = pd.DataFrame(daily_rows)
    daily_dataframe["wealth"] = wealth.numpy()
    daily_dataframe["drawdown"] = drawdown.numpy()

    asset_dataframe = pd.DataFrame(asset_rows)

    asset_dataframe.to_csv(
        output_dir / "asset_predictions.csv",
        index=False,
    )

    daily_dataframe.to_csv(
        output_dir / "daily_portfolio.csv",
        index=False,
    )

    summary = {
        "dataset": dataset_name,
        "alpha": alpha,
        "lambda": risk_aversion,
        "mse_scale": mse_scale,
        "max_weight": max_weight,
        "active_threshold": active_threshold,
        "seed": seed,
        "best_epoch": int(
            checkpoint["best_epoch"]
        ),
        "validation_combined": float(
            checkpoint["validation_combined"]
        ),
        "validation_regret": float(
            checkpoint["validation_regret"]
        ),
        "validation_mse": float(
            checkpoint["validation_mse"]
        ),
        "test_combined":
            total_combined / total_samples,
        "test_regret":
            total_regret / total_samples,
        "test_mse":
            total_mse / total_samples,
        "average_active_assets": float(
            daily_dataframe[
                "active_asset_count"
            ].mean()
        ),
        **portfolio_metrics,
    }

    with (
        output_dir / "summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

    return summary


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(
            "--alpha는 0 이상 1 이하여야 합니다."
        )

    if args.mse_scale <= 0:
        raise ValueError(
            "--mse-scale은 0보다 커야 합니다."
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
        dropout=args.dropout,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
    ).float().to(device)

    optimizer = torch.optim.AdamW(
        model.return_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    dataset_name = price_path.stem.replace(
        "_adjusted_close",
        "",
    )

    output_dir = (
        PROJECT_ROOT
        / "implementations"
        / "dfl_mvo_lee2025"
        / "outputs"
        / "combined"
        / dataset_name
        / f"alpha_{args.alpha:.2f}"
        / f"lambda_{args.risk_aversion:.2f}"
        / f"maxw_{args.max_weight:.2f}"
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        output_dir / "best_model.pt"
    )

    history_path = (
        output_dir / "history.csv"
    )

    best_validation_combined = float("inf")
    best_epoch = 0
    patience_count = 0
    history_rows: list[dict] = []

    print("=" * 80)
    print("DFL-MVO Combined Loss Training")
    print("=" * 80)
    print("Device:", device)
    print("Alpha:", args.alpha)
    print("MSE scale:", args.mse_scale)
    print("Risk aversion:", args.risk_aversion)
    print("Maximum weight:", args.max_weight)
    print("Active threshold:", args.active_threshold)
    print("Seed:", args.seed)
    print("Train samples:", len(train_set))
    print("Validation samples:", len(validation_set))
    print("Test samples:", len(test_set))
    print("=" * 80)

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        (
            train_combined,
            train_regret,
            train_mse,
        ) = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            alpha=args.alpha,
            mse_scale=args.mse_scale,
            optimizer=optimizer,
            active_threshold=args.active_threshold,
            cap_tolerance=args.cap_tolerance,
            show_diagnostics=True,
        )

        (
            validation_combined,
            validation_regret,
            validation_mse,
        ) = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            alpha=args.alpha,
            mse_scale=args.mse_scale,
            optimizer=None,
            active_threshold=args.active_threshold,
            cap_tolerance=args.cap_tolerance,
            show_diagnostics=False,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_combined": train_combined,
                "train_regret": train_regret,
                "train_mse": train_mse,
                "validation_combined":
                    validation_combined,
                "validation_regret":
                    validation_regret,
                "validation_mse":
                    validation_mse,
            }
        )

        pd.DataFrame(history_rows).to_csv(
            history_path,
            index=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Combined: {train_combined:.8f} | "
            f"Train Regret: {train_regret:.8f} | "
            f"Train MSE: {train_mse:.8f} | "
            f"Validation Combined: "
            f"{validation_combined:.8f} | "
            f"Validation Regret: "
            f"{validation_regret:.8f} | "
            f"Validation MSE: "
            f"{validation_mse:.8f}"
        )

        if (
            validation_combined
            < best_validation_combined
        ):
            best_validation_combined = (
                validation_combined
            )
            best_epoch = epoch
            patience_count = 0

            torch.save(
                {
                    "model_state_dict":
                        model.return_model.state_dict(),
                    "best_epoch":
                        epoch,
                    "validation_combined":
                        validation_combined,
                    "validation_regret":
                        validation_regret,
                    "validation_mse":
                        validation_mse,
                    "alpha":
                        args.alpha,
                    "mse_scale":
                        args.mse_scale,
                    "risk_aversion":
                        args.risk_aversion,
                    "max_weight":
                        args.max_weight,
                    "active_threshold":
                        args.active_threshold,
                    "seed":
                        args.seed,
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

    summary = evaluate_and_save_test_results(
        model=model,
        loader=test_loader,
        device=device,
        tickers=dataset.tickers,
        output_dir=output_dir,
        dataset_name=dataset_name,
        alpha=args.alpha,
        mse_scale=args.mse_scale,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        seed=args.seed,
        active_threshold=args.active_threshold,
        checkpoint=checkpoint,
    )

    print("=" * 80)
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Best Validation Combined: "
        f"{best_validation_combined:.8f}"
    )
    print(
        f"Test Combined: "
        f"{summary['test_combined']:.8f}"
    )
    print(
        f"Test Regret: "
        f"{summary['test_regret']:.8f}"
    )
    print(
        f"Test MSE: "
        f"{summary['test_mse']:.8f}"
    )
    print(
        f"Annualized Return (CAGR): "
        f"{summary['annualized_return_cagr']:.6f}"
    )
    print(
        f"Sharpe Ratio: "
        f"{summary['sharpe_ratio']:.6f}"
    )
    print(
        f"Maximum Drawdown: "
        f"{summary['maximum_drawdown']:.6f}"
    )
    print(
        f"Final Wealth: "
        f"{summary['final_wealth']:.6f}"
    )
    print(
        f"Average Active Assets: "
        f"{summary['average_active_assets']:.4f}"
    )
    print(
        f"Saved directory: {output_dir}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
