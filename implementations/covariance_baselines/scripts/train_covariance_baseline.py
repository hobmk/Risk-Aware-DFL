from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from implementations.common.portfolio_evaluation import evaluate_portfolio
from implementations.covariance_baselines.src.dataset import (
    CovarianceBaselineDataset,
)
from implementations.dfl_mvo_lee2025.src.dataset import (
    FeatureStandardizedSubset,
    chronological_split,
    fit_feature_standardizer,
)
from implementations.eigen_rcr_dfl.src.model import EigenRCRDFL
from implementations.eigen_rcr_dfl.src.oracle_cache import (
    load_or_build_oracle_cache,
)
from implementations.eigen_rcr_dfl.src.trainer import fit


def parse_args():
    parser = argparse.ArgumentParser(
        description="DFL with covariance shrinkage baselines"
    )

    parser.add_argument(
        "--method",
        choices=["ridge", "ledoit_wolf"],
        required=True,
    )

    parser.add_argument(
        "--ridge-shrinkage",
        type=float,
        default=0.10,
    )

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
        "--train-end",
        default="2021-12-31",
    )

    parser.add_argument(
        "--validation-end",
        default="2022-12-31",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--mse-scale",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
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
        "--patience",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--risk-cache-dir",
        default=(
            "implementations/"
            "covariance_baselines/"
            "cache/risk_matrix"
        ),
    )

    parser.add_argument(
        "--oracle-cache-dir",
        default=(
            "implementations/"
            "covariance_baselines/"
            "cache/oracle"
        ),
    )

    parser.add_argument(
        "--force-rebuild-risk-cache",
        action="store_true",
    )

    parser.add_argument(
        "--force-rebuild-oracle-cache",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA를 사용할 수 없습니다."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def default_output_dir(args) -> Path:
    root = Path(
        "implementations/"
        "covariance_baselines/"
        "outputs"
    )

    if args.method == "ridge":
        method_dir = (
            root
            / "ridge"
            / f"rho_{args.ridge_shrinkage:.2f}"
        )

    else:
        method_dir = (
            root
            / "ledoit_wolf"
        )

    return (
        method_dir
        / f"lambda_{args.risk_aversion:g}"
        / f"seed_{args.seed}"
    )


def collect_portfolio_outputs(
    model: EigenRCRDFL,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()

    dates = []
    all_weights = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(
                device=device,
                dtype=torch.float32,
            )

            covariance = batch["covariance"].to(
                device="cpu",
                dtype=torch.float64,
            )

            eigen_risk = batch["eigen_risk"].to(
                device="cpu",
                dtype=torch.float64,
            )

            output = model(
                features=features,
                covariance=covariance,
                eigen_risk=eigen_risk,
            )

            all_weights.append(
                output.predicted_weights
                .detach()
                .cpu()
                .numpy()
            )

            all_predictions.append(
                output.predicted_returns
                .detach()
                .cpu()
                .numpy()
            )

            all_targets.append(
                batch["target"]
                .detach()
                .cpu()
                .numpy()
            )

            dates.extend(
                list(
                    batch["target_date"]
                )
            )

    weights = np.concatenate(
        all_weights,
        axis=0,
    )

    predictions = np.concatenate(
        all_predictions,
        axis=0,
    )

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    portfolio_returns = np.sum(
        weights * targets,
        axis=1,
    )

    return (
        dates,
        weights,
        predictions,
        targets,
        portfolio_returns,
    )


def evaluate_split(
    model: EigenRCRDFL,
    loader: DataLoader,
    device: torch.device,
):
    (
        dates,
        weights,
        predictions,
        targets,
        portfolio_returns,
    ) = collect_portfolio_outputs(
        model=model,
        loader=loader,
        device=device,
    )

    metrics = evaluate_portfolio(
        portfolio_returns=portfolio_returns,
        weights=weights,
        asset_returns=targets,
        cost_bps=(0.0, 5.0, 10.0),
        active_threshold=1e-3,
    )

    return (
        dates,
        weights,
        predictions,
        targets,
        portfolio_returns,
        metrics,
    )


def save_test_outputs(
    output_dir: Path,
    tickers: list[str],
    dates,
    weights: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    portfolio_returns: np.ndarray,
    metrics: dict,
) -> None:
    weights_df = pd.DataFrame(
        weights,
        columns=tickers,
    )

    weights_df.insert(
        0,
        "date",
        dates,
    )

    weights_df.to_csv(
        output_dir
        / "portfolio_weights.csv",
        index=False,
    )

    predictions_df = pd.DataFrame(
        predictions,
        columns=tickers,
    )

    predictions_df.insert(
        0,
        "date",
        dates,
    )

    predictions_df.to_csv(
        output_dir
        / "predicted_returns.csv",
        index=False,
    )

    targets_df = pd.DataFrame(
        targets,
        columns=tickers,
    )

    targets_df.insert(
        0,
        "date",
        dates,
    )

    targets_df.to_csv(
        output_dir
        / "true_returns.csv",
        index=False,
    )

    pd.DataFrame({
        "date": dates,
        "portfolio_return": (
            portfolio_returns
        ),
    }).to_csv(
        output_dir
        / "portfolio_daily.csv",
        index=False,
    )

    pd.DataFrame([
        metrics
    ]).to_csv(
        output_dir
        / "portfolio_summary.csv",
        index=False,
    )


def main():
    args = parse_args()

    if (
        args.method == "ridge"
        and not (
            0.0
            <= args.ridge_shrinkage
            <= 1.0
        )
    ):
        raise ValueError(
            "--ridge-shrinkage는 "
            "0 이상 1 이하여야 합니다."
        )

    set_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(args)
    )

    completed_file = (
        output_dir
        / "portfolio_summary.csv"
    )

    if (
        completed_file.exists()
        and not args.overwrite
    ):
        print(
            "SKIP | completed run "
            f"already exists: {output_dir}"
        )
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                "미완료 output directory가 "
                f"이미 존재합니다: {output_dir}"
            )

        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Method: {args.method}"
    )

    if args.method == "ridge":
        print(
            "Ridge shrinkage: "
            f"{args.ridge_shrinkage:.4f}"
        )

    dataset = CovarianceBaselineDataset(
        price_csv=args.price_csv,
        method=args.method,
        ridge_shrinkage=(
            args.ridge_shrinkage
        ),
        lookback=args.lookback,
        return_type="simple",
        covariance_jitter=1e-6,
        dtype=torch.float64,
        risk_cache_dir=(
            args.risk_cache_dir
        ),
        force_rebuild_risk_cache=(
            args.force_rebuild_risk_cache
        ),
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = chronological_split(
        dataset=dataset,
        train_end=args.train_end,
        validation_end=(
            args.validation_end
        ),
    )

    (
        feature_mean,
        feature_std,
    ) = fit_feature_standardizer(
        dataset=dataset,
        train_subset=(
            train_dataset
        ),
    )

    train_dataset = (
        FeatureStandardizedSubset(
            train_dataset,
            feature_mean,
            feature_std,
        )
    )

    validation_dataset = (
        FeatureStandardizedSubset(
            validation_dataset,
            feature_mean,
            feature_std,
        )
    )

    test_dataset = (
        FeatureStandardizedSubset(
            test_dataset,
            feature_mean,
            feature_std,
        )
    )

    torch.save(
        {
            "feature_mean": (
                feature_mean
            ),
            "feature_std": (
                feature_std
            ),
        },
        output_dir
        / "feature_standardizer.pt",
    )

    print(
        "Input standardization: ON"
    )

    print(
        f"N assets: {dataset.n_assets}"
    )

    print(
        f"Train: {len(train_dataset)}"
    )

    print(
        "Validation: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Test: {len(test_dataset)}"
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = EigenRCRDFL(
        n_assets=dataset.n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        risk_aversion=(
            args.risk_aversion
        ),
        max_weight=args.max_weight,

        # covariance 자체가
        # Ridge/LW estimator이므로
        # 추가 residual risk 없음.
        eta=0.0,

        effective_jitter=0.0,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.return_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    oracle_cache_config = {
        "cache_version": (
            "covariance_baseline_oracle_v1"
        ),
        "method": args.method,
        "risk_cache_key": (
            dataset.risk_cache_key
        ),
        "ridge_shrinkage": (
            args.ridge_shrinkage
            if args.method == "ridge"
            else None
        ),
        "lookback": args.lookback,
        "risk_aversion": (
            args.risk_aversion
        ),
        "max_weight": (
            args.max_weight
        ),
        "n_assets": (
            dataset.n_assets
        ),
        "n_samples": (
            len(dataset)
        ),
    }

    (
        oracle_weights_cache,
        oracle_cache_path,
        oracle_cache_hit,
    ) = load_or_build_oracle_cache(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        cache_dir=(
            args.oracle_cache_dir
        ),
        cache_config=(
            oracle_cache_config
        ),
        force_rebuild=(
            args.force_rebuild_oracle_cache
        ),
    )

    print(
        "Oracle cache: "
        f"{'HIT' if oracle_cache_hit else 'BUILD'}"
    )

    print(
        "Oracle cache path: "
        f"{oracle_cache_path}"
    )

    config = vars(
        args
    ).copy()

    config.update({
        "n_assets": (
            dataset.n_assets
        ),
        "standardize_inputs": True,
        "feature_standardizer_file": (
            "feature_standardizer.pt"
        ),
        "feature_mean": (
            feature_mean.tolist()
        ),
        "feature_std": (
            feature_std.tolist()
        ),
        "risk_method": (
            args.method
        ),
        "risk_cache_key": (
            dataset.risk_cache_key
        ),
        "risk_cache_path": (
            dataset.risk_cache_path
        ),
        "risk_cache_hit": bool(
            dataset.risk_cache_hit
        ),
        "oracle_cache_path": str(
            oracle_cache_path
        ),
        "oracle_cache_hit": bool(
            oracle_cache_hit
        ),
    })

    if (
        args.method
        == "ledoit_wolf"
    ):
        config[
            "mean_ledoit_wolf_shrinkage"
        ] = float(
            dataset
            .ledoit_wolf_shrinkages
            .mean()
            .item()
        )

    with open(
        output_dir
        / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    fit(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        alpha=args.alpha,
        mse_scale=args.mse_scale,
        output_dir=output_dir,
        oracle_weights_cache=(
            oracle_weights_cache
        ),
    )

    (
        _,
        _,
        _,
        _,
        _,
        validation_metrics,
    ) = evaluate_split(
        model=model,
        loader=validation_loader,
        device=device,
    )

    pd.DataFrame([
        validation_metrics
    ]).to_csv(
        output_dir
        / "validation_portfolio_summary.csv",
        index=False,
    )

    (
        dates,
        weights,
        predictions,
        targets,
        portfolio_returns,
        test_metrics,
    ) = evaluate_split(
        model=model,
        loader=test_loader,
        device=device,
    )

    save_test_outputs(
        output_dir=output_dir,
        tickers=dataset.tickers,
        dates=dates,
        weights=weights,
        predictions=predictions,
        targets=targets,
        portfolio_returns=(
            portfolio_returns
        ),
        metrics=test_metrics,
    )

    print(
        "\nValidation Portfolio"
    )

    print(
        "Sharpe            : "
        f"{validation_metrics['gross_sharpe']:.6f}"
    )

    print(
        "Annualized Return : "
        f"{validation_metrics['gross_annualized_return']:.6f}"
    )

    print(
        "\nTest Portfolio"
    )

    print(
        "Annualized Return : "
        f"{test_metrics['gross_annualized_return']:.6f}"
    )

    print(
        "Volatility        : "
        f"{test_metrics['gross_volatility']:.6f}"
    )

    print(
        "Sharpe            : "
        f"{test_metrics['gross_sharpe']:.6f}"
    )

    print(
        "Sortino           : "
        f"{test_metrics['gross_sortino']:.6f}"
    )

    print(
        "Max Drawdown      : "
        f"{test_metrics['gross_max_drawdown']:.6f}"
    )

    print(
        "Annual Turnover   : "
        f"{test_metrics['annualized_turnover']:.6f}"
    )

    print(
        "HHI               : "
        f"{test_metrics['hhi']:.6f}"
    )

    print(
        "Effective N       : "
        f"{test_metrics['effective_n']:.3f}"
    )


if __name__ == "__main__":
    main()