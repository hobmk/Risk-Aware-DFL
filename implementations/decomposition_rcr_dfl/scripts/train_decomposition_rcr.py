from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from implementations.eigen_rcr_dfl.src.dataset import (
    chronological_split,
)
from implementations.eigen_rcr_dfl.src.model import EigenRCRDFL
from implementations.eigen_rcr_dfl.src.trainer import fit
from implementations.eigen_rcr_dfl.src.backtest import evaluate_portfolio
from implementations.eigen_rcr_dfl.src.oracle_cache import (
    load_or_build_oracle_cache,
)
from implementations.decomposition_rcr_dfl.src.dataset import (
    DecompositionRCRDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Residual Risk Decomposition & "
            "Reconstruction RCR-DFL"
        )
    )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument(
        "--date-column",
        default="Date",
    )
    parser.add_argument(
        "--return-type",
        choices=["simple", "log"],
        default="simple",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
    )

    # ---------------------------------------------------------
    # CAPM / Market
    # ---------------------------------------------------------
    parser.add_argument(
        "--market-mode",
        choices=["equal_weight", "external"],
        default="equal_weight",
    )
    parser.add_argument(
        "--market-price-csv",
        default=None,
    )
    parser.add_argument(
        "--market-column",
        default=None,
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--no-capm-intercept",
        action="store_true",
    )

    # ---------------------------------------------------------
    # Residual decomposition
    # ---------------------------------------------------------
    parser.add_argument(
        "--decomposition-k",
        type=int,
        default=3,
    )

    # ---------------------------------------------------------
    # Numerical stability
    # ---------------------------------------------------------
    parser.add_argument(
        "--covariance-jitter",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--effective-jitter",
        type=float,
        default=0.0,
    )

    # ---------------------------------------------------------
    # Chronological split
    # ---------------------------------------------------------
    parser.add_argument(
        "--train-end",
        default="2021-12-31",
    )
    parser.add_argument(
        "--validation-end",
        default="2022-12-31",
    )

    # ---------------------------------------------------------
    # Return prediction model
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Portfolio optimization
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # DFL loss
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------
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
        "--gradient-clip-norm",
        type=float,
        default=None,
    )

    # ---------------------------------------------------------
    # Smoke test options
    # ---------------------------------------------------------
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
    )

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Oracle cache
    # ---------------------------------------------------------
    parser.add_argument(
        "--oracle-cache-dir",
        default=(
            "implementations/"
            "decomposition_rcr_dfl/"
            "cache/oracle"
        ),
    )
    parser.add_argument(
        "--rebuild-oracle-cache",
        action="store_true",
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def resolve_device(name: str):
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


def file_signature(
    path: str | Path | None,
) -> dict | None:
    if path is None:
        return None

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {path}"
        )

    stat = path.stat()

    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main():
    args = parse_args()

    if args.decomposition_k < 0:
        raise ValueError(
            "decomposition_k는 0 이상이어야 합니다."
        )

    set_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    # =========================================================
    # Output directory
    # =========================================================
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            Path(
                "implementations/"
                "decomposition_rcr_dfl/"
                "outputs"
            )
            / f"k_{args.decomposition_k}"
            / (
                f"alpha_{args.alpha:.2f}_"
                f"seed_{args.seed}"
            )
        )
    )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"이미 존재합니다: "
                f"{output_dir}"
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

    # =========================================================
    # Dataset
    # =========================================================
    dataset = DecompositionRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type=args.return_type,
        covariance_jitter=(
            args.covariance_jitter
        ),
        market_mode=args.market_mode,
        market_price_csv=(
            args.market_price_csv
        ),
        market_column=args.market_column,
        risk_free_rate=(
            args.risk_free_rate
        ),
        fit_intercept=(
            not args.no_capm_intercept
        ),
        decomposition_k=(
            args.decomposition_k
        ),
        dtype=torch.float64,
    )

    train_dataset, validation_dataset, test_dataset = (
        chronological_split(
            dataset,
            args.train_end,
            args.validation_end,
        )
    )

    print(
        f"N assets: {dataset.n_assets}"
    )
    print(
        f"Train: {len(train_dataset)}"
    )
    print(
        f"Validation: {len(validation_dataset)}"
    )
    print(
        f"Test: {len(test_dataset)}"
    )

    # =========================================================
    # Risk diagnostics
    # =========================================================
    retained = (
        dataset.retained_ratio
    )

    decomposition_error = (
        dataset.decomposition_error_fro
    )

    print(
        f"Mean retained residual variance "
        f"(K={args.decomposition_k}): "
        f"{retained.mean().item():.6f}"
    )

    print(
        f"Mean CAPM decomposition "
        f"Frobenius error: "
        f"{decomposition_error.mean().item():.6e}"
    )

    print(
        f"Max CAPM decomposition "
        f"Frobenius error : "
        f"{decomposition_error.max().item():.6e}"
    )

    # =========================================================
    # DataLoaders
    # =========================================================
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

    # =========================================================
    # Model
    #
    # IMPORTANT:
    #
    # dataset["covariance"] already contains:
    #
    # Sigma_eff^(K)
    #   = Sigma_market
    #   + U_K Lambda_K U_K^T
    #   + D_K
    #
    # Therefore EigenRCRDFL eta MUST be 0.
    # =========================================================
    model = EigenRCRDFL(
        n_assets=dataset.n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        risk_aversion=(
            args.risk_aversion
        ),
        max_weight=args.max_weight,

        # Decomposition model에서는
        # 추가 residual term을 더하면 안 됨.
        eta=0.0,

        effective_jitter=(
            args.effective_jitter
        ),
    ).to(
        device
    )

    # =========================================================
    # Oracle cache
    #
    # Oracle portfolio는 neural-network seed와 무관하다.
    # 동일한 K / data / MVO 설정에서는 seed 42,43,44가
    # 동일 cache를 공유한다.
    # =========================================================
    oracle_cache_config = {
        "cache_version": 1,

        "risk_method": (
            "residual_decomposition_"
            "reconstruction_v1"
        ),

        "price_csv": file_signature(
            args.price_csv
        ),

        "date_column": (
            args.date_column
        ),

        "return_type": (
            args.return_type
        ),

        "lookback": (
            args.lookback
        ),

        "market_mode": (
            args.market_mode
        ),

        "market_price_csv": (
            file_signature(
                args.market_price_csv
            )
            if args.market_mode
            == "external"
            else None
        ),

        "market_column": (
            args.market_column
        ),

        "risk_free_rate": (
            args.risk_free_rate
        ),

        "fit_intercept": (
            not args.no_capm_intercept
        ),

        "covariance_jitter": (
            args.covariance_jitter
        ),

        "effective_jitter": (
            args.effective_jitter
        ),

        "decomposition_k": (
            args.decomposition_k
        ),

        # EigenRCRDFL interface를 재사용하지만
        # residual addition은 사용하지 않는다.
        "model_eta": 0.0,

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

        "first_target_date": (
            dataset.target_dates[0]
            .strftime("%Y-%m-%d")
        ),

        "last_target_date": (
            dataset.target_dates[-1]
            .strftime("%Y-%m-%d")
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
        cache_dir=args.oracle_cache_dir,
        cache_config=oracle_cache_config,
        force_rebuild=(
            args.rebuild_oracle_cache
        ),
    )

    # =========================================================
    # Optimizer
    # =========================================================
    optimizer = torch.optim.AdamW(
        model.return_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # =========================================================
    # Config
    # =========================================================
    config = vars(
        args
    ).copy()

    config.update({
        "n_assets": (
            dataset.n_assets
        ),

        "market_name": (
            dataset.market_name
        ),

        "device_resolved": (
            str(device)
        ),

        "risk_method": (
            "residual_decomposition_"
            "reconstruction"
        ),

        "decomposition_k": (
            args.decomposition_k
        ),

        "model_eta": 0.0,

        "mean_retained_ratio": (
            retained.mean().item()
        ),

        "mean_decomposition_error_fro": (
            decomposition_error
            .mean()
            .item()
        ),

        "max_decomposition_error_fro": (
            decomposition_error
            .max()
            .item()
        ),

        "risk_definition": (
            "Sigma_eff = "
            "Sigma_market + "
            "U_K Lambda_K U_K^T + D_K"
        ),

        "residual_diagonal_definition": (
            "D_K = Diag(diag("
            "Sigma_epsilon - "
            "U_K Lambda_K U_K^T))"
        ),

        "oracle_cache_path": (
            str(oracle_cache_path)
        ),

        "oracle_cache_hit": (
            oracle_cache_hit
        ),
    })

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================
    # Training
    # =========================================================
    result = fit(
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
        gradient_clip_norm=(
            args.gradient_clip_norm
        ),
        max_train_batches=(
            args.max_train_batches
        ),
        max_validation_batches=(
            args.max_validation_batches
        ),
        max_test_batches=(
            args.max_test_batches
        ),
    )

    print(
        "\nTraining Summary"
    )

    print(
        f"Best Epoch         : "
        f"{result.best_epoch}"
    )

    print(
        f"Best Validation    : "
        f"{result.best_validation_loss:.8f}"
    )

    print(
        f"Test Total Loss    : "
        f"{result.test_metrics.total_loss:.8f}"
    )

    print(
        f"Test MSE           : "
        f"{result.test_metrics.mse:.8f}"
    )

    print(
        f"Test Regret        : "
        f"{result.test_metrics.regret:.8f}"
    )

    # =========================================================
    # Portfolio backtest
    # =========================================================
    portfolio_metrics = evaluate_portfolio(
        model=model,
        dataloader=test_loader,
        device=device,
        tickers=dataset.tickers,
        output_dir=output_dir,
    )

    print(
        "\nPortfolio Backtest"
    )

    print(
        f"Cumulative Return : "
        f"{portfolio_metrics['cumulative_return']:.6f}"
    )

    print(
        f"Annualized Return : "
        f"{portfolio_metrics['annualized_return']:.6f}"
    )

    print(
        f"Volatility        : "
        f"{portfolio_metrics['volatility']:.6f}"
    )

    print(
        f"Sharpe            : "
        f"{portfolio_metrics['sharpe']:.6f}"
    )

    print(
        f"Max Drawdown      : "
        f"{portfolio_metrics['max_drawdown']:.6f}"
    )

    print(
        f"Calmar            : "
        f"{portfolio_metrics['calmar']:.6f}"
    )

    print(
        f"Annual Turnover   : "
        f"{portfolio_metrics['annualized_turnover']:.6f}"
    )

    print(
        f"Active Assets     : "
        f"{portfolio_metrics['active_assets']:.3f}"
    )


if __name__ == "__main__":
    main()