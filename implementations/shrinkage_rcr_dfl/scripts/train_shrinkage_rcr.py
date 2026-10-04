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

from implementations.shrinkage_rcr_dfl.src.dataset import (
    ShrinkageRCRDataset,
)
from implementations.eigen_rcr_dfl.src.model import EigenRCRDFL
from implementations.eigen_rcr_dfl.src.trainer import fit
from implementations.eigen_rcr_dfl.src.backtest import evaluate_portfolio
from implementations.eigen_rcr_dfl.src.oracle_cache import (
    load_or_build_oracle_cache,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Eigenvalue Shrinkage RCR-DFL"
    )

    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--return-type", choices=["simple", "log"], default="simple")
    parser.add_argument("--lookback", type=int, default=60)

    parser.add_argument("--market-mode", choices=["equal_weight", "external"], default="equal_weight")
    parser.add_argument("--market-price-csv", default=None)
    parser.add_argument("--market-column", default=None)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--no-capm-intercept", action="store_true")

    parser.add_argument(
        "--shrinkage-c",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--scale-mode",
        choices=["none", "preserve_residual_trace", "match_base_trace"],
        default="none",
    )

    parser.add_argument("--covariance-jitter", type=float, default=1e-6)
    parser.add_argument("--effective-jitter", type=float, default=0.0)

    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--validation-end", default="2022-12-31")

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.5)

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mse-scale", type=float, default=30.0)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--gradient-clip-norm", type=float, default=None)

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument(
        "--oracle-cache-dir",
        default="implementations/shrinkage_rcr_dfl/cache/oracle",
    )

    parser.add_argument(
        "--rebuild-oracle-cache",
        action="store_true",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")

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
            raise RuntimeError("CUDA를 사용할 수 없습니다.")
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

    set_seed(args.seed)

    device = resolve_device(
        args.device
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            Path(
                "implementations/"
                "shrinkage_rcr_dfl/"
                "outputs"
            )
            / f"c_{args.shrinkage_c:.2f}"
            / f"scale_{args.scale_mode}"
            / f"eta_{args.eta:.2f}"
            / f"alpha_{args.alpha:.2f}_seed_{args.seed}"
        )
    )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"이미 존재합니다: {output_dir}"
            )

        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(f"Device: {device}")

    dataset = ShrinkageRCRDataset(
        price_csv=args.price_csv,
        lookback=args.lookback,
        date_column=args.date_column,
        return_type=args.return_type,
        covariance_jitter=args.covariance_jitter,
        market_mode=args.market_mode,
        market_price_csv=args.market_price_csv,
        market_column=args.market_column,
        risk_free_rate=args.risk_free_rate,
        fit_intercept=not args.no_capm_intercept,
        shrinkage_c=args.shrinkage_c,
        scale_mode=args.scale_mode,
        dtype=torch.float64,
    )

    train_dataset, validation_dataset, test_dataset = (
        chronological_split(
            dataset,
            args.train_end,
            args.validation_end,
        )
    )

    print(f"N assets: {dataset.n_assets}")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(validation_dataset)}")
    print(f"Test: {len(test_dataset)}")

    retained = dataset.retained_ratio

    print(
        f"Mean raw shrinkage trace ratio "
        f"(c={args.shrinkage_c}): "
        f"{retained.mean().item():.4f}"
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

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
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        eta=args.eta,
        effective_jitter=args.effective_jitter,
    ).to(device)

    oracle_cache_config = {
        "cache_version": 1,
        "risk_method": "adaptive_eigenvalue_shrinkage_v1",

        "price_csv": file_signature(
            args.price_csv
        ),

        "date_column": args.date_column,
        "return_type": args.return_type,
        "lookback": args.lookback,

        "market_mode": args.market_mode,

        "market_price_csv": (
            file_signature(
                args.market_price_csv
            )
            if args.market_mode == "external"
            else None
        ),

        "market_column": args.market_column,

        "risk_free_rate": args.risk_free_rate,
        "fit_intercept": (
            not args.no_capm_intercept
        ),

        "covariance_jitter": (
            args.covariance_jitter
        ),

        "shrinkage_c": (
            args.shrinkage_c
        ),

        "scale_mode": (
            args.scale_mode
        ),

        "eta": args.eta,

        "risk_aversion": (
            args.risk_aversion
        ),

        "max_weight": (
            args.max_weight
        ),

        "effective_jitter": (
            args.effective_jitter
        ),

        "n_assets": dataset.n_assets,
        "n_samples": len(dataset),

        "first_target_date": (
            dataset.target_dates[0]
            .strftime("%Y-%m-%d")
        ),

        "last_target_date": (
            dataset.target_dates[-1]
            .strftime("%Y-%m-%d")
        ),
    }

    oracle_weights_cache, oracle_cache_path, oracle_cache_hit = (
        load_or_build_oracle_cache(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            cache_dir=args.oracle_cache_dir,
            cache_config=oracle_cache_config,
            force_rebuild=args.rebuild_oracle_cache,
        )
    )

    optimizer = torch.optim.AdamW(
        model.return_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config = vars(args).copy()

    config.update({
        "n_assets": dataset.n_assets,
        "market_name": dataset.market_name,
        "device_resolved": str(device),

        "risk_method": (
            "adaptive_eigenvalue_shrinkage"
        ),

        "mean_retained_ratio": (
            retained.mean().item()
        ),

        "risk_definition": (
            "Sigma_eff = Sigma + eta * "
            "U diag(lambda^2 / "
            "(lambda + c * mean(lambda))) U^T"
        ),

        "oracle_cache_path": str(
            oracle_cache_path
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

    fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        alpha=args.alpha,
        mse_scale=args.mse_scale,
        output_dir=output_dir,
        oracle_weights_cache=oracle_weights_cache,
        gradient_clip_norm=args.gradient_clip_norm,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        max_test_batches=args.max_test_batches,
    
    )

    portfolio_metrics = evaluate_portfolio(
        model=model,
        dataloader=test_loader,
        device=device,
        tickers=dataset.tickers,
        output_dir=output_dir,
    )

    print("\nPortfolio Backtest")
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