from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from implementations.eigen_rcr_dfl.src.backtest import evaluate_portfolio
from implementations.eigen_rcr_dfl.src.dataset import chronological_split
from implementations.eigen_rcr_dfl.src.model import EigenRCRDFL
from implementations.eigen_rcr_dfl.src.oracle_cache import load_or_build_oracle_cache
from implementations.eigen_rcr_dfl.src.trainer import fit
from implementations.dfl_mvo_lee2025.src.dataset import (
    FeatureStandardizedSubset,
    fit_feature_standardizer,
)

from implementations.rank_tail_rcr_dfl.src.dataset import RankTailRCRDataset
from implementations.rank_tail_rcr_dfl.src.cache_config import (
    build_oracle_cache_config,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank-Tail Residual Risk DFL"
    )

    parser.add_argument(
        "--price-csv",
        default="data/raw/dow30_adjusted_close.csv",
    )
    parser.add_argument("--date-column", default="Date")
    parser.add_argument(
        "--return-type",
        choices=["simple", "log"],
        default="simple",
    )
    parser.add_argument("--lookback", type=int, default=60)

    parser.add_argument(
        "--market-mode",
        choices=["equal_weight", "external"],
        default="equal_weight",
    )
    parser.add_argument("--market-price-csv", default=None)
    parser.add_argument("--market-column", default=None)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--no-capm-intercept", action="store_true")

    parser.add_argument("--tail-q", type=float, default=0.20)
    parser.add_argument(
        "--tail-side",
        choices=["down", "up"],
        default="down",
    )

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

    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--validation-end", default="2022-12-31")

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.2)

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mse-scale", type=float, default=30.0)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=None,
    )

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

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--risk-cache-dir",
        default=(
            "implementations/"
            "rank_tail_rcr_dfl/"
            "cache/risk_matrix"
        ),
    )

    parser.add_argument(
        "--force-rebuild-risk-cache",
        action="store_true",
    )

    parser.add_argument(
        "--oracle-cache-dir",
        default=(
            "implementations/"
            "rank_tail_rcr_dfl/"
            "cache/oracle"
        ),
    )
    parser.add_argument(
        "--force-rebuild-oracle-cache",
        action="store_true",
    )

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")

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
            raise RuntimeError("CUDA를 사용할 수 없습니다.")
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            Path("implementations/rank_tail_rcr_dfl/outputs")
            / f"q_{args.tail_q:.2f}"
            / f"lambda_{args.risk_aversion:g}"
            / f"alpha_{args.alpha:.2f}_seed_{args.seed}"
        )
    )

    completed_file = (
        output_dir
        / "portfolio_summary.csv"
    )

    if completed_file.exists():
        print(
            f"SKIP | completed run already exists: "
            f"{output_dir}"
        )
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"미완료 output directory가 이미 존재합니다: "
                f"{output_dir}\n"
                f"다시 실행하려면 --overwrite를 사용하세요."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(f"Device: {device}")

    dataset = RankTailRCRDataset(
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
        tail_q=args.tail_q,
        tail_side=args.tail_side,
        risk_cache_dir=args.risk_cache_dir,
        force_rebuild_risk_cache=(
            args.force_rebuild_risk_cache
        ),
        dtype=torch.float64,
    )

    train_dataset, validation_dataset, test_dataset = chronological_split(
        dataset,
        args.train_end,
        args.validation_end,
    )

    feature_mean, feature_std = fit_feature_standardizer(
        dataset=dataset,
        train_subset=train_dataset,
    )
    train_dataset = FeatureStandardizedSubset(
        train_dataset,
        feature_mean,
        feature_std,
    )
    validation_dataset = FeatureStandardizedSubset(
        validation_dataset,
        feature_mean,
        feature_std,
    )
    test_dataset = FeatureStandardizedSubset(
        test_dataset,
        feature_mean,
        feature_std,
    )
    torch.save(
        {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        output_dir / "feature_standardizer.pt",
    )

    print("Input standardization: ON")
    print(f"N assets: {dataset.n_assets}")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(validation_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print(f"Tail q: {args.tail_q:.2f}")
    print(f"Tail side: {args.tail_side}")
    print(
    f"Risk cache: "
    f"{'HIT' if dataset.risk_cache_hit else 'BUILD'}"
    )

    print(
        f"Risk cache key: "
        f"{dataset.risk_cache_key}"
    )

    print(
        f"Risk cache path: "
        f"{dataset.risk_cache_path}"
    )

    print(
        "Mean tail fraction: "
        f"{dataset.tail_fraction.mean().item():.6f}"
    )
    print(
        "Mean offdiag RT dependence: "
        f"{dataset.mean_offdiag_dependence.mean().item():.6f}"
    )
    print(
        "Mean |offdiag RT dependence|: "
        f"{dataset.mean_abs_offdiag_dependence.mean().item():.6f}"
    )
    print(
        "Max residual diagonal preservation error: "
        f"{dataset.diagonal_preservation_error.max().item():.3e}"
    )
    print(
        "Mean CAPM decomposition Fro error: "
        f"{dataset.capm_decomposition_error.mean().item():.3e}"
    )
    print(
        "Min effective covariance eigenvalue: "
        f"{dataset.minimum_effective_eigenvalue.min().item():.3e}"
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

    # Rank-Tail Sigma_eff가 dataset의 covariance에 이미 들어 있다.
    # 따라서 기존 EigenRCRDFL에서는 eta=0으로 설정한다.
    model = EigenRCRDFL(
        n_assets=dataset.n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        eta=0.0,
        effective_jitter=args.effective_jitter,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.return_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ---------------------------------------------------------
    # Oracle cache
    # ---------------------------------------------------------

    oracle_cache_config = (
        build_oracle_cache_config(
            dataset=dataset,
            price_csv=args.price_csv,
            date_column=args.date_column,
            lookback=args.lookback,
            return_type=args.return_type,
            covariance_jitter=(
                args.covariance_jitter
            ),
            effective_jitter=(
                args.effective_jitter
            ),
            market_mode=args.market_mode,
            market_price_csv=(
                args.market_price_csv
            ),
            market_column=(
                args.market_column
            ),
            risk_free_rate=(
                args.risk_free_rate
            ),
            fit_intercept=(
                not args.no_capm_intercept
            ),
            tail_q=args.tail_q,
            tail_side=args.tail_side,
            risk_aversion=(
                args.risk_aversion
            ),
            max_weight=args.max_weight,
        )
    )

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
        force_rebuild=args.force_rebuild_oracle_cache,
    )

    print(
        f"Oracle cache: "
        f"{'HIT' if oracle_cache_hit else 'BUILD'}"
    )
    print(
        f"Oracle cache path: "
        f"{oracle_cache_path}"
    )

    # ---------------------------------------------------------
    # Config 저장
    # ---------------------------------------------------------

    config = vars(args).copy()

    config.update({
        "n_assets": dataset.n_assets,
        "market_name": dataset.market_name,
        "device_resolved": str(device),
        "standardize_inputs": True,
        "feature_standardizer_file": "feature_standardizer.pt",
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),

        "risk_method": (
            "rank_tail_residual_dependence_v1"
        ),

        "model_eta": 0.0,
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
        "oracle_cache_config": (
            oracle_cache_config
        ),

        "mean_tail_fraction": (
            dataset.tail_fraction.mean().item()
        ),

        "mean_offdiag_rank_tail_dependence": (
            dataset.mean_offdiag_dependence.mean().item()
        ),

        "mean_abs_offdiag_rank_tail_dependence": (
            dataset.mean_abs_offdiag_dependence.mean().item()
        ),

        "max_diagonal_preservation_error": (
            dataset.diagonal_preservation_error.max().item()
        ),

        "mean_capm_decomposition_fro_error": (
            dataset.capm_decomposition_error.mean().item()
        ),

        "minimum_effective_eigenvalue": (
            dataset.minimum_effective_eigenvalue.min().item()
        ),

        "risk_definition": (
            "Sigma_eff = "
            "beta beta^T Var(r_m) + "
            "D_sigma R_rank_tail D_sigma + "
            "jitter I"
        ),
    })

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Portfolio backtest
    # ---------------------------------------------------------

    portfolio_metrics = evaluate_portfolio(
        model=model,
        dataloader=test_loader,
        device=device,
        tickers=dataset.tickers,
        output_dir=output_dir,
    )

    print("\nPortfolio Backtest")

    for label, key in [
        ("Cumulative Return", "cumulative_return"),
        ("Annualized Return", "annualized_return"),
        ("Volatility", "volatility"),
        ("Sharpe", "sharpe"),
        ("Max Drawdown", "max_drawdown"),
        ("Calmar", "calmar"),
        ("Annual Turnover", "annualized_turnover"),
        ("Active Assets", "active_assets"),
    ]:
        print(
            f"{label:18s}: "
            f"{portfolio_metrics[key]:.6f}"
        )


if __name__ == "__main__":
    main()