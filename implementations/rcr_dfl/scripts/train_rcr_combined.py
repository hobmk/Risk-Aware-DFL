from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from implementations.rcr_dfl.src.dataset import (
    FeatureStandardizedSubset,
    RCRRollingMVODataset,
    chronological_split,
    fit_feature_standardizer,
)
from implementations.rcr_dfl.src.decision_model import RCRMLPWithMarkowitz
from implementations.rcr_dfl.src.trainer import fit
from implementations.rcr_dfl.src.reporting import generate_run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual-correlation-aware DFL training")
    parser.add_argument("--price-csv", default="data/raw/dow30_adjusted_close.csv")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--return-type", choices=["simple", "log"], default="simple")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--covariance-jitter", type=float, default=1e-6)
    parser.add_argument("--market-mode", choices=["equal_weight", "external"], default="equal_weight")
    parser.add_argument("--market-price-csv", default=None)
    parser.add_argument("--market-column", default=None)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--no-capm-intercept", action="store_true", help="CAPM alpha를 0으로 고정합니다.")
    parser.add_argument("--residual-correlation-shrinkage", type=float, default=0.0)
    parser.add_argument("--correlation-scaling", choices=["none", "trace"], default="trace")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--validation-end", default="2022-12-31")
    parser.add_argument(
        "--standardize-inputs",
        action="store_true",
        help="Train 구간 통계로 MLP 입력 features만 종목별 표준화합니다.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--effective-jitter", type=float, default=0.0)
    parser.add_argument("--project-psd", action="store_true")
    parser.add_argument("--minimum-eigenvalue", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mse-scale", type=float, default=15.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--make-report",
        action="store_true",
        help="Best checkpoint로 포트폴리오 성과와 그래프를 자동 생성합니다.",
    )
    parser.add_argument(
        "--baseline-run-dir",
        default=None,
        help="비교할 기존 DFL-MVO seed 실행 폴더입니다.",
    )
    parser.add_argument(
        "--baseline-label",
        default="DFL-MVO",
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
        "--report-dpi",
        type=int,
        default=300,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda를 지정했지만 CUDA를 사용할 수 없습니다.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_output_dir(path: str | Path, overwrite: bool) -> Path:
    output_dir = Path(path)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"출력 폴더가 이미 존재합니다: {output_dir}\n덮어쓰려면 --overwrite를 추가하세요."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha는 0 이상 1 이하여야 합니다.")
    if args.eta < 0:
        raise ValueError("eta는 0 이상이어야 합니다.")
    if not 0.0 <= args.residual_correlation_shrinkage <= 1.0:
        raise ValueError("residual-correlation-shrinkage는 0 이상 1 이하여야 합니다.")
    if args.batch_size <= 0:
        raise ValueError("batch_size는 0보다 커야 합니다.")
    if args.num_workers != 0:
        raise ValueError("현재 단계에서는 Windows와 CVXPYLayer 안정성을 위해 --num-workers 0만 사용하세요.")
    if args.active_threshold < 0:
        raise ValueError("active-threshold는 0 이상이어야 합니다.")
    if args.cap_tolerance < 0:
        raise ValueError("cap-tolerance는 0 이상이어야 합니다.")
    if args.report_dpi <= 0:
        raise ValueError("report-dpi는 0보다 커야 합니다.")
    if args.baseline_run_dir is not None and not args.make_report:
        raise ValueError(
            "--baseline-run-dir를 사용하려면 --make-report도 함께 지정하세요."
        )

    if args.correlation_scaling == "none":
        print("WARNING: correlation-scaling=none은 Sigma와 A_res의 단위가 다르므로 진단용으로만 권장합니다.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    default_output_dir = (
        Path("implementations/rcr_dfl/outputs/combined")
        / f"eta_{args.eta:.2f}_rho_{args.residual_correlation_shrinkage:.2f}"
        / f"alpha_{args.alpha:.2f}_seed_{args.seed}"
    )
    output_dir = prepare_output_dir(args.output_dir or default_output_dir, args.overwrite)

    dataset = RCRRollingMVODataset(
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
        residual_correlation_shrinkage=args.residual_correlation_shrinkage,
        correlation_scaling=args.correlation_scaling,
        dtype=torch.float64,
    )
    train_subset, validation_subset, test_subset = chronological_split(
        dataset=dataset,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )

    feature_mean: torch.Tensor | None = None
    feature_std: torch.Tensor | None = None

    if args.standardize_inputs:
        feature_mean, feature_std = fit_feature_standardizer(
            dataset=dataset,
            train_subset=train_subset,
        )
        train_dataset = FeatureStandardizedSubset(
            train_subset,
            feature_mean,
            feature_std,
        )
        validation_dataset = FeatureStandardizedSubset(
            validation_subset,
            feature_mean,
            feature_std,
        )
        test_dataset = FeatureStandardizedSubset(
            test_subset,
            feature_mean,
            feature_std,
        )
    else:
        train_dataset = train_subset
        validation_dataset = validation_subset
        test_dataset = test_subset

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = RCRMLPWithMarkowitz(
        n_assets=dataset.n_assets,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        eta=args.eta,
        effective_jitter=args.effective_jitter,
        project_psd=args.project_psd,
        minimum_eigenvalue=args.minimum_eigenvalue,
    ).float()
    optimizer = torch.optim.AdamW(
        model.return_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    standardizer_path: Path | None = None

    if args.standardize_inputs:
        if feature_mean is None or feature_std is None:
            raise RuntimeError("표준화 통계가 생성되지 않았습니다.")

        standardizer_path = output_dir / "feature_standardizer.pt"
        torch.save(
            {
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "tickers": dataset.tickers,
                "train_end": args.train_end,
            },
            standardizer_path,
        )

    config = vars(args).copy()
    config.update(
        {
            "resolved_device": str(device),
            "resolved_output_dir": str(output_dir),
            "n_assets": dataset.n_assets,
            "market_name": dataset.market_name,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "test_samples": len(test_dataset),
            "first_target_date": str(dataset.target_dates[0].date()),
            "last_target_date": str(dataset.target_dates[-1].date()),
            "feature_standardizer_file": (
                standardizer_path.name
                if standardizer_path is not None
                else None
            ),
            "feature_mean": (
                feature_mean.tolist()
                if feature_mean is not None
                else None
            ),
            "feature_std": (
                feature_std.tolist()
                if feature_std is not None
                else None
            ),
            "a_res_definition": "tr(Sigma)/N * shrunk residual correlation",
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print(f"device: {device}")
    print(f"assets: {dataset.n_assets}")
    print(f"market: {dataset.market_name}")
    print(f"standardize inputs: {args.standardize_inputs}")

    if args.standardize_inputs:
        if feature_mean is None or feature_std is None:
            raise RuntimeError("표준화 통계가 없습니다.")

        print(
            "feature standardizer | "
            f"max_abs_mean={feature_mean.abs().max().item():.3e} | "
            f"min_std={feature_std.min().item():.3e} | "
            f"max_std={feature_std.max().item():.3e}"
        )
    print(
        f"samples | train={len(train_dataset)} | validation={len(validation_dataset)} | "
        f"test={len(test_dataset)}"
    )
    print(
        f"alpha={args.alpha} | eta={args.eta} | rho={args.residual_correlation_shrinkage} | "
        f"lambda={args.risk_aversion} | max_weight={args.max_weight}"
    )
    training_result = fit(
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
        gradient_clip_norm=args.gradient_clip_norm,
        metadata=config,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        max_test_batches=args.max_test_batches,
    )

    if args.make_report:
        generate_run_report(
            model=model,
            test_loader=test_loader,
            tickers=dataset.tickers,
            output_dir=output_dir,
            device=device,
            metadata=config,
            best_epoch=training_result.best_epoch,
            best_validation_loss=(
                training_result.best_validation_loss
            ),
            baseline_run_dir=args.baseline_run_dir,
            baseline_label=args.baseline_label,
            active_threshold=args.active_threshold,
            cap_tolerance=args.cap_tolerance,
            periods_per_year=252,
            dpi=args.report_dpi,
        )


if __name__ == "__main__":
    main()
