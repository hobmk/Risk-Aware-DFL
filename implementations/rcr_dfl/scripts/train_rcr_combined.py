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
    RCRRollingMVODataset,
    chronological_split,
)
from implementations.rcr_dfl.src.decision_model import (
    RCRMLPWithMarkowitz,
)
from implementations.rcr_dfl.src.trainer import fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual Collective Risk-Aware DFL training"
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
        "--covariance-jitter",
        type=float,
        default=1e-6,
    )

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
        help="지정하면 CAPM alpha를 0으로 고정합니다.",
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
        default=128,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
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
        "--eta",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "trace"],
        default="trace",
    )
    parser.add_argument(
        "--effective-jitter",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--project-psd",
        action="store_true",
    )
    parser.add_argument(
        "--minimum-eigenvalue",
        type=float,
        default=0.0,
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
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    # Smoke test에서 전체 데이터를 돌리지 않도록 제한할 수 있다.
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
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda를 지정했지만 CUDA를 사용할 수 없습니다."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def prepare_output_dir(
    path: str | Path,
    overwrite: bool,
) -> Path:
    output_dir = Path(path)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"출력 폴더가 이미 존재합니다: {output_dir}\n"
                "덮어쓰려면 --overwrite를 추가하세요."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return output_dir


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(
            "alpha는 0 이상 1 이하여야 합니다."
        )

    if args.eta < 0:
        raise ValueError(
            "eta는 0 이상이어야 합니다."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size는 0보다 커야 합니다."
        )

    if args.num_workers != 0:
        raise ValueError(
            "현재 단계에서는 Windows와 CVXPYLayer 안정성을 위해 "
            "--num-workers 0만 사용하세요."
        )

    set_seed(args.seed)
    device = resolve_device(args.device)

    default_output_dir = (
        Path("implementations/rcr_dfl/outputs")
        / (
            f"eta_{args.eta:.2f}_"
            f"alpha_{args.alpha:.2f}_"
            f"lambda_{args.risk_aversion:.2f}_"
            f"maxw_{args.max_weight:.2f}_"
            f"seed_{args.seed}"
        )
    )

    output_dir = prepare_output_dir(
        args.output_dir or default_output_dir,
        args.overwrite,
    )

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
        dtype=torch.float64,
    )

    train_dataset, validation_dataset, test_dataset = chronological_split(
        dataset=dataset,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )

    generator = torch.Generator().manual_seed(
        args.seed
    )

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
        normalization=args.normalization,
        effective_jitter=args.effective_jitter,
        project_psd=args.project_psd,
        minimum_eigenvalue=args.minimum_eigenvalue,
    ).float()

    optimizer = torch.optim.AdamW(
        model.return_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
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
            "first_target_date": str(
                dataset.target_dates[0].date()
            ),
            "last_target_date": str(
                dataset.target_dates[-1].date()
            ),
        }
    )

    with (
        output_dir / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"device: {device}")
    print(f"assets: {dataset.n_assets}")
    print(f"market: {dataset.market_name}")

    print(
        f"samples | "
        f"train={len(train_dataset)} | "
        f"validation={len(validation_dataset)} | "
        f"test={len(test_dataset)}"
    )

    print(
        f"alpha={args.alpha} | "
        f"eta={args.eta} | "
        f"lambda={args.risk_aversion} | "
        f"max_weight={args.max_weight}"
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
        gradient_clip_norm=args.gradient_clip_norm,
        metadata=config,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        max_test_batches=args.max_test_batches,
    )


if __name__ == "__main__":
    main()