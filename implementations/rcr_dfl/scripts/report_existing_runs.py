from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from implementations.rcr_dfl.src.dataset import (
    RCRRollingMVODataset,
    chronological_split,
    FeatureStandardizedSubset,
)
from implementations.rcr_dfl.src.decision_model import RCRMLPWithMarkowitz
from implementations.rcr_dfl.src.reporting import generate_run_report


RCR_ROOT = Path(
    "implementations/rcr_dfl/outputs/"
    "alpha_lambda_grid_eta050_maxw020_standardized"
)

DFL_ROOT = Path(
    "implementations/dfl_mvo_lee2025/outputs/combined/dow30/"
    "final_full_grid_h64_d0_s30_std"
)

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
LAMBDAS = [0.10, 0.50, 1.00, 2.00, 5.00]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for alpha in ALPHAS:
    for risk_aversion in LAMBDAS:
        run_dir = (
            RCR_ROOT
            / f"alpha_{alpha:.2f}_lambda_{risk_aversion:.2f}"
        )

        if not run_dir.exists():
            raise FileNotFoundError(
                f"RCR run을 찾을 수 없습니다: {run_dir}"
            )

        with (run_dir / "config.json").open(
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        max_weight = float(config["max_weight"])
        seed = int(config["seed"])

        baseline_run_dir = (
            DFL_ROOT
            / f"alpha_{alpha:.2f}"
            / f"lambda_{risk_aversion:.2f}"
            / f"maxw_{max_weight:.2f}"
            / f"seed_{seed}"
        )

        baseline_daily = (
            baseline_run_dir
            / "daily_portfolio.csv"
        )

        if not baseline_daily.exists():
            raise FileNotFoundError(
                f"DFL-MVO baseline이 없습니다: {baseline_daily}"
            )

        print("\n" + "=" * 100)
        print(
            f"Generating report | "
            f"alpha={alpha:.2f} | "
            f"lambda={risk_aversion:.2f}"
        )
        print(f"RCR : {run_dir}")
        print(f"DFL : {baseline_run_dir}")
        print("=" * 100)

        # --------------------------------------------------------
        # RCR Dataset 재구성
        # --------------------------------------------------------
        dataset = RCRRollingMVODataset(
            price_csv=config["price_csv"],
            date_column=config["date_column"],
            return_type=config["return_type"],
            lookback=config["lookback"],
            covariance_jitter=config["covariance_jitter"],
            market_mode=config["market_mode"],
            market_price_csv=config["market_price_csv"],
            market_column=config["market_column"],
            risk_free_rate=config["risk_free_rate"],
            fit_intercept=not config["no_capm_intercept"],
            residual_correlation_shrinkage=config[
                "residual_correlation_shrinkage"
            ],
            correlation_scaling=config["correlation_scaling"],
            dtype=torch.float32,
        )

        _, _, test_subset = chronological_split(
            dataset,
            train_end=config["train_end"],
            validation_end=config["validation_end"],
        )

        # --------------------------------------------------------
        # 학습 시 사용한 입력 표준화 복원
        # --------------------------------------------------------
        if config["standardize_inputs"]:
            standardizer = torch.load(
                run_dir / "feature_standardizer.pt",
                map_location="cpu",
                weights_only=False,
            )

            test_subset = FeatureStandardizedSubset(
                test_subset,
                standardizer["feature_mean"],
                standardizer["feature_std"],
            )

        test_loader = DataLoader(
            test_subset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        # --------------------------------------------------------
        # RCR 모델 복원
        # --------------------------------------------------------
        model = RCRMLPWithMarkowitz(
            n_assets=dataset.n_assets,
            lookback=config["lookback"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            risk_aversion=config["risk_aversion"],
            max_weight=config["max_weight"],
            eta=config["eta"],
            effective_jitter=config["effective_jitter"],
            project_psd=config["project_psd"],
            minimum_eigenvalue=config["minimum_eigenvalue"],
        ).float().to(device)

        checkpoint = torch.load(
            run_dir / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model.eval()

        # --------------------------------------------------------
        # RCR + Equal Weight + 동일 조건 DFL-MVO
        # --------------------------------------------------------
        generate_run_report(
            model=model,
            test_loader=test_loader,
            tickers=list(dataset.tickers),
            output_dir=run_dir,
            device=device,
            metadata=config,
            best_epoch=checkpoint["epoch"],
            best_validation_loss=checkpoint["validation_loss"],
            baseline_run_dir=baseline_run_dir,
            baseline_label="DFL-MVO",
            active_threshold=config.get(
                "active_threshold",
                1e-3,
            ),
            cap_tolerance=config.get(
                "cap_tolerance",
                1e-4,
            ),
            dpi=config.get(
                "report_dpi",
                300,
            ),
        )

print("\nALL 25 BASELINE-COMPARISON REPORTS COMPLETE")